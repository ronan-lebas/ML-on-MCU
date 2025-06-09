import time
import serial
import serial.tools.list_ports
import numpy as np
import torch
import torchaudio.transforms as T
import os
import onnx
from onnx import numpy_helper
import onnxruntime as ort

from config import config
from dataloader import SpeechCommandsDataset

from dotenv import load_dotenv
load_dotenv()
dataset_path = os.getenv("DATASET_PATH")


# === Utils functions
def run_onnx_inference(model_path, input):
    sess = ort.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    result = sess.run([output_name], {input_name: input[np.newaxis, :]})  # Add batch dim
    return np.argmax(result[0])  # Return predicted class


def quantize_sample(sample, scale, zero_point):
    # Clamp before casting
    quantized = np.clip(np.round(sample / scale + zero_point), -128, 127).astype(np.int8)
    return quantized


# === Load data
transform = T.MFCC(
    sample_rate=config['sampling_rate'],
    n_mfcc=config['n_mfcc'],
    melkwargs=config['melkwargs']
)

dataset = SpeechCommandsDataset(
        dataset_path,
        "TEST",
        sampling_rate=config['sampling_rate'],
        transform=transform,
        max_sample_per_class=config['max_sample_per_class'],
        only_classes=config['only_classes']
    )

model_path = os.path.normpath("reports/20250601-154841/model_quantized.onnx")

model = onnx.load(model_path)
input_tensor = model.graph.input[0]
input_name = input_tensor.name

for node in model.graph.initializer:
    if "scale" in node.name and input_name in node.name:
        scale = numpy_helper.to_array(node)
    if "zero_point" in node.name and input_name in node.name:
        zero_point = numpy_helper.to_array(node)

print("Input scale:", scale)
print("Input zero point:", zero_point)


# === Find ports
print("Available ports:")
ports = list(serial.tools.list_ports.comports())
for p in ports:
    print(p)

port = 'COM4'
baudrate = 115200


# === Open serial port
ser = serial.Serial(port, baudrate, timeout=1)
print(f"Connected to {port} at {baudrate} baud.")

print("Listening on", ser.port)
try:
    while True:
        idx = np.random.randint(0, len(dataset))
        sample, label = dataset[idx]
        sample = sample.numpy().astype(np.float32)
        sample_quantized = quantize_sample(sample.reshape(-1), scale, zero_point)
        ser.write(b'S')  # Trigger byte
        ser.write(sample_quantized.tobytes())
        ser.flush()
        print(f"Sent sample {idx}, true label: {dataset.classes[label]} ({label}), predicted label by the quantized model on PC: {run_onnx_inference(model_path, sample)}")
        while ser.in_waiting == 0:
            pass
        line = ser.readline().decode('utf-8').strip()
        if line:
            print("[STM32] >", line)
            pred = int(line.split()[1])
            question = "\033[38;2;255;255;0mIs the model on device right?\033[0m"
            if pred == label:
                print(question + ' ' + "\033[38;2;0;255;0mYes\033[0m")
            else:
                print(question + ' ' + "\033[38;2;255;0;0mNo\033[0m")
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nExiting...")
    ser.close()
