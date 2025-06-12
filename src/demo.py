import os
import sys
import time
import serial
import numpy as np
import torch
import torchaudio.transforms as T
import onnx
from onnx import numpy_helper
import onnxruntime as ort
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel
import sounddevice as sd

from config import config
from dataloader import SpeechCommandsDataset

from dotenv import load_dotenv
load_dotenv()
dataset_path = os.getenv("DATASET_PATH")


# === Config ===
model_path = "reports/20250611-120913/model_quantized.onnx"
port = 'COM4'
baudrate = 115200
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
classes_to_str = dataset.classes


# === Load ONNX model and quantization params ===
model = onnx.load(model_path)
input_name = model.graph.input[0].name
for node in model.graph.initializer:
    if "scale" in node.name and input_name in node.name:
        scale = numpy_helper.to_array(node)
    if "zero_point" in node.name and input_name in node.name:
        zero_point = numpy_helper.to_array(node)

def quantize_sample(sample, scale, zero_point):
    return np.clip(np.round(sample / scale + zero_point), -128, 127).astype(np.int8)


# === UI App ===
class KeywordSpottingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.ser = serial.Serial(port, baudrate, timeout=1)
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Keyword Spotting Demo')
        layout = QVBoxLayout()

        self.status_label = QLabel("Press to record a 1s sample")
        layout.addWidget(self.status_label)

        self.button = QPushButton("🎙 Record")
        self.button.clicked.connect(self.record_and_send)
        layout.addWidget(self.button)

        self.setLayout(layout)
        self.resize(300, 120)

    def record_and_send(self):
        self.status_label.setText("Recording...")
        QApplication.processEvents()

        # === Record audio
        duration = 1.0  # seconds
        sr = config['sampling_rate']
        audio = sd.rec(int(sr * duration), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
        waveform = torch.from_numpy(audio.T)

        # === Extract MFCC
        sample = transform(waveform).numpy().astype(np.float32)

        # === Send to STM32
        quantized = quantize_sample(sample.reshape(-1), scale, zero_point)
        self.ser.write(b'S')
        self.ser.write(quantized.tobytes())
        self.ser.flush()

        # === Wait for STM32 response
        while self.ser.in_waiting == 0:
            pass
        line = self.ser.readline().decode('utf-8').strip()
        if line:
            pred_mcu = int(line.split()[1])
            result = f"[MCU]: {classes_to_str[pred_mcu]}"
            self.status_label.setText(result)

    def closeEvent(self, event):
        self.ser.close()
        event.accept()


# === Main ===
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = KeywordSpottingApp()
    window.show()
    sys.exit(app.exec_())
