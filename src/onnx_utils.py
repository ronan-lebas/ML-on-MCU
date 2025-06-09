import numpy as np
from torch import nn
import torch.onnx
from torch.utils.data import DataLoader
import torchaudio.transforms as T
import onnx
from onnxruntime.quantization import quantize_static, QuantFormat, CalibrationMethod, CalibrationDataReader, QuantType
import csv
from tqdm import tqdm
from dotenv import load_dotenv
import os

from model import AudioCNN
from dataloader import SpeechCommandsDataset
from config import config

load_dotenv()
dataset_path = os.getenv("DATASET_PATH")


def convert_onnx(model, weights_path, output_path):
    batch_size = 64
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()
    
    # Input to the model
    x = torch.randn(batch_size, 1, 40, 51, requires_grad=True)
    torch_out = model(x)

    # Export the model
    torch.onnx.export(model,               # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    output_path,   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=13,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})


def check_onnx(path):
    onnx_model = onnx.load(path)
    onnx.checker.check_model(onnx_model)


def export_csv(
    dataset,
    output_csv_path,
    num_samples
):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    input_list = []
    output_list = []

    for i, (inputs, labels) in enumerate(tqdm(dataloader, total=num_samples, desc="Exporting CSVs")):
        if i >= num_samples:
            break

        # Flatten the MFCC tensor (1, n_mfcc, time) → 1D row
        flat = inputs.squeeze(0).numpy().flatten()
        input_list.append(flat)
        one_hot_label = np.zeros(len(dataset.classes), dtype=np.float32)
        one_hot_label[labels.item()] = 1.0
        output_list.append(one_hot_label)

    input_path = output_csv_path + "/inputs.csv"
    output_path = output_csv_path + "/outputs.csv"
    
    with open(input_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        for row in input_list:
            writer.writerow(row)
    with open(output_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        for row in output_list:
            writer.writerow(row)
    
    print(f"✅ Exported {num_samples} samples to: {input_path} and {output_path}")


class DataReader(CalibrationDataReader):
    def __init__(self, dataset, num_samples):
        self.dataset = dataset
        self.num_samples = num_samples
        self.enum_data = None
        self.preprocess()

    def preprocess(self):
        self.enum_data = []
        for i in range(min(self.num_samples, len(self.dataset))):
            x, _ = self.dataset[i]
            x = x.unsqueeze(0).numpy().astype(np.float32)
            self.enum_data.append({"input": x})
        self.enum_data = iter(self.enum_data)

    def get_next(self):
        return next(self.enum_data, None)


def quantize_onnx(input_path, output_path, dataset, quantization_type=QuantType.QInt8):
    reader = DataReader(dataset, num_samples=500)
    quantize_static(
        model_input=input_path,
        model_output=output_path,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        weight_type=quantization_type,
        activation_type=quantization_type,
        calibrate_method=CalibrationMethod.MinMax
    )


if __name__ == '__main__':
    folder = "temp/20250601-154841"
    weights_path = folder + "/model.pth"
    onnx_path = folder + "/model.onnx"
    quantized_path = folder + "/model_quantized.onnx"
    
    model = AudioCNN(num_classes=len(config['only_classes']), in_channels=1)
    transform = T.MFCC(
        sample_rate=config['sampling_rate'],
        n_mfcc=config['n_mfcc'],
        melkwargs=config['melkwargs']
    )
    
    val_dataset = SpeechCommandsDataset(
                dataset_path,
                "VAL",
                sampling_rate=config['sampling_rate'],
                transform=transform,
                max_sample_per_class=config['max_sample_per_class'],
                only_classes=config['only_classes']
            )
    
    test_dataset = SpeechCommandsDataset(
                dataset_path,
                "TEST",
                sampling_rate=config['sampling_rate'],
                transform=transform,
                max_sample_per_class=config['max_sample_per_class'],
                only_classes=config['only_classes']
            )
    
    convert_onnx(model, weights_path, onnx_path)
    check_onnx(onnx_path)
    print("ONNX model converted and checked successfully.")
    
    quantize_onnx(onnx_path, quantized_path, val_dataset, quantization_type=QuantType.QInt8)
    check_onnx(quantized_path)
    print("ONNX model quantized and checked successfully.")
    
    export_csv(
        dataset=test_dataset,
        output_csv_path=folder,
        num_samples=100
    )
    print("CSVs for on-device test exported successfully.")