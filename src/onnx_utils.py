import numpy as np

from torch import nn
import torch.onnx
import onnx

from model import AudioCNN

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
                    opset_version=10,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})

def check_onnx(path):
    onnx_model = onnx.load(path)
    onnx.checker.check_model(onnx_model)


if __name__ == '__main__':
    weights_path = "../project/temp/20250506-172951/model.pth"
    
    output_path = "model.onnx"
    
    model = AudioCNN(num_classes=4, in_channels=1)
    
    convert_onnx(model, weights_path, output_path)
    
    check_onnx(output_path)