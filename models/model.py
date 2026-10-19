from pathlib import Path
import torch
import torch.nn as nn
import onnx
from onnx import numpy_helper


def load_model():
    layers = [nn.Flatten(), nn.Linear(784, 100), nn.ReLU(),
              nn.Linear(100, 100), nn.ReLU(),
              nn.Linear(100, 100), nn.ReLU(),
              nn.Linear(100, 10)]

    model = nn.Sequential(*layers)
    path = Path(__file__).with_name("mnist_relu_3_100.onnx")
    network = onnx.load(str(path))

    weights = {
        tensor.name: torch.tensor(numpy_helper.to_array(tensor).copy())
        for tensor in network.graph.initializer
    }

    linear_layers = [layer for layer in model if isinstance(layer, nn.Linear)]
    gemm_nodes = [node for node in network.graph.node if node.op_type == "Gemm"]

    with torch.no_grad():
        for layer, node in zip(linear_layers, gemm_nodes):
            weight = weights[node.input[1]]
            bias = weights[node.input[2]]
            layer.weight.copy_(weight if layer.weight.shape == weight.shape else weight.T)
            layer.bias.copy_(bias)

    model.eval()
    return model