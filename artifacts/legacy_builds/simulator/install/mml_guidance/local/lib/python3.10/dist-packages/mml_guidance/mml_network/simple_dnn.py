import numpy as np
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"


class SimpleDNN(nn.Module):
    def __init__(
        self,
        input_size,
        num_layers,
        nodes_per_layer,
        output_size=2,
        activation_fn="relu",
    ):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Linear(input_size, nodes_per_layer))
            else:
                self.layers.append(nn.Linear(nodes_per_layer, nodes_per_layer))
        self.layers.append(nn.Linear(nodes_per_layer, output_size))
        if activation_fn == "relu":
            self.activation_fn = nn.ReLU()
        elif activation_fn == "sigmoid":
            self.activation_fn = nn.Sigmoid()
        else:
            raise ValueError("Invalid activation function")
        self.layers.to(device)
        # for layer in self.layers:
        # layer.float()

        # move weights to device 
        self.to(device)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation_fn(layer(x))
        return self.layers[-1](x)

    def predict(self, x):
        with torch.no_grad():
            return (
                self.forward(torch.from_numpy(x.astype(np.float32)).to(device))
                .cpu()
                .detach()
                .numpy()
            )
