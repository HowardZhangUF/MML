import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import os


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=10):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=100, dropout=0.01):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=100, dropout=0.01):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=10)

    def forward(self, src):
        output = self.pos_encoder(src)
        for layer in self.layers:
            output = layer(output)
        return output


class TransformerModel(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        num_layers,
        dim_feedforward=100,
        dropout=0.01,
        output_size=2,
    ):
        super().__init__()
        self.input_projection = nn.Linear(2, d_model)
        self.transformer = TransformerEncoder(
            d_model, nhead, num_layers, dim_feedforward, dropout
        )
        self.output_projection = nn.Linear(d_model, output_size)

    def forward(self, x):
        batch_size, sequence_length = x.shape
        x = x.view(
            batch_size, sequence_length // 2, 2
        )  # Reshape to (batch_size, 10, 2)
        x = self.input_projection(x)
        x = self.transformer(x)
        x = self.output_projection(x[:, -1, :])  # Use the output of the last time step
        return x

    def predict(self, x):
        return self.forward(torch.from_numpy(x.astype(np.float32))).detach().numpy()


def main():
    from simple_dnn import train, evaluate, parameter_search, plot_NN_output

    is_velocities = False
    path = os.path.expanduser("~/mml_ws/src/mml_guidance/hardware_data/")
    # print the files in the directory
    if is_velocities:
        df_vel = pd.read_csv(path + "converted_velocities_training_data.csv")
    df = pd.read_csv(path + "converted_training_data.csv")
    X = df.iloc[:, :-2].values if not is_velocities else df_vel.iloc[:, :-2].values
    y = df.iloc[:, -2:].values if not is_velocities else df_vel.iloc[:, -2:].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train = torch.from_numpy(X_train.astype(np.float32))
    X_test = torch.from_numpy(X_test.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32))
    y_test = torch.from_numpy(y_test.astype(np.float32))

    # Define the hyperparameters to tune
    param_grid = {
        "d_model": [16],
        "nhead": [1],
        "num_layers": [2],
        "dim_feedforward": [10],
    }

    ModelClass = TransformerModel

    weights_filename = "transformer.pth"

    best_params = parameter_search(
        ModelClass,
        param_grid,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=40,
        weights_filename=weights_filename,
    )

    model = TransformerModel(**best_params)
    # load the saved weights
    model.load_state_dict(torch.load(weights_filename))

    criterion = nn.MSELoss()
    # Evaluation
    with torch.no_grad():
        y_pred = model(X_test)
        test_loss = criterion(y_pred, y_test)
        print(f"Test Loss: {test_loss.item()}")

    plot_NN_output(X_test, y_pred, y_test, is_velocities, df)
    return 0


if __name__ == "__main__":
    main()
