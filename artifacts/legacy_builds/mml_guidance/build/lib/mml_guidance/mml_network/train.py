#!/usr/bin/env python3
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
import os

try:
    # for running the script from the scripts directory
    from .simple_dnn import SimpleDNN
    from .scratch_transformer import ScratchTransformer
except ImportError:
    # for running the script with ROS
    from mml_network.simple_dnn import SimpleDNN
    from mml_network.scratch_transformer import ScratchTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"


def train(
    model,
    criterion,
    optimizer,
    X_train,
    y_train,
    epochs=100,
    online=False,
    is_velocities=False,
    is_time=False,
):
    losses = []
    for epoch in range(epochs):
        outputs = model(X_train)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Training Loss: {loss.item()}")

    if online:
        X_train = X_train.to("cpu").detach().numpy()
        outputs = outputs.to("cpu").detach().numpy()
        y_train = y_train.to("cpu").detach().numpy()
        plot_NN_output(
            X_train.reshape(X_train.shape[0], -1),
            outputs.reshape(X_train.shape[0], -1),
            y_train.reshape(X_train.shape[0], -1),
            is_velocities=is_velocities,
            is_time=is_time,
            title="Online",
        )
    return losses


def evaluate(model, criterion, X_test, y_test):
    """Evaluate the model on the test set. Return the loss and the outputs."""
    with torch.no_grad():
        outputs = model.forward(X_test)
        loss = criterion(outputs, y_test)
    return loss.item(), outputs


def parameter_search(
    ModelClass, param_grid, X_train, y_train, X_test, y_test, epochs, weights_filename
):
    grid = ParameterGrid(param_grid)

    best_loss = float("inf")
    best_params = None

    # list of keys that have more than one value in the grid
    changing_keys = [key for key in param_grid.keys() if len(param_grid[key]) > 1]

    for params in grid:
        print("Training with parameters: ", params)

        if "n_head" in list(params.keys()):
            # check that n_embed is larger than n_head and n_embed is divisible by n_head
            if params["n_embed"] < params["n_head"]:
                print("n_embed must be larger than n_head")
                continue

        # initialize the model with the given parameters, make it so that it can take any parameter for any model
        model = ModelClass(**params)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        losses = train(model, criterion, optimizer, X_train, y_train, epochs)
        print("Training Loss: ", losses[-1])
        test_loss, _ = evaluate(model, criterion, X_test, y_test)
        print("Test Loss: ", test_loss, "\n")
        if test_loss < best_loss:
            best_loss = test_loss
            best_params = params
            torch.save(model.state_dict(), weights_filename)

        params = {key: params[key] for key in changing_keys}
        plt.plot(
            losses,
            label=f"{params}",
        )

    print(f"\nBest parameters: {best_params}")
    print(f"Best test loss: {best_loss}")
    print("Model saved to: ", weights_filename)

    plt.legend(bbox_to_anchor=(1, 1))
    plt.show()

    return best_params


def plot_NN_output(
    features, predictions, true_label, is_velocities, is_time, title="Test"
):
    # font sizes for the plot labels
    sns.set_theme()
    sns.set_style("white")
    sns.set_style({"font.family": "Times New Roman"})
    sns.set_context("paper", font_scale=2)

    num_plots = 2 if title == "Test" else 1
    nx = 3 if is_time else 2

    for k in range(num_plots):

        # figure with legends
        fig, ax = plt.subplots()

        occ_widths = [0.5, 0.6, 0.6, 0.6]
        occ_centers = [[-1.5, 1], [-0.75, 0.0], [1.1, 1], [1.5, 0.0]]
        # [x_min, x_max, y_min, y_max]
        occlusions = [
            [
                occ_centers[i][0] - occ_widths[i] / 2,
                occ_centers[i][0] + occ_widths[i] / 2,
                occ_centers[i][1] - occ_widths[i] / 2,
                occ_centers[i][1] + occ_widths[i] / 2,
            ]
            for i in range(len(occ_centers))
        ]
        # plot the squares representing occlusions
        for occlusion in occlusions:
            plt.plot(
                [occlusion[0], occlusion[1], occlusion[1], occlusion[0], occlusion[0]],
                [occlusion[2], occlusion[2], occlusion[3], occlusion[3], occlusion[2]],
                color="black",
            )
        # legend
        plt.plot(
            occlusion[0],
            occlusion[2],
            color="black",
            label="Occlusion",
        )

        clip_arrow = 1  # clip the arrow length
        if is_velocities:
            pos = features[:, -3:-1] if is_time else features[:, -2:]
            gain = 0.3
            # time step
            if is_time:
                arrow_length = np.clip(features[:, -1] * gain, -clip_arrow, clip_arrow)
            else:
                arrow_length = np.ones(features.shape[0]) * 0.333 * gain
            for i in range(pos.shape[0] - 1):
                plt.arrow(
                    pos[i, -2],
                    pos[i, -1],
                    predictions[i, -2] * arrow_length[i],
                    predictions[i, -1] * arrow_length[i],
                    color="red",
                    alpha=0.4,
                    width=0.003,
                    head_width=0.015,
                )
                if k == 0:
                    plt.arrow(
                        pos[i, -2],
                        pos[i, -1],
                        true_label[i, 0] * arrow_length[i],
                        true_label[i, 1] * arrow_length[i],
                        color="blue",
                        alpha=0.4,
                        width=0.003,
                        head_width=0.015,
                    )
        else:
            # Calculate arrow directions for predictions and actual values (position - prev_position)
            gain = 0.2
            pred_dx = np.clip(
                (predictions[:, 0] - features[:, -nx]) * gain,
                -clip_arrow / 10,
                clip_arrow / 10,
            )
            pred_dy = np.clip(
                (predictions[:, 1] - features[:, -nx + 1]) * gain,
                -clip_arrow / 10,
                clip_arrow / 10,
            )
            actual_dx = np.clip(
                (true_label[:, 0] - features[:, -nx]) * gain,
                -clip_arrow / 10,
                clip_arrow / 10,
            )
            actual_dy = np.clip(
                (true_label[:, 1] - features[:, -nx + 1]) * gain,
                -clip_arrow / 10,
                clip_arrow / 10,
            )

            arrow_length = 2.5  # higher value means shorter arrows
            if k == 0:
                # Plot actual arrows
                plt.quiver(
                    features[:, -nx],
                    features[:, -nx + 1],
                    actual_dx,
                    actual_dy,
                    color="blue",
                    alpha=0.7,
                    scale=arrow_length,
                    headwidth=1.5,  # Increase the width of the arrowhead
                    width=0.0025,
                )
            # Plot predicted arrows
            plt.quiver(
                features[:, -nx],
                features[:, -nx + 1],
                pred_dx,
                pred_dy,
                color="red",
                alpha=0.7,
                scale=arrow_length,
                headwidth=1.5,  # Increase the width of the arrowhead
                width=0.0025,
            )

        # Plot 3 paths
        for i in range(-1, -1 * min(4, features.shape[0]), -1):
            ax.plot(
                features[i, 0 : -nx + 1 : nx],
                features[i, 1:-1:nx] if nx == 3 else features[i, 1::nx],
                color="black",
                marker="o",
                linewidth=1,
                alpha=0.5,
            )
        # label for legend of path
        plt.plot(
            features[0, 0],
            features[0, 1],
            color="black",
            marker="o",
            label="Path History Sample",
            linewidth=1,
        )
        # for legend
        plt.arrow(0, 0, 0, 0, color="red", label="Predicted")
        plt.arrow(0, 0, 0, 0, color="blue", label="Actual") if k == 0 else None

        plt.title(title)
        plt.xlabel("$x_g$ [m]")
        plt.ylabel("$y_g$ [m]")
        # equal aspect ratio
        plt.gca().set_aspect("equal", adjustable="box")
        # plt.legend(bbox_to_anchor=(1, 1))
        font_properties = FontProperties()
        font_properties.set_family("serif")
        font_properties.set_name("Times New Roman")
        ax.legend(prop=font_properties, bbox_to_anchor=(1, 1))

        plt.legend(loc="lower left")
        if title == "Online":
            plt.savefig(
                "/home/andres/mml_ws/src/mml_guidance/sim_data/online_network_test_output.png",
                bbox_inches="tight",
            )
        plt.show()
        plt.close(fig)


def initialize_weights(model):
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)


def select_model(model_name, input_size, input_dim=2):
    if model_name == "SimpleDNN":
        model = SimpleDNN(
            input_size=input_size,
            num_layers=4,
            nodes_per_layer=40,
            output_size=2,
            activation_fn="relu",
        )
    elif model_name == "ScratchTransformer":
        block_size = input_size // input_dim
        model = ScratchTransformer(
            input_dim=input_dim,
            block_size=block_size,
            n_embed=40,
            n_head=6,
            n_layer=1,
            hidden_dim=10,
        )
    else:
        raise ValueError("Invalid model name")

    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
    model = model.to(device)

    # Set the random seed for reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Reinitialize the weights to ensure they are random
    initialize_weights(model)

    return model


def select_parameters(model_name, input_size, input_dim=3):
    if model_name == "SimpleDNN":
        param_grid = {
            "input_size": [input_size],
            "output_size": [2],
            "activation_fn": ["relu"],
            "num_layers": [2, 4, 8, 16],
            "nodes_per_layer": [20, 40, 80],
        }
    elif model_name == "ScratchTransformer":
        param_grid = {
            "hidden_dim": [20, 30],
            "n_embed": [20, 40],  # 40 also good
            "n_head": [6, 8],  # 8 also good
            "n_layer": [1, 2],
            "input_dim": [input_dim],
        }
    else:
        raise ValueError("Invalid model name")

    return param_grid


def main():
    is_velocities = True
    is_occlusion = True
    is_time = False
    is_perfect = False
    is_parameter_search = True
    evaluate_model = False
    is_connected_graph = False
    get_every_n = 1
    model_name = "ScratchTransformer"  # "ScratchTransformer"  # "SimpleDNN"
    prefix_name = "noisy_5_v2_" if not is_perfect else "perfect_5_v2_"

    print(
        f"Training NN w is_velocities: {is_velocities}, model: {model_name}, is_parameter_search: {is_parameter_search}, prefix_name: {prefix_name}"
    )

    path = os.path.expanduser("~/mml_ws/src/mml_guidance/sim_data/training_data/")

    if is_time:
        prefix_name = prefix_name + "time_"
        input_dim = 3  # 2 for positions, 1 for time
    else:
        input_dim = 2

    prefix_name = (
        prefix_name + "velocities_" if is_velocities else prefix_name + "position_"
    )

    if is_occlusion or evaluate_model:
        df_no_occlusion = pd.read_csv(
            path + "converted_" + prefix_name + "training_data.csv"
        )  # non-occluded
        prefix_name = prefix_name + "occlusions_" if is_occlusion else prefix_name
        print("no occlusion df shape: ", df_no_occlusion.shape)

    prefix_name = (
        "connected_graph_" + prefix_name if is_connected_graph else prefix_name
    )

    df = pd.read_csv(path + "converted_" + prefix_name + "training_data.csv")
    # df = pd.read_csv(path + "online_data.csv")
    print("df shape: ", df.shape)
    X = df.iloc[::get_every_n, :-input_dim].values
    if input_dim == 3:
        slice_ = slice(-input_dim, -1)
    else:
        slice_ = slice(-input_dim, None)
    Y = df.iloc[::get_every_n, slice_].values
    print("shape of X: ", X.shape)
    print("shape of Y: ", Y.shape)

    train_percent = 0.7
    epochs = 100

    def get_ordered_train_test(data):
        length = len(data)
        data_train, data_test = (
            data[math.floor(length * (1 - train_percent)) :],
            data[: math.floor(length * (1 - train_percent))],
        )
        return data_train, data_test

    X_train, X_test = get_ordered_train_test(X)  # (N, T*C)
    y_train, y_test = get_ordered_train_test(Y)
    if is_occlusion or evaluate_model:
        train_percent = 0.7
        _, X_test = get_ordered_train_test(
            df_no_occlusion.iloc[::get_every_n, :-input_dim].values
        )
        if is_time:
            _, y_test = get_ordered_train_test(
                df_no_occlusion.iloc[::get_every_n, -input_dim:-1].values
            )
        else:
            _, y_test = get_ordered_train_test(
                df_no_occlusion.iloc[::get_every_n, -input_dim:].values
            )

    # split the data into training and testing sets
    X_train = torch.from_numpy(X_train.astype(np.float32)).to(device)
    X_test = torch.from_numpy(X_test.astype(np.float32)).to(device)
    y_train = torch.from_numpy(y_train.astype(np.float32)).to(device)
    y_test = torch.from_numpy(y_test.astype(np.float32)).to(device)
    print("Train size: ", X_train.shape, y_train.shape)
    print("Test size: ", X_test.shape, y_test.shape)

    criterion = nn.MSELoss()

    model = select_model(
        model_name=model_name, input_size=X_train.shape[1], input_dim=input_dim
    )

    weights_filename = (
        f"../../scripts/mml_network/models/best_{prefix_name}{model_name}.pth"
    )
    if is_parameter_search:
        param_grid = select_parameters(
            model_name=model_name, input_size=X_train.shape[1], input_dim=input_dim
        )

        ModelClass = model.__class__

        best_params = parameter_search(
            ModelClass,
            param_grid,
            X_train,
            y_train,
            X_test,
            y_test,
            epochs,
            weights_filename=weights_filename,
        )
        model = ModelClass(**best_params)
        model.load_state_dict(torch.load(weights_filename, weights_only=True))
    else:
        optimizer = optim.Adam(model.parameters(), lr=4e-3)
        losslist = train(model, criterion, optimizer, X_train, y_train, epochs=epochs)

        # plot the loss
        plt.plot(losslist)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

        torch.save(model.state_dict(), weights_filename)
        print("Model saved to: ", weights_filename)

    model.eval()

    # evaluate training data
    train_loss, y_pred_train = evaluate(model, criterion, X_train, y_train)
    y_pred_train = y_pred_train.to("cpu").detach().numpy()
    y_train = y_train.to("cpu").detach().numpy()
    X_train = X_train.to("cpu").detach().numpy()
    print("Train Loss: ", train_loss)

    plot_NN_output(
        X_train[::2], y_pred_train[::2], y_train[::2], is_velocities, is_time, "Train"
    )

    # evaluate test data
    test_loss, y_pred = evaluate(model, criterion, X_test, y_test)
    print("Test Loss: ", test_loss)
    X_test = X_test.to("cpu").detach().numpy()
    y_pred = y_pred.to("cpu").detach().numpy()
    y_test = y_test.to("cpu").detach().numpy()
    plot_NN_output(
        X_test[::2], y_pred[::2], y_test[::2], is_velocities, is_time, "Test"
    )

    if evaluate_model and False:
        # weights_filename = f"../../scripts/mml_network/models/best_connected_graph_noisy_velocities_ScratchTransformer.pth"
        weights_filename = f"../../scripts/mml_network/models/online_model.pth"
        model.load_state_dict(torch.load(weights_filename, weights_only=True))
        model.eval()
        X_test = torch.from_numpy(X_test.astype(np.float32)).to(device)
        y_test = torch.from_numpy(y_test.astype(np.float32)).to(device)
        test_loss, y_pred = evaluate(model, criterion, X_test, y_test)
        print("Online Model Test Loss: ", test_loss)
        X_test = X_test.to("cpu").detach().numpy()
        y_pred = y_pred.to("cpu").detach().numpy()
        y_test = y_test.to("cpu").detach().numpy()
        plot_NN_output(X_test, y_pred, y_test, is_velocities, "Test")
    return 0


if __name__ == "__main__":
    main()
