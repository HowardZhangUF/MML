"""Optional online training for a production particle filter's live history."""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class OnlineLearner:
    def __init__(self, particle_filter):
        self.filter = particle_filter

    def optimize_learned_model_callback(self, event=None):
        """Optimize the neural network model with the particles and weights"""
        pf = self.filter
        filled_elements = np.count_nonzero(pf.state_history) // pf.Nx
        if not np.all(pf.state_history):
            print(f"Filled elements: {filled_elements} / {pf.state_history.shape[0]}")
        # if measurement history buffer is full
        if filled_elements > pf.N_th + pf.max_batch_size * 0.6 and (
            pf.prediction_method in {"NN", "Transformer"}
        ):
            pf.optimize_learned_model(filled_elements)

    def convert_state_history_to_training_batches(
        self, filled_elements, all_data=False
    ):
        """Convert the measurement history to training batches for the neural network
        Input: filled_elements: number of elements in the measurement history buffer
        Output: X_train: input training data, y_train: output training data
        """
        pf = self.filter
        state_history = pf.state_history[-filled_elements:]
        dt_state_history = pf.dt_state_history[-filled_elements:]
        if all_data:
            batch_size = filled_elements - pf.N_th
        else:
            batch_size = np.min((pf.max_batch_size, filled_elements - pf.N_th))
            if batch_size <= 0:
                return None, None
        X_train = np.zeros((batch_size, pf.N_th, pf.Nx))
        dt_time_features = np.zeros((batch_size, pf.N_th))
        y_train = np.zeros((batch_size, pf.Nx))

        # sample n_batches indexes from the data
        if all_data:
            idx = np.arange(batch_size)
        else:
            idx = np.random.choice(
                state_history.shape[0] - pf.N_th, batch_size, replace=False
            )

        # TODO: avoid for loop
        include_index = []
        for i, sampled_i in enumerate(idx):
            X_train[i] = state_history[sampled_i : sampled_i + pf.N_th].copy()
            dt_time_features[i] = (
                dt_state_history[sampled_i : sampled_i + pf.N_th].copy()
                - dt_state_history[sampled_i + pf.N_th]
            )
            y_train[i] = state_history[sampled_i + pf.N_th].copy()

            # remove samples with dt > 4 seconds
            if dt_time_features[i][0] < 4.0:
                include_index.append(i)

        X_train = np.concatenate((X_train, dt_time_features[..., np.newaxis]), axis=2)

        include_index = np.array(include_index)
        print(f"eliminated samples: {batch_size - len(include_index)}")
        X_train = X_train[include_index]
        y_train = y_train[include_index]

        if pf.is_velocity:
            y_train = (y_train - X_train[:, -1, :2]) / X_train[:, -1, 2].reshape(
                -1, 1
            )  # vel = (x(t+1) - x(t)) / dt

        return X_train, y_train

    def optimize_learned_model(self, filled_elements):
        """Optimize the neural network model with the particles and weights"""
        pf = self.filter
        print("\n")
        print("Optimizing the learned model")
        # training parameters
        epochs = 5
        criterion = nn.MSELoss().to(torch.float32)
        optimizer = optim.Adam(pf.motion_model.parameters(), lr=0.001)
        # data transformation
        X_train, y_train = pf.convert_state_history_to_training_batches(
            filled_elements
        )
        if pf.prediction_method == "NN":
            X_train = X_train.reshape(
                X_train.shape[0], -1
            )  # flatten last two dimensions

        X_train = torch.from_numpy(X_train.astype(np.float32)).to(pf.device)
        y_train = torch.from_numpy(y_train.astype(np.float32)).to(pf.device)

        from mml_research.motion.train import train

        losses = train(
            pf.motion_model,
            criterion,
            optimizer,
            X_train,
            y_train,
            epochs,
            online=True,
        )

    def save_model(self, save_path=None):
        """Save the model to a file"""
        pf = self.filter
        if save_path is None:
            save_path = "models/online_model.pth"

        if pf.prediction_method in {
            "NN", "Transformer", "CFM", "CFM-AS", "NF", "Diffusion"
        }:
            torch.save(pf.motion_model.state_dict(), save_path)
            csv_file = save_path.replace(".pth", "_data.csv")
            os.makedirs(os.path.dirname(csv_file), exist_ok=True)
            filled_elements = np.count_nonzero(pf.state_history) // pf.Nx
            X_train, y_train = pf.convert_state_history_to_training_batches(
                filled_elements, all_data=True
            )
            if X_train is None:
                print("No data to save")
                return
            X_train_flattened = X_train.reshape(X_train.shape[0], -1)
            state_history_flattened = np.concatenate(
                (X_train_flattened, y_train, np.zeros((X_train_flattened.shape[0], 1))),
                axis=1,
            )
            np.savetxt(csv_file, state_history_flattened, delimiter=",")
            print(f"Data saved to {csv_file}")
        elif pf.prediction_method == "DMMN":
            pf.motion_model.saveModel(time=time.time(), savePath=save_path)
        print(f"Model saved to {save_path}")
