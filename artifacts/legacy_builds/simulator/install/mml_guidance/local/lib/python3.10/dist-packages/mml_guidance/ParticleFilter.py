#!/usr/bin/env python3
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time

from .mml_network.motion_model_network import MotionModel
from .mml_network.simple_dnn import SimpleDNN
from .mml_network.scratch_transformer import ScratchTransformer
from .mml_network.train import train

class ParticleFilter:
    def __init__(
        self,
        num_particles=10,
        prediction_method="NN",
        drone_height=2.0,
        model_path=None,
        initial_time=0.0,
        initial_detection=None,
    ):
        # Initialize variables
        deg2rad = lambda deg: np.pi * deg / 180
        if prediction_method in {"MultiKF", "MultiPFVel"}:
            prediction_method = "Velocity"  # temporary fix
        elif prediction_method in {"MultiPFTra"}:
            prediction_method = "Transformer"  # temporary fix
        elif prediction_method not in {"NN", "Transformer", "Unicycle", "Velocity", "KF", "DMMN"}:
            raise ValueError(
                f"Invalid prediction method: {prediction_method}. Choose from 'NN', 'Transformer', 'Unicycle', 'Velocity', 'KF', or 'DMMN'."
            )
        self.prediction_method = prediction_method
        # boundary of the lab [[x_min, y_min], [x_max, y_,max]] [m]
        self.APRILab_dims = np.array([[-3.0, -3.0], [3.0, 3.0]])  # sim

        if self.prediction_method in {"NN", "Transformer"}:
            self.N_th = 5  # Number of time history particles
            self.is_velocity = True
            is_occlusions_weights = True
            self.is_time_weights = False
            input_dim = 3 if self.is_time_weights else 2  # x, y, time
            self.nn_input_size = self.N_th * input_dim
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            prefix_name = "noisy_5_v2_"  # "connected_graph_noisy_5_" or "noisy_5_" for pretrained on road network
            prefix_name = prefix_name + "time_" if self.is_time_weights else prefix_name
            prefix_name = (
                prefix_name + "velocities_"
                if self.is_velocity
                else prefix_name + "position_"
            )
            prefix_name = (
                prefix_name + "occlusions_" if is_occlusions_weights else prefix_name
            )

            if self.prediction_method == "NN":
                prefix_name = prefix_name + "SimpleDNN"
                self.motion_model = SimpleDNN(
                    input_size=self.nn_input_size,
                    nodes_per_layer=80,
                    num_layers=2,
                    output_size=2,
                    activation_fn="relu",
                )
            elif self.prediction_method == "Transformer":
                prefix_name = prefix_name + "ScratchTransformer"
                self.motion_model = ScratchTransformer(
                    input_dim=input_dim,
                    block_size=self.N_th,
                    n_embed=20,
                    n_head=8,
                    n_layer=2,
                    hidden_dim=20,
                )

            # Load model if path is provided
            if model_path:
                try:
                    self.motion_model.load_state_dict(
                        torch.load(model_path, map_location=self.device)
                    )
                except FileNotFoundError:
                    print(f"Error: The file '{model_path}' does not exist. Change to the correct directory.")
            else:
                print("No model path provided for NN/Transformer. Using untrained model.")
        else:
            self.N_th = 2

        # PF
        self.N = num_particles
        self.weights = np.ones(self.N) / self.N
        self.prev_weights = np.copy(self.weights)
        self.weighted_mean = np.array([0, 0, 0])
        self.is_update = False
        self.is_occlusion = False
        self.neff = self.nEff(self.weights)

        if self.prediction_method == "Velocity":
            self.Nx = 4  # number of states [x, y, vx, vy]
            self.vmax = 0.4  # m/s maximum velocity
            self.best_measurement_covariance = np.diag([0.006, 0.006])
            # Process noise parameters - improved from reference code
            self.proc_pos_std = 0.05  # m per sqrt(s) - position process noise
            self.proc_vel_std = 0.15   # m/s per sqrt(s) - velocity process noise
            self.process_covariance = np.diag([self.proc_pos_std**2, self.proc_pos_std**2, 
                                             self.proc_vel_std**2, self.proc_vel_std**2])
        elif self.prediction_method == "Unicycle":
            self.Nx = 3
            self.best_measurement_covariance = np.diag([0.01, 0.01, deg2rad(5)])
            self.process_covariance = np.diag([0.8, 0.8, deg2rad(5.)])
        elif self.prediction_method in {"KF", "DMMN"}:
            self.Nx = 2
            self.best_measurement_covariance = np.diag([0.006, 0.006])
            self.process_covariance = np.diag([0.0001, 0.0001, 0.00001, 0.00001])

        if self.prediction_method == "DMMN":
            inputSize = 2  # number of inputs to NN
            gamma = 0.1  # a learning for online learning
            k1 = 4.0  # a learning for online learning
            # NN parameters
            numberHiddenLayers = 2  # number of hidden layers
            hiddenSize = 6  # size of each hidden layer
            outputSize = 5  # size of the output of the NN
            probHiddenDrop = 0.0  # dropout probability for hidden layers
            useAttention = False  # use attention at output layer

            # learning parameters
            alpha = 0.001  # learning rate for SGD
            memorySize = 150  # number of data points to save for each training
            batchSize = 20  # size of a batch for each training epoch, should be multiple of memory size
            numberEpochs = 25  # 25  # number of iterations for each training
            minDistance = 0.005  # minimum distance between any two data points

            self.motion_model = MotionModel(
                alpha,
                numberHiddenLayers,
                inputSize,
                hiddenSize,
                outputSize,
                probHiddenDrop,
                useAttention,
                memorySize,
                batchSize,
                numberEpochs,
                minDistance,
                gamma,
                k1,
            )

            # Load model if path is provided
            if model_path:
                self.motion_model.loadModel(model_path)

        if self.prediction_method in {"NN", "Transformer"}:
            self.Nx = 2
            self.best_measurement_covariance = np.diag([0.006, 0.006])
            self.process_covariance = np.diag([0.001, 0.001])
            buffer_size = (
                300  # large but not infinite to not have dynamic memory allocation
            )
            self.max_batch_size = 64
            assert (
                self.N_th + self.max_batch_size < buffer_size
            ), "Buffer size too small"
        else:
            buffer_size = 1

        self.update_measurement_covariance(height=drone_height)

        self.state_history = np.zeros((self.N_th + buffer_size, self.Nx))
        self.dt_history = np.ones(self.N_th) * 0.1  # for prediction
        self.dt_state_history = np.zeros(self.state_history.shape[0])  # for creating history to optimize
        # Use multivariate normal if you know the initial condition
        self.particles = self.uniform_sample()
        if initial_detection is not None:
            self.gaussian_reset(initial_detection)

        # Process noise: q11, q22 is meters of error per meter, q33 is radians of error per revolution
        self.var = np.zeros_like(self.process_covariance)

        # Particles to be resampled whether we have measurements or not (in guidance.py)
        self.resample_index = np.arange(self.N)
        self.last_time = (
            initial_time if initial_time is not None else 0.0
        )
        self.t_since_last_update = 0.0
        self.iteration = 0

    def uniform_sample(self) -> np.ndarray:
        """Uniformly samples the particles between min and max values per state.
        Positions are always sampled according to the lab dimensions.
        Output:
            local_particles: N_th sets (time histories) of N particles with Nx states
        """
        if self.prediction_method == "Velocity":
            local_particles = np.random.uniform(
                [
                    self.APRILab_dims[0, 0],
                    self.APRILab_dims[0, 1],
                    -self.vmax,
                    -self.vmax,
                ],
                [
                    self.APRILab_dims[1, 0],
                    self.APRILab_dims[1, 1],
                    self.vmax,
                    self.vmax,
                ],
                (self.N_th, self.N, self.Nx),
            )
        elif self.prediction_method == "Unicycle":
            local_particles = np.random.uniform(
                [self.APRILab_dims[0, 0], self.APRILab_dims[0, 1], -np.pi],
                [self.APRILab_dims[1, 0], self.APRILab_dims[1, 1], np.pi],
                (self.N_th, self.N, self.Nx),
            )
        elif self.prediction_method in {"NN", "KF", "DMMN", "Transformer"}:
            local_particles = np.random.uniform(
                [self.APRILab_dims[0, 0], self.APRILab_dims[0, 1]],
                [self.APRILab_dims[1, 0], self.APRILab_dims[1, 1]],
                (self.N_th, self.N, self.Nx),
            )
        self.weights = np.ones(self.N) / self.N
        return local_particles

    def pf_loop(
        self,
        noisy_measurement,
        t,
        ang_vel=np.array([0]),
        lin_vel=np.array([0]),
    ):
        """Main function of the particle filter where the predict,
        update, resample and estimate steps are called.
        """
        dt = t - self.last_time
        # update measurement history with noisy_measurement
        noisy_measurement = noisy_measurement[:2] if self.Nx > 2 else noisy_measurement
        self.update_state_and_dt_buffer(noisy_measurement[:2], dt=dt)

        # Prediction step
        if self.prediction_method in {"NN", "Transformer"}:
            self.particles = self.predict_mml(np.copy(self.particles), self.dt_history)
        elif self.prediction_method == "Unicycle":
            self.particles = self.predict(
                self.particles,
                self.dt_history[-1],
                angular_velocity=ang_vel,
                linear_velocity=lin_vel,
            )
        elif self.prediction_method == "Velocity":

            self.particles = self.predict(
                self.particles,
                self.dt_history[-1],
            )

        # Update step
        if self.is_update:
            self.prev_weights = np.copy(self.weights)
            self.weights = self.update(self.weights, self.particles, noisy_measurement)

        # Resampling step
        outbounds = self.outside_bounds(self.particles[-1])
        self.neff = self.nEff(self.weights)
        if outbounds > self.N * 0.8:
            print(
                f"{self.outside_bounds(self.particles[-1])} particles outside the lab boundaries. Uniformly resampling."
            )
            self.particles = self.uniform_sample()
        else:
            if (self.neff < self.N * 0.7):  
                if (self.neff < self.N * 0.1) and self.is_update:
                    # most particles are bad, resample from Gaussian around the measurement
                    self.gaussian_reset(noisy_measurement)
                else:
                    # some are good but some are bad, resample
                    self.multinomial_resample()
            elif self.neff > np.inf:
                print("All particles have zero weight. Reinitializing weights uniformly.")
                self.weights = np.ones(self.N) / self.N

        # estimate mean and variance
        self.weighted_mean, self.var = self.estimate(self.particles[-1], self.weights)

        stuck_threshold = 0.03 if self.prediction_method == "Velocity" else 0.03
        max_time_since_last_update = 12.0
        wm_pos_diff = np.linalg.norm(self.state_history[0][:2] - self.weighted_mean[:2])
        time_condition = (
            np.floor(self.t_since_last_update) % max_time_since_last_update == 0
        ) and self.t_since_last_update > max_time_since_last_update
        diff_position_condition = wm_pos_diff < stuck_threshold and not self.is_update
        if diff_position_condition or time_condition:
            print(
                f"Particle filter stuck: {diff_position_condition} or lost measurement: {time_condition}. Uniformly resampling."
            )
            self.particles = self.uniform_sample()

        self.last_time = t
    
    def gaussian_reset(self, noisy_measurement):
        """Reset particles around the noisy measurement with Gaussian noise."""
        if self.prediction_method == "Velocity":
            # Improved resampling for velocity method
            self.particles[-1, :, :2] = np.random.multivariate_normal(
                noisy_measurement[:2],
                self.measurement_covariance[:2, :2],
                self.N,
            )
            # backwards difference for velocities
            estimate_velocity = (
                noisy_measurement - self.state_history[-1, :2]
            ) * self.dt_history[-1]
            self.particles[-1, :, 2:] = np.random.multivariate_normal(
                estimate_velocity,
                self.dt_history[-1] * self.measurement_covariance,
                self.N,
            )
        else:
            noise = np.random.multivariate_normal(
                np.zeros(self.Nx),
                self.measurement_covariance,
                size=(self.N_th, self.N),
            )
            # create a uniform noise of the angle
            if self.Nx > 2:
                noise[:, :, 2] = np.random.uniform(-np.pi, np.pi, size=(self.N_th, self.N))
            # repeat the measurement history to be the same size as the particles
            filled_elements = np.count_nonzero(self.state_history) // self.Nx
            if filled_elements < self.N_th:
                self.particles[-1, :, :2] = noisy_measurement[:self.Nx] + noise[-1, :, :2]
            else:
                state_history_repeated = np.tile(
                    self.state_history[-self.N_th :].reshape(
                        self.N_th, 1, self.Nx
                    ),
                    (1, self.N, 1),
                )
                self.particles = state_history_repeated + noise

        self.weights = np.ones(self.N) / self.N
 
    def optimize_learned_model_callback(self, event=None):
        """Optimize the neural network model with the particles and weights"""
        filled_elements = np.count_nonzero(self.state_history) // self.Nx
        if not np.all(self.state_history):
            print(f"Filled elements: {filled_elements} / {self.state_history.shape[0]}")
        # if measurement history buffer is full
        if filled_elements > self.N_th + self.max_batch_size * 0.6 and (
            self.prediction_method in {"NN", "Transformer"}
        ):
            self.optimize_learned_model(filled_elements)

    def update(self, weights, particles, noisy_turtle_pose):
        """Updates the belief (weights) of the particle distribution.
        Input: Likelihood of the particles from measurement model and prior belief of the particles
        Output: Updated (posterior) weight of the particles
        """
        weights = weights * self.likelihood(
            particles[-1, :, :], np.tile(noisy_turtle_pose, (self.N, 1))
        )
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights
        return weights

    def likelihood(self, particles, y_act):
        """Particles that are closer to the noisy measurements are weighted higher than
        particles which don't match the measurements very well.
        There are two methods to compute this.
        """

        # Method 1: Manual for loop with normal multivariate equation
        shape = particles.shape[0]
        like = np.zeros(shape)
        for ii in range(shape):
            # The factor sqrt(det((2*pi)*measurement_cov)) is not included in the
            # likelihood, but it does not matter since it can be factored
            # and then cancelled out during the normalization or expectation.
            like[ii] = np.exp(
                -0.5
                * (particles[ii, :2] - y_act[ii, :2]).T
                @ self.noise_inv
                @ (particles[ii, :2] - y_act[ii, :2])
            )

        # Method 2: Vectorized using scipy.stats
        # TODO fix to account for different measurements
        # like = stats.multivariate_normal.pdf(
        #    x=particles, mean=y_act[0], cov=self.measurement_covariance
        # )
        return like

    def predict_mml(self, particles, dt_history):
        """Propagates the current particles through the motion model learned with the
        neural network.
        Input: N_th (number of time histories) sets of particles up to time k-1
        Output: N_th sets of propagated particles up to time k
        """
        if self.is_time_weights:
            tiled_dt_hist = np.tile(
                dt_history.reshape(-1, 1), (1, particles.shape[1])
            ).reshape(-1, particles.shape[1], 1)
            particles = np.concatenate((particles, tiled_dt_hist), axis=2)

        next_parts = self.motion_model.predict(
            np.transpose(particles[:, :, :], (1, 0, 2)).reshape(
                particles.shape[1], self.nn_input_size
            )
        )

        if self.is_velocity:
            next_parts = (
                particles[-1, :, :2] + next_parts * dt_history[-1]
            )  # x(t+1) = x(t) + v(t) * dt

        # roll particles in time, add latest prediction
        particles = np.concatenate(
            (
                particles[1:, :, :2],
                next_parts.reshape(1, next_parts.shape[0], next_parts.shape[1]),
            ),
            axis=0,
        )
        # important: will not work without this
        particles[-1, :, :] = self.add_noise(
            particles[-1, :, :], 2 * self.process_covariance
        )
        return particles

    def predict(
        self,
        particles,
        dt,
        linear_velocity=np.zeros(2),
        angular_velocity=np.zeros(1),
    ):
        """Uses the unicycle model plus noise or the to propagate the belief in the system state.
        The position and orientation of the particles are updated according to the linear
        and angular velocity of the turtlebot.
        In the Velocity method, the particles are propagated according to the velocity of each particle.
        Input: State of the particles at time k-1
        Output: Predicted (propagated) state of the particles up to time k
        """
        particles[:-1, :, :] = particles[1:, :, :]  # shift particles in time
        if self.prediction_method == "Unicycle":
            delta_theta = angular_velocity[0] * dt
            particles[-1, :, 2] = (
                particles[-2, :, 2]
                + delta_theta
                + (delta_theta / (2 * np.pi))
                * self.add_noise(
                    np.zeros(particles.shape[1]),
                    self.process_covariance[2, 2],
                    size=particles.shape[1],
                )
            )

            for ii in range(particles.shape[1]):
                if np.abs(particles[-1, ii, 2]) > np.pi:
                    # Wraps angle
                    particles[-1, ii, 2] = (
                        particles[-1, ii, 2] - np.sign(particles[-1, ii, 2]) * 2 * np.pi
                    )

            norm_lin_vel = np.linalg.norm(linear_velocity)
            delta_distance = norm_lin_vel * dt + norm_lin_vel * dt * self.add_noise(
                0, self.process_covariance[0, 0], size=particles.shape[1]
            )
            particles[-1, :, :2] = (
                particles[-2, :, :2]
                + np.array(
                    [
                        delta_distance * np.cos(particles[-1, :, 2]),
                        delta_distance * np.sin(particles[-1, :, 2]),
                    ]
                ).T
            )
        elif self.prediction_method == "Velocity":
            dt = max(dt, 1e-6)  # Avoid division by zero
            
            # Position update: x = x + vx*dt + noise
            pos_noise = self.add_noise(
                np.array([0, 0]), 
                np.diag([self.proc_pos_std**2, self.proc_pos_std**2]) * dt,
                size=particles.shape[1]
            )
            particles[-1, :, :2] = (
                particles[-2, :, :2] + 
                particles[-1, :, 2:4] * dt + 
                pos_noise
            )
            
            # Velocity update: vx = vx + noise
            vel_noise = self.add_noise(
                np.array([0, 0]), 
                np.diag([self.proc_vel_std**2, self.proc_vel_std**2]) * dt,
                size=particles.shape[1]
            )
            particles[-1, :, 2:4] = particles[-1, :, 2:4] + vel_noise
            
            # Velocity bounds to prevent unrealistic velocities
            particles[-1, :, 2:4] = np.clip(particles[-1, :, 2:4], -self.vmax, self.vmax)

        return particles

    def multinomial_resample(self):
        """Uses the multinomial resampling algorithm to update the belief in the system state.
        The particles are copied randomly with probability proportional to the weights plus
        some roughening based on the spread of the states.
        Inputs: Updated state of the particles
        Outputs: Resampled updated state of the particles
        """
        N_r = len(self.resample_index)
        if N_r > 0:
            # Normalize weights
            self.weights = (
                self.weights / np.sum(self.weights)
                if np.sum(self.weights) > 0
                else self.weights
            )
            # Copy particles that are weighted higher
            rand_ind = np.random.choice(a=self.N, size=N_r, p=self.weights)
            self.particles[-1, self.resample_index, :] = self.particles[-1, rand_ind, :]
            self.weights[self.resample_index] = self.weights[rand_ind]
            # Roughening. See Bootstrap Filter from Crassidis and Junkins.
            G = 1.1 if self.prediction_method == "Velocity" else 0.6
            E = np.zeros(self.Nx)
            for ii in range(self.Nx):
                E[ii] = np.max(self.particles[-1, :, ii]) - np.min(
                    self.particles[-1, :, ii]
                )
            cov = (G * E * N_r ** (-1 / 3)) ** 2
            P_sigmas = np.diag(cov)

            for ii in self.resample_index:
                self.particles[-1, ii, :] = self.add_noise(
                    self.particles[-1, ii, :], P_sigmas
                )

    def estimate(self, particles, weights):
        """returns mean and variance of the weighted particles"""
        assert np.sum(weights) > 0.0, "Sum of weights must be greater than zero, now its " + str(np.sum(weights))
        weighted_mean = np.average(particles, weights=weights, axis=0)
        var = np.zeros_like(self.var)
        pos_diff = particles[:, :2] - weighted_mean[:2]
        # Weighted covariance computation: Cov = E[(X-μ)(X-μ)^T]
        pos_cov = np.average(pos_diff[:, :, np.newaxis] * pos_diff[:, np.newaxis, :], 
                           weights=weights, axis=0)
        var[:2, :2] = pos_cov
        
        if self.prediction_method == "Unicycle":
            # Component mean in the complex plane to prevent wrong average
            # source: https://www.rosettacode.org/wiki/Averages/Mean_angle#C.2B.2B
            self.yaw_mean = np.arctan2(
            np.sum(self.weights * np.sin(particles[:, 2])),
            np.sum(self.weights * np.cos(particles[:, 2])),
            )
            weighted_mean[2] = self.yaw_mean
            var[2, 2] = np.average(
            (particles[:, 2] - weighted_mean[2]) ** 2,
            weights=weights,
            axis=0,
            )
        elif self.prediction_method == "Velocity":
            var[2:, 2:] = np.average(
                (particles[:, 2:] - weighted_mean[2:]) ** 2,
                weights=weights,
                axis=0,
            )
        return weighted_mean, var

    def outside_bounds(self, particles):
        """returns the number of particles outside of the lab boundaries"""
        return np.sum(
            (particles[:, 0] < self.APRILab_dims[0, 0])
            | (particles[:, 0] > self.APRILab_dims[1, 0])
            | (particles[:, 1] < self.APRILab_dims[0, 1])
            | (particles[:, 1] > self.APRILab_dims[1, 1])
        )

    def update_state_and_dt_buffer(self, noisy_measurement, dt):
        """Update the measurement history and the time history"""
        # pf dt history
        self.dt_history[:-1] = self.dt_history[1:] + dt  # delta time from index i to -1
        self.dt_history[-1] = dt

        if self.is_update:
            self.state_history = np.roll(self.state_history, -1, axis=0)
            self.state_history[-1, :2] = noisy_measurement[:2]
            self.dt_state_history[:-1] = (
                self.dt_state_history[1:] + self.t_since_last_update + dt
            )
            self.dt_state_history[-1] = self.t_since_last_update + dt
            self.t_since_last_update = 0.0
        else:
            self.t_since_last_update += dt

    def update_measurement_covariance(self, height):
        """Update the measurement covariance matrix based on the height of the drone agent"""
        gain = 1 + (max(height, 1.1) - 1.1) / (
            3.0 - 1.1
        )  # start with 2 (heght=1.8) and decrease to 1 (height=1.1)
        gain = 1.  # Override TODO: remove this line
        self.measurement_covariance = gain * self.best_measurement_covariance
        self.noise_inv = np.linalg.inv(self.measurement_covariance[:2, :2])

    def convert_state_history_to_training_batches(
        self, filled_elements, all_data=False
    ):
        """Convert the measurement history to training batches for the neural network
        Input: filled_elements: number of elements in the measurement history buffer
        Output: X_train: input training data, y_train: output training data
        """
        state_history = self.state_history[-filled_elements:]
        dt_state_history = self.dt_state_history[-filled_elements:]
        if all_data:
            batch_size = filled_elements - self.N_th
        else:
            batch_size = np.min((self.max_batch_size, filled_elements - self.N_th))
            if batch_size <= 0:
                return None, None
        X_train = np.zeros((batch_size, self.N_th, self.Nx))
        dt_time_features = np.zeros((batch_size, self.N_th))
        y_train = np.zeros((batch_size, self.Nx))

        # sample n_batches indexes from the data
        if all_data:
            idx = np.arange(batch_size)
        else:
            idx = np.random.choice(
                state_history.shape[0] - self.N_th, batch_size, replace=False
            )

        # TODO: avoid for loop
        include_index = []
        for i, sampled_i in enumerate(idx):
            X_train[i] = state_history[sampled_i : sampled_i + self.N_th].copy()
            dt_time_features[i] = (
                dt_state_history[sampled_i : sampled_i + self.N_th].copy()
                - dt_state_history[sampled_i + self.N_th]
            )
            y_train[i] = state_history[sampled_i + self.N_th].copy()

            # remove samples with dt > 4 seconds
            if dt_time_features[i][0] < 4.0:
                include_index.append(i)

        X_train = np.concatenate((X_train, dt_time_features[..., np.newaxis]), axis=2)

        include_index = np.array(include_index)
        print(f"eliminated samples: {batch_size - len(include_index)}")
        X_train = X_train[include_index]
        y_train = y_train[include_index]

        if self.is_velocity:
            y_train = (y_train - X_train[:, -1, :2]) / X_train[:, -1, 2].reshape(
                -1, 1
            )  # vel = (x(t+1) - x(t)) / dt

        return X_train, y_train

    def optimize_learned_model(self, filled_elements):
        """Optimize the neural network model with the particles and weights"""
        print("\n")
        print("Optimizing the learned model")
        # training parameters
        epochs = 5
        criterion = nn.MSELoss().to(torch.float32)
        optimizer = optim.Adam(self.motion_model.parameters(), lr=0.001)
        # data transformation
        X_train, y_train = self.convert_state_history_to_training_batches(
            filled_elements
        )
        if self.prediction_method == "NN":
            X_train = X_train.reshape(
                X_train.shape[0], -1
            )  # flatten last two dimensions

        X_train = torch.from_numpy(X_train.astype(np.float32)).to(self.device)
        y_train = torch.from_numpy(y_train.astype(np.float32)).to(self.device)

        losses = train(
            self.motion_model,
            criterion,
            optimizer,
            X_train,
            y_train,
            epochs,
            online=True,
        )

    def save_model(self, save_path=None):
        """Save the model to a file"""
        if save_path is None:
            save_path = "models/online_model.pth"

        if self.prediction_method in {"NN", "Transformer"}:
            torch.save(self.motion_model.state_dict(), save_path)
            csv_file = save_path.replace(".pth", "_data.csv")
            os.makedirs(os.path.dirname(csv_file), exist_ok=True)
            filled_elements = np.count_nonzero(self.state_history) // self.Nx
            X_train, y_train = self.convert_state_history_to_training_batches(
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
        elif self.prediction_method == "DMMN":
            self.motion_model.saveModel(time=time.time(), savePath=save_path)
        print(f"Model saved to {save_path}")

    @staticmethod
    def add_noise(mean, covariance, size=1):
        """Add noise to the mean from a gaussian distribution with covariance matrix"""
        if type(mean) is np.ndarray and type(covariance) is np.ndarray:
            if mean.ndim > 1:
                size = mean.shape[0]
                noise = np.random.multivariate_normal(
                    np.zeros(mean.shape[1]), covariance, size
                )
            else:
                size = mean.shape[0]
                noise = np.random.multivariate_normal(np.zeros(size), covariance)
        else:
            noise = np.random.normal(0, covariance, size)

        return mean + noise

    @staticmethod
    def nEff(wgts):
        """Compute the number of effective particles
        Source: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb
        """
        return 1.0 / np.sum(np.square(wgts))
