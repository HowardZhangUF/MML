#!/usr/bin/env python3
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time

from mml_core.motion.motion_model_network import MotionModel
from mml_core.motion.simple_dnn import SimpleDNN
from mml_core.motion.scratch_transformer import ScratchTransformer
from mml_core.motion.cfm_motion_model import CFMMotionModel

from .prediction import ParticlePrediction
from .online_learning import OnlineLearner


class ParticleFilter:
    """Posterior state, measurement updates, resampling, and filter lifecycle."""

    @property
    def prediction(self):
        # Lazy binding also supports instances restored from older pickles.
        if "_prediction" not in self.__dict__:
            self._prediction = ParticlePrediction(self)
        return self._prediction

    @property
    def online_learner(self):
        if "_online_learner" not in self.__dict__:
            self._online_learner = OnlineLearner(self)
        return self._online_learner

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
        self.requested_prediction_method = prediction_method
        if prediction_method in {"MultiKF", "MultiPFVel"}:
            prediction_method = "Velocity"  # temporary fix
        elif prediction_method in {"MultiPFTra"}:
            prediction_method = "Transformer"  # temporary fix
        elif prediction_method == "PF+CFM":
            prediction_method = "CFM"
        elif prediction_method == "PF+CFM-AS":
            prediction_method = "CFM-AS"
        elif prediction_method == "PF+NF":
            prediction_method = "NF"
        elif prediction_method == "PF+Diffusion":
            prediction_method = "Diffusion"
        elif prediction_method not in {
            "NN",
            "Transformer",
            "Unicycle",
            "Velocity",
            "KF",
            "DMMN",
            "CFM",
            "CFM-AS",
            "NF",
            "Diffusion",
            "DiffPF",
        }:
            raise ValueError(
                f"Invalid prediction method: {prediction_method}."
            )
            
        self.prediction_method = prediction_method
        self.N = int(num_particles)
        self.cfm_base_sigma = None
        self.cfm_adaptive_gain = 0.05
        self.cfm_sigma_max = 2.5
        self.steps_since_measurement = 0
        # boundary of the lab [[x_min, y_min], [x_max, y_,max]] [m]
        self.APRILab_dims = np.array([[-3.0, -3.0], [3.0, 3.0]])  # sim

        self.prediction.configure_model(model_path)

        # PF
        self.weights = np.ones(self.N) / self.N
        self.prev_weights = np.copy(self.weights)
        self.weighted_mean = np.array([0, 0, 0])
        self.is_update = False
        self.is_occlusion = False
        self.neff = self.nEff(self.weights)

        buffer_size = self.prediction.configure_dynamics(model_path)

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
        elif self.prediction_method in {
            "NN",
            "KF",
            "DMMN",
            "Transformer",
            "CFM",
            "CFM-AS",
            "NF",
            "Diffusion",
            "DiffPF",
        }:
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
        if self.prediction_method == "DiffPF":
            self._diffpf_loop(noisy_measurement, t, dt)
            return

        # update measurement history with noisy_measurement
        noisy_measurement = noisy_measurement[:2] if self.Nx > 2 else noisy_measurement
        self.update_state_and_dt_buffer(noisy_measurement[:2], dt=dt)

        # Prediction step
        if self.prediction_method == "CFM-AS":
            self.motion_model._cfm.anchor_sigma = min(
                self.cfm_sigma_max,
                self.cfm_base_sigma
                * (1.0 + self.cfm_adaptive_gain * self.steps_since_measurement),
            )

        if self.prediction_method in {"CFM", "CFM-AS", "NF", "Diffusion"}:
            self.particles = self.predict_cfm(np.copy(self.particles), self.dt_history)
        elif self.prediction_method in {"NN", "Transformer"}:
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
            self.steps_since_measurement = 0
        else:
            self.steps_since_measurement += 1

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
            elif not np.isfinite(self.neff) or np.sum(self.weights) <= 0:
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
        # Only flag "stuck" after the PF has gone without an update for >3 s.
        # Without this guard, wm_pos_diff ≈ 0 at initialisation (both
        # state_history[0] and weighted_mean start at [0,0]) and the condition
        # fires on every invisible step, causing a spam of uniform_sample() calls
        # that scatter particles across the whole environment and then cause all
        # weights to underflow to 0 on the next visible step → crash.
        diff_position_condition = (self.t_since_last_update > 3.0 and
                                   wm_pos_diff < stuck_threshold and
                                   not self.is_update)
        if diff_position_condition or time_condition:
            print(
                f"Particle filter stuck: {diff_position_condition} or lost measurement: {time_condition}. Uniformly resampling."
            )
            self.particles = self.uniform_sample()

        self.last_time = t

    def _diffpf_loop(self, noisy_measurement, t, dt):
        """Run one DiffPF posterior-sampling update and mirror its state."""
        measurement = (
            np.asarray(noisy_measurement, dtype=np.float32)[:2]
            if self.is_update
            else None
        )
        estimate = self.diffpf_tracker.step(measurement)
        self.particles = self.diffpf_tracker.particles[None, :, :].copy()
        self.weights = np.full(self.N, 1.0 / self.N, dtype=np.float64)
        self.prev_weights = self.weights.copy()
        self.weighted_mean, self.var = self.estimate(
            self.particles[-1],
            self.weights,
        )
        self.weighted_mean[:2] = estimate

        if self.is_update:
            self.t_since_last_update = 0.0
            self.steps_since_measurement = 0
        else:
            self.t_since_last_update += max(float(dt), 0.0)
            self.steps_since_measurement += 1
        self.last_time = t
    
    def gaussian_reset(self, noisy_measurement):
        """Reset particles around the noisy measurement with Gaussian noise."""
        if self.prediction_method == "DiffPF":
            self.diffpf_tracker.n_particles = self.N
            self.diffpf_tracker.reset(
                np.asarray(noisy_measurement, dtype=np.float32)[:2],
                init_spread=0.05,
            )
            self.particles = self.diffpf_tracker.particles[None, :, :].copy()
            self.weights = np.full(self.N, 1.0 / self.N, dtype=np.float64)
            self.steps_since_measurement = 0
            return

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
        return self.online_learner.optimize_learned_model_callback(event)

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
        return self.prediction.predict_mml(particles, dt_history)
    
    def predict_cfm(self, particles, dt_history):
        return self.prediction.predict_cfm(particles, dt_history)
    
    def predict(
        self,
        particles,
        dt,
        linear_velocity=np.zeros(2),
        angular_velocity=np.zeros(1),
    ):
        return self.prediction.predict(particles, dt, linear_velocity, angular_velocity)

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

            if N_r == self.N:
                self.weights = np.full(self.N, 1.0 / self.N, dtype=np.float64)
            else:
                weight_sum = np.sum(self.weights)
                self.weights = (
                    self.weights / weight_sum
                    if np.isfinite(weight_sum) and weight_sum > 0.0
                    else np.full(self.N, 1.0 / self.N, dtype=np.float64)
                )

    def estimate(self, particles, weights):
        """returns mean and variance of the weighted particles"""
        w_sum = np.sum(weights)
        if not (np.isfinite(w_sum) and w_sum > 0.0):
            weights = np.ones(len(weights)) / len(weights)
        else:
            weights = weights / w_sum
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
        return self.online_learner.convert_state_history_to_training_batches(filled_elements, all_data)

    def optimize_learned_model(self, filled_elements):
        return self.online_learner.optimize_learned_model(filled_elements)

    def save_model(self, save_path=None):
        return self.online_learner.save_model(save_path)

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
