#!/usr/bin/env python3

from mml_core.sensing import Occlusions, construct_fov, is_in_fov

import numpy as np
import copy
import scipy.stats as stats
from .lawnmower import LawnmowerPath
from mml_core.tracking.kalman_filter import KalmanFilter
from mml_core.tracking.particle_filter import ParticleFilter
from mml_core.tracking.kalman_gnn import KalmanFilterGNNAssociator
from mml_core.tracking.particle_gnn import ParticleFilterGNNAssociator

MULTI_PF_METHOD_MAP = {
    "MultiPFVel": "Velocity",
    "MultiPFTra": "Transformer",
    "CFM": "CFM",
    "PF+CFM": "PF+CFM",
    "PF+CFM-AS": "PF+CFM-AS",
    "PF+NF": "PF+NF",
    "PF+Diffusion": "PF+Diffusion",
    "DiffPF": "DiffPF",
}

try:
    from stable_baselines3 import PPO
    from sb3_contrib import RecurrentPPO  
    PPO_AVAILABLE = True
except ImportError:
    PPO_AVAILABLE = False
    print("Warning: stable_baselines3 not available. PPO guidance modes will not work.")


from .guidance_policy import GuidancePolicy
from .guidance_information import InformationGuidance


class Guidance:
    """Tracking and goal orchestration with policy and information collaborators."""

    @property
    def policy(self):
        if "_policy" not in self.__dict__:
            self._policy = GuidancePolicy()
            # Preserve state from instances saved before component extraction.
            for name in ("ppo_model", "is_recurrent_ppo", "lstm_states"):
                if name in self.__dict__:
                    setattr(self._policy, name, self.__dict__.pop(name))
        return self._policy

    @property
    def information(self):
        if "_information" not in self.__dict__:
            self._information = InformationGuidance(self)
        return self._information

    @property
    def ppo_model(self):
        return self.policy.ppo_model

    @ppo_model.setter
    def ppo_model(self, value):
        self.policy.ppo_model = value

    @property
    def is_recurrent_ppo(self):
        return self.policy.is_recurrent_ppo

    @is_recurrent_ppo.setter
    def is_recurrent_ppo(self, value):
        self.policy.is_recurrent_ppo = value

    @property
    def lstm_states(self):
        return self.policy.lstm_states

    @lstm_states.setter
    def lstm_states(self, value):
        self.policy.lstm_states = value

    def __init__(
        self,
        guidance_mode="Information",
        prediction_method="Transformer",
        occ_centers=[[]],
        occ_widths=[],
        drone_height=2.0,
        is_height_constant=True,
        filter: ParticleFilter = None,
        initial_time=None,
        model_path=None,
        ppo_model_path=None,  # NEW: Path to trained PPO model
        measurement_seed=None,
    ):
        print(
            "Initializing Guidance with mode:",
            guidance_mode,
            "and prediction method:",
            prediction_method,
        )
        self.init_finished = False
        
        self.guidance_modes_set = {
            "Information", "WeightedMean", "Lawnmower", "Predefined", "Estimator",
            "MultiKFInfo", "MultiPFInfo", "MultiPFPPO", "MultiPFInfoPPO", "MultiPFInfoPPO_LSTM"  # ADD THIS
        }
        assert guidance_mode in self.guidance_modes_set, f"Guidance mode {guidance_mode} not recognized. Choose from {self.guidance_modes_set}"
        self.guidance_mode = guidance_mode
        
        self.prediction_methods_set = {
            "NN",
            "Transformer",
            "Unicycle",
            "Velocity",
            "KF",
            "MultiKF",
            *MULTI_PF_METHOD_MAP.keys(),
        }
        self.prediction_method = prediction_method
        assert prediction_method in self.prediction_methods_set, f"Prediction method {prediction_method} not recognized. Choose from {self.prediction_methods_set}"
        self.filter = filter  
        self.N = filter.N if filter is not None else 500
        self.measurement_rng = (
            np.random.default_rng(measurement_seed)
            if measurement_seed is not None
            else None
        )
        
        self.policy.load(
            guidance_mode, ppo_model_path, PPO_AVAILABLE,
            globals().get("PPO"), globals().get("RecurrentPPO"),
        )

        # Initialization of robot variables
        self.quad_position = np.array([0.0, 0.0])
        self.actual_turtle_pose = np.array([0.0, 0.0, 0.0])
        self.noisy_turtle_pose = np.array([0.0, 0.0, 0.0])
        self.goal_position = np.array([0.0, 0.0])
        self.future_tracks = []
        self.linear_velocity = np.array([0.0, 0.0])
        self.angular_velocity = np.array([0.0])
        deg2rad = lambda deg: np.pi * deg / 180
        self.initial_time = initial_time if initial_time is not None else 0

        # Camera Model
        self.is_height_constant = is_height_constant
        self.height = drone_height  # initial height of the quadcopter in meters
        self.CAMERA_ANGLES = np.array(
            [deg2rad(45), deg2rad(45)]
        )  # camera angle in radians (horizontal, vertical)
        self.update_FOV_dims_and_measurement_cov()
        self.FOV = self.construct_FOV(self.quad_position)

        ## INFO-DRIVEN GUIDANCE ##
        self.N_s = 25  # Number of sampled particles
        self.K = 4  # Time steps to propagate in the future for EER
        self.Hp_t = 1.0  # partial entropy
        self.prev_Hp = np.ones((5, 1))
        self.eer_particle = 0  # initialize future particle to follow
        self.sampled_index = np.arange(self.N_s)
        self.sampled_particles = self.filter.particles[:, : self.N_s, :]
        self.sampled_weights = np.ones(self.N_s) / self.N_s
        self.position_following = True
        self.avg_time = None
        self.max_time = 0.0
        self.min_time = 10.0
        self.iteration = 0

        # Occlusions
        if occ_centers is not None:
            occ_centers = [occ_centers[i*2:(i+1)*2] for i in range(len(occ_widths))]
        self.occlusions = Occlusions(occ_centers, occ_widths)

        if self.guidance_mode in {"Lawnmower", "Predefined"}:
            # Lawnmower Method
            lawnmower = LawnmowerPath(POINTS_PER_SLICE=8)
            bounds = [
                self.filter.APRILab_dims[0],
                np.array(
                    [self.filter.APRILab_dims[1][0], self.filter.APRILab_dims[0][1]]
                ),
                self.filter.APRILab_dims[1],
                np.array(
                    [self.filter.APRILab_dims[0][0], self.filter.APRILab_dims[1][1]]
                ),
            ]
            self.path, _ = lawnmower.generate_path(bounds, path_dist=0.4, angle=0)
            self.lawnmower_idx = 0
            self.increment = 1

        if self.prediction_method == "KF":
            # Kalman Filter
            H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
            self.kf = KalmanFilter(
                R=self.filter.measurement_covariance,
                Q=self.filter.process_covariance,
                H=H,
            )
        elif self.prediction_method == "MultiKF":
            # Initialize multi-target tracker
            self.multi_filter = KalmanFilterGNNAssociator(process_noise=0.15, measurement_noise=0.8)
        elif self.prediction_method in MULTI_PF_METHOD_MAP:
            pf_prediction = MULTI_PF_METHOD_MAP[self.prediction_method]
            # Initialize multi-target particle filter tracker
            self.multi_filter = ParticleFilterGNNAssociator(
                num_particles=200,
                prediction_method=pf_prediction,
                max_association_distance=3.0,
                model_path=model_path,
                drone_height=drone_height,
                initial_time=self.initial_time
            )
            self.multi_filter.in_occlusion = self.occlusions.in_occlusion  # temp fix
            self.multi_filter.is_in_FOV = self.is_in_FOV

        self.init_finished = True

    def get_ppo_action_lstm(self, agent_state, tracked_states, lstm_states=None, episode_start=None):
        return self.policy.get_ppo_action_lstm(agent_state, tracked_states, lstm_states, episode_start)

    def get_ppo_action_all_tracks(self, agent_state, tracked_states, max_targets=10,
                                   lstm_states=None, episode_start=None,
                                   visibility_flags=None):
        return self.policy.get_ppo_action_all_tracks(agent_state, tracked_states, max_targets, lstm_states, episode_start, visibility_flags)

    def get_ppo_action_nearest_track(self, agent_state, tracked_states, reset_lstm=False):
        return self.policy.get_ppo_action_nearest_track(agent_state, tracked_states, reset_lstm)

    # Keep the old method for backward compatibility but mark as deprecated
    def get_ppo_action(self, agent_state, target_estimates):
        return self.policy.get_ppo_action(agent_state, target_estimates)

    def current_entropy(self, event=None) -> None:
        return self.information.current_entropy(event)

    def propagate_particles(self, future_parts, dt=0.33, particle_filter=None):
        return self.information.propagate_particles(future_parts, dt, particle_filter)

    def information_driven_guidance(self, future_parts):
        return self.information.information_driven_guidance(future_parts)

    def info_driven_guidance_multi_kf(self, future_estimates):
        return self.information.info_driven_guidance_multi_kf(future_estimates)

    def info_driven_guidance_multi_pf(self, future_tracks):
        return self.information.info_driven_guidance_multi_pf(future_tracks)

    def entropy_gaussian(self, mean, cov):
        return self.information.entropy_gaussian(mean, cov)

    def entropy_particle(
        self,
        prev_particles,
        prev_wgts,
        particles,
        wgts=np.array([]),
        z_meas=np.array([]),
    ) -> np.int64:
        return self.information.entropy_particle(prev_particles, prev_wgts, particles, wgts, z_meas)

    @staticmethod
    def reject_spikes(prev_values, current_value, threshold_factor=3):
        """
        Rejects spikes in time-series data of previous values.

        Parameters:
            prev_values (numpy.ndarray): An array containing previous data points in the time series.

            current_value (float or int): The current data point in the time series.
            threshold_factor (float, optional): A factor to scale the standard deviation
                to determine the threshold for rejecting spikes. Defaults to 3.

        Returns:
            float or int: The filtered value. If the current value is considered a spike,
                it is replaced by the median of the values in the moving window and current_value;
                otherwise, the current value is returned unchanged.

        Raises:
            None
        """
        # join array of values prev_values and current_value
        values = np.append(prev_values, current_value)
        median = np.median(values)
        std = np.std(values)
        threshold = median + threshold_factor * std
        if abs(current_value - median) > threshold:
            return median
        else:
            return current_value

    def is_in_FOV(self, sparticles, fov):
        return is_in_fov(sparticles, fov)

    def construct_FOV(self, fov_center=np.array([0, 0])) -> np.ndarray:
        return construct_fov(fov_center, self.FOV_dims)

    def update_FOV_dims_and_measurement_cov(self):
        """Update the FOV dimensions based on the camera angles and height of drone
        as well as the measurement covariance based on the height of the drone
        """
        self.FOV_dims = np.tan(self.CAMERA_ANGLES) * self.height
        self.filter.update_measurement_covariance(self.height)

    def lawnmower(self) -> np.ndarray:
        """Return the position of the measurement if there is one,
        else return the next position in the lawnmower path.
        If the rate of pub_desired_state changes, the POINTS_PER_SLICE
        variable needs to change
        """
        if self.filter.is_update:
            return self.noisy_turtle_pose[:2]
        else:
            if (
                np.linalg.norm(self.quad_position - self.path[self.lawnmower_idx, :2])
                > 1.0
            ):
                self.lawnmower_idx = np.argmin(
                    np.linalg.norm(self.path[:, :2] - self.quad_position, axis=1)
                )
            if self.lawnmower_idx <= 0:
                self.increment = 1
            if self.lawnmower_idx >= self.path.shape[0] - 1:
                self.increment = -1
            self.lawnmower_idx += self.increment
            return self.path[int(np.floor(self.lawnmower_idx)), :2]

    def predefined_path(self) -> np.ndarray:
        """Advance on the fixed lawnmower path, independent of detections."""
        if (
            np.linalg.norm(self.quad_position - self.path[self.lawnmower_idx, :2])
            > 1.0
        ):
            self.lawnmower_idx = int(
                np.argmin(np.linalg.norm(self.path[:, :2] - self.quad_position, axis=1))
            )
        if self.lawnmower_idx <= 0:
            self.increment = 1
        if self.lawnmower_idx >= self.path.shape[0] - 1:
            self.increment = -1
        self.lawnmower_idx += self.increment
        return self.path[int(np.floor(self.lawnmower_idx)), :2]

    def update_goal_position(self, dt=0.33):
        """Get the goal position based on the guidance mode.
        Output: goal_position (numpy.array of shape (2,))"""
        if self.guidance_mode == "Information":
            self.sampled_index = np.random.choice(a=self.N, size=self.N_s)
            self.sampled_particles = np.copy(
                self.filter.particles[:, self.sampled_index, :]
            )
            self.sampled_weights = np.copy(self.filter.weights[self.sampled_index])
            future_parts = np.copy(self.sampled_particles)
            for k in range(self.K):  # propagate k steps in the future
                future_parts = self.propagate_particles(future_parts, dt)
            self.information_driven_guidance(future_parts)
            self.goal_position = future_parts[-1, self.eer_particle, :2]
            self.future_tracks = [future_parts]
        elif self.guidance_mode == "WeightedMean":
            self.goal_position = self.filter.weighted_mean
        elif self.guidance_mode == "Lawnmower":
            mower_position = self.lawnmower()
            self.goal_position = mower_position
        elif self.guidance_mode == "Predefined":
            self.goal_position = self.predefined_path()
        elif self.guidance_mode == "Estimator":
            self.goal_position = self.actual_turtle_pose[:2]
        elif self.guidance_mode == "MultiKFInfo":
            future_estimate = self.multi_filter.tracks.copy()
            perform_information = True
            for k in range(self.K):
                future_estimate = self.multi_filter.predict_tracks(future_estimate, steps_ahead=k+1)
                if future_estimate is None or len(future_estimate) <= 0:
                    # random walk
                    self.goal_position = self.goal_position + np.random.uniform(-0.2, 0.2, size=(2,))
                    perform_information = False
                    break
            if perform_information:
                self.goal_position = self.info_driven_guidance_multi_kf(future_estimate)
        elif self.guidance_mode == "MultiPFInfo":
            future_tracks = copy.deepcopy(self.multi_filter.tracks)
            perform_information = True
            # Predict tracks forward for K time steps
            for k in range(self.K):
                if len(future_tracks) <= 0:
                    # random walk if no tracks
                    self.goal_position = self.goal_position + np.random.uniform(-0.2, 0.2, size=(2,))
                    perform_information = False
                    break
                # Predict each track forward
                for track in future_tracks:
                    if track.pf.prediction_method == "DiffPF":
                        track.pf.is_update = False
                        track.pf.pf_loop(
                            np.zeros(2, dtype=np.float32),
                            track.pf.last_time + dt,
                        )
                    else:
                        track.pf.particles = self.propagate_particles(
                            track.pf.particles,
                            dt,
                            particle_filter=track.pf,
                        )
            self.future_tracks = copy.deepcopy(future_tracks)
            if perform_information:
                # Use particle filter information-driven guidance
                self.goal_position = self.info_driven_guidance_multi_pf(future_tracks)

        # set height depending on runtime
        if self.is_height_constant:
            self.height = self.height
        else:
            dheight = 0.02
            if self.filter.is_update:
                self.height -= dheight
            else:
                self.height += dheight
            self.height = np.clip(self.height, 1.1, 1.8)
        self.update_FOV_dims_and_measurement_cov()

    @staticmethod
    def euler_from_quaternion(q):
        """Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        x = q[0]
        y = q[1]
        z = q[2]
        w = q[3]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = np.arcsin(t2)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)
        return np.array([roll_x, pitch_y, yaw_z])

    def update_target_pose(
        self, position, orientation=None, linear_vel=None, angular_vel=None
    ):
        """Update target pose from external source"""
        if orientation is not None:
            if len(orientation) == 4:  # quaternion
                _, _, theta_z = self.euler_from_quaternion(orientation)
                self.actual_turtle_pose = np.array([position[0], position[1], theta_z])
            else:  # assume it's already yaw angle
                self.actual_turtle_pose = np.array(
                    [position[0], position[1], orientation[0]]
                )
        else:
            self.actual_turtle_pose = np.array([position[0], position[1], 0.0])

        if linear_vel is not None:
            self.linear_velocity = np.array(linear_vel[:2])
        if angular_vel is not None:
            self.angular_velocity = np.array([angular_vel[0]])
        
        # print("Actual turtle pose:", self.actual_turtle_pose, 
        #       "Linear velocity:", self.linear_velocity, 
        #       "Angular velocity:", self.angular_velocity)

        # Add noise to measurement
        if self.measurement_rng is None:
            self.noisy_turtle_pose[:2] = self.filter.add_noise(
                self.actual_turtle_pose[:2],
                self.filter.measurement_covariance[:2, :2],
            )
        else:
            measurement_noise = self.measurement_rng.multivariate_normal(
                np.zeros(2),
                self.filter.measurement_covariance[:2, :2],
            )
            self.noisy_turtle_pose[:2] = (
                self.actual_turtle_pose[:2] + measurement_noise
            )
        self.noisy_turtle_pose[2] = self.actual_turtle_pose[2]

        # Update FOV and occlusion status
        if self.occlusions.in_occlusion(self.actual_turtle_pose[:2]):
            self.filter.is_occlusion = True
            self.filter.is_update = False
        else:
            self.filter.is_occlusion = False
            if self.is_in_FOV(self.actual_turtle_pose[:2], self.FOV):
                self.filter.is_update = True
            else:
                self.filter.is_update = False

    def update_agent_position(self, position):
        """Update quadcopter position from external source (non-ROS)"""
        self.quad_position = np.array(position[:2])
        self.FOV = self.construct_FOV(self.quad_position)

    def guidance_filter_loop(self, t, detections=[]):
        """Runs the filter loop based on the prediction method"""
        if self.guidance_mode == "Information":
            self.current_entropy()

        if not self.filter.is_update and self.guidance_mode not in {"MultiKFInfo", "MultiPFInfo"}:
            # negative information if no measurement
            self.filter.resample_index = np.where(
                np.logical_and(
                    self.is_in_FOV(self.filter.particles[-1], self.FOV),
                    ~self.occlusions.in_occlusion(self.filter.particles[-1, :, :2]),
                )
            )[0]
            if self.filter.t_since_last_update > 1.0:
                # set weights of samples close to zero
                self.filter.weights[self.filter.resample_index] = 1e-10
                # normalize weights
                self.filter.weights = self.filter.weights / np.sum(self.filter.weights)
        else:
            # Select to resample all particles if there is a measurement
            self.filter.resample_index = np.arange(self.N)

        # Run the particle filter loop
        if self.prediction_method == "Unicycle":
            self.filter.pf_loop(
                self.noisy_turtle_pose,
                t,
                self.angular_velocity,
                self.linear_velocity,
            )
        elif self.prediction_method in {"Velocity", "NN", "Transformer"}:
            self.filter.pf_loop(self.noisy_turtle_pose, t)
        elif self.prediction_method == "KF":
            if self.kf.X is None:
                # initialize the KF with the true position
                self.kf.X = np.array(
                    [self.actual_turtle_pose[0], self.actual_turtle_pose[1], 0.1, 0.1]
                )
            self.kf.predict(dt=0.333)
            if self.filter.is_update:
                self.kf.update(self.noisy_turtle_pose[:2])
                self.kf.t_since_last_update = 0.0
            else:
                self.kf.t_since_last_update += 0.333
                if self.kf.t_since_last_update > 9.0:
                    self.kf.t_since_last_update = 0.0
                    self.kf.X = np.array([0.0, 0.0, 0.05, 0.0])
            self.filter.weighted_mean = np.array([self.kf.X[0], self.kf.X[1]])
        elif self.prediction_method == "MultiKF" or self.prediction_method in MULTI_PF_METHOD_MAP:
            # Multi-target tracking
            self.multi_filter.update_time(t)  # Update tracker time
            dt = t - getattr(self, 'last_multi_time', -0.1)
            self.last_multi_time = t

            if self.prediction_method in {"MultiKF"}:
                # if a KF estimate is outside of bounds, eliminate the track
                outside_bounds = set()
                for track in self.multi_filter.tracks:
                    est_pos = self.multi_filter.get_track_states(tracks=[track])[0][:2]
                    if not np.all(
                        [
                            est_pos[0] > self.filter.APRILab_dims[0][0],
                            est_pos[0] < self.filter.APRILab_dims[1][0],
                            est_pos[1] > self.filter.APRILab_dims[0][1],
                            est_pos[1] < self.filter.APRILab_dims[1][1],
                        ]
                    ):
                        outside_bounds.add(track)
                if len(outside_bounds) > 0:
                    print(f"Removing {len(outside_bounds)} tracks outside of bounds")
                    self.multi_filter.tracks -= outside_bounds
            else:
                self.multi_filter.FOV = self.FOV

            confidences = [1.0] * len(detections)
            self.multi_filter.process_detections(detections, confidences, current_time=t)
