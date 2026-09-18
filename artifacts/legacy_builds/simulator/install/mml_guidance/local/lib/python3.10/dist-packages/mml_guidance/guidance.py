#!/usr/bin/env python3

import numpy as np
import copy
import scipy.stats as stats
from .lawnmower import LawnmowerPath
from .kf import KalmanFilter
from .ParticleFilter import ParticleFilter
from .gnn_associator import KalmanFilterGNNAssociator
from .particle_filter_gnn_associator import ParticleFilterGNNAssociator

try:
    from stable_baselines3 import PPO
    from sb3_contrib import RecurrentPPO  
    PPO_AVAILABLE = True
except ImportError:
    PPO_AVAILABLE = False
    print("Warning: stable_baselines3 not available. PPO guidance modes will not work.")


class Guidance:
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
    ):
        print(
            "Initializing Guidance with mode:",
            guidance_mode,
            "and prediction method:",
            prediction_method,
        )
        self.init_finished = False
        
        self.guidance_modes_set = {
            "Information", "WeightedMean", "Lawnmower", "Estimator", 
            "MultiKFInfo", "MultiPFInfo", "MultiPFPPO", "MultiPFInfoPPO", "MultiPFInfoPPO_LSTM"  # ADD THIS
        }
        assert guidance_mode in self.guidance_modes_set, f"Guidance mode {guidance_mode} not recognized. Choose from {self.guidance_modes_set}"
        self.guidance_mode = guidance_mode
        
        self.prediction_methods_set = {"NN", "Transformer", "Unicycle", "Velocity", "KF", "MultiKF", "MultiPFVel", "MultiPFTra"}
        self.prediction_method = prediction_method
        assert prediction_method in self.prediction_methods_set, f"Prediction method {prediction_method} not recognized. Choose from {self.prediction_methods_set}"
        self.filter = filter  
        self.N = filter.N if filter is not None else 500
        
        self.ppo_model = None
        self.is_recurrent_ppo = False   
        self.lstm_states = None  # LSTM hidden states
        
        if guidance_mode in {"MultiPFPPO", "MultiPFInfoPPO", "MultiPFInfoPPO_LSTM"}:  # ADD LSTM mode
            if not PPO_AVAILABLE:
                raise ImportError("stable_baselines3 is required for PPO guidance modes")
            if ppo_model_path is not None:
                try:
                    print(f"Loading PPO model from: {ppo_model_path}")
                    
                    # Try loading as RecurrentPPO first
                    try:
                        self.ppo_model = RecurrentPPO.load(ppo_model_path)
                        self.is_recurrent_ppo = True
                        print("✓ Loaded RecurrentPPO (LSTM) model")
                    except Exception as e:
                        # Fall back to standard PPO
                        print(f"Not a RecurrentPPO model, trying standard PPO: {e}")
                        self.ppo_model = PPO.load(ppo_model_path)
                        self.is_recurrent_ppo = False
                        print("✓ Loaded standard PPO model")
                    
                    print("PPO model loaded successfully!")
                except Exception as e:
                    print(f"Error loading PPO model: {e}")
                    raise
            else:
                raise ValueError(f"PPO model path required for guidance mode '{guidance_mode}'")
        
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

        if self.guidance_mode == "Lawnmower":
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
        elif self.prediction_method in {"MultiPFVel", "MultiPFTra"}:
            prediction_method = "Velocity" if self.prediction_method == "MultiPFVel" else "Transformer"
            # Initialize multi-target particle filter tracker
            self.multi_filter = ParticleFilterGNNAssociator(
                num_particles=200,
                prediction_method=prediction_method,
                max_association_distance=3.0,
                model_path=model_path,
                drone_height=drone_height,
                initial_time=self.initial_time
            )
            self.multi_filter.in_occlusion = self.occlusions.in_occlusion  # temp fix
            self.multi_filter.is_in_FOV = self.is_in_FOV

        self.init_finished = True

    def get_ppo_action_lstm(self, agent_state, tracked_states, lstm_states=None, episode_start=None):
        """
        Get PPO action for LSTM policy with proper state management.
        
        Args:
            agent_state: Agent state [x, y, theta, vx, vy, omega]
            tracked_states: List of track states from multi_filter.get_track_states()
            lstm_states: Previous LSTM hidden states (None to reset)
            episode_start: np.array([True/False]) indicating episode start
        
        Returns:
            action: [vx, vy, omega] if tracks exist, None if no tracks
            lstm_states: Updated LSTM states
        """
        if not self.is_recurrent_ppo:
            print("Warning: Called get_ppo_action_lstm but model is not RecurrentPPO")
            return self.get_ppo_action_nearest_track(agent_state, tracked_states), None
        
        if self.ppo_model is None:
            print("Warning: PPO model not loaded")
            return None, None
        
        if len(tracked_states) == 0:
            return None, lstm_states
        
        # Build observation (same as nearest track)
        agent_pos = agent_state[:2]
        agent_vel = agent_state[3:5]
        
        min_distance = float('inf')
        nearest_estimate = None
        
        for track_state in tracked_states:
            target_pos = track_state[:2]
            distance = np.linalg.norm(target_pos - agent_pos)
            
            if distance < min_distance:
                min_distance = distance
                if len(track_state) >= 4:
                    nearest_estimate = track_state[:4]
                else:
                    nearest_estimate = np.concatenate([track_state[:2], [0.0, 0.0]])
        
        target_pos = nearest_estimate[:2]
        target_vel = nearest_estimate[2:4]
        
        rel_pos = target_pos - agent_pos
        rel_vel = target_vel - agent_vel
        
        obs = np.array([
            agent_pos[0], agent_pos[1],
            agent_vel[0], agent_vel[1],
            rel_pos[0], rel_pos[1],
            rel_vel[0], rel_vel[1]
        ], dtype=np.float32)
        
        # Handle episode_start default
        if episode_start is None:
            episode_start = np.array([lstm_states is None])
        
        # Get action with LSTM states
        action, lstm_states = self.ppo_model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_start,
            deterministic=True
        )
        
        return action, lstm_states

    def get_ppo_action_all_tracks(self, agent_state, tracked_states, max_targets=10, 
                                   lstm_states=None, episode_start=None):
        """
        Get PPO action using ALL tracked targets with Transformer/Attention.
        Now supports both standard PPO and RecurrentPPO (LSTM).
        
        The Transformer will internally decide which targets are important
        via attention mechanism, rather than pre-selecting nearest.
        
        Args:
            agent_state: Agent state [x, y, theta, vx, vy, omega]
            tracked_states: List of track states
            max_targets: Maximum number of targets to track
            lstm_states: LSTM hidden states (for RecurrentPPO only)
            episode_start: Episode start flag (for RecurrentPPO only)
        
        Returns:
            If RecurrentPPO: (action, lstm_states)
            If standard PPO: action
        """
        if self.ppo_model is None:
            print("Warning: PPO model not loaded")
            return (None, lstm_states) if self.is_recurrent_ppo else None
        
        if len(tracked_states) == 0:
            return (None, lstm_states) if self.is_recurrent_ppo else None
        
        # ========== BUILD DICT OBSERVATION ==========
        agent_pos = agent_state[:2]
        agent_vel = agent_state[3:5]
        agent_obs = np.array([agent_pos[0], agent_pos[1], agent_vel[0], agent_vel[1]], dtype=np.float32)
        
        # ALL targets observation (sorted by distance for consistency)
        targets_obs = np.zeros((max_targets, 4), dtype=np.float32)
        mask = np.zeros(max_targets, dtype=np.float32)
        
        # Sort targets by distance (nearest first)
        target_data = []
        for track_state in tracked_states:
            target_pos = track_state[:2]
            distance = np.linalg.norm(target_pos - agent_pos)
            target_vel = track_state[2:4] if len(track_state) >= 4 else np.array([0.0, 0.0])
            target_data.append((distance, target_pos, target_vel))
        
        target_data.sort(key=lambda x: x[0])  # Sort by distance
        
        # Fill in up to max_targets
        num_tracks = min(len(target_data), max_targets)
        for i in range(num_tracks):
            _, target_pos, target_vel = target_data[i]
            rel_pos = target_pos - agent_pos
            rel_vel = target_vel - agent_vel
            
            targets_obs[i] = [rel_pos[0], rel_pos[1], rel_vel[0], rel_vel[1]]
            mask[i] = 1.0  # Valid target
        
        # Dict observation for Transformer
        obs = {
            'agent': agent_obs,
            'targets': targets_obs,
            'mask': mask
        }
        
        # ========== GET ACTION (RECURRENT vs STANDARD) ==========
        if self.is_recurrent_ppo:
            # RecurrentPPO: Handle LSTM states
            if episode_start is None:
                episode_start = np.array([lstm_states is None])
            
            action, lstm_states = self.ppo_model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_start,
                deterministic=True
            )
            return action, lstm_states
        else:
            # Standard PPO
            action, _states = self.ppo_model.predict(obs, deterministic=True)
            return action

    def get_ppo_action_nearest_track(self, agent_state, tracked_states, reset_lstm=False):
        """
        Get PPO action for tracking the nearest target.
        Supports both standard PPO and RecurrentPPO (LSTM).
        
        Args:
            agent_state: Agent state [x, y, theta, vx, vy, omega]
            tracked_states: List of track states from multi_filter.get_track_states()
            reset_lstm: Whether to reset LSTM states (e.g., on episode start)
        
        Returns:
            action: [vx, vy, omega] if tracks exist, None if no tracks
        """
        if self.ppo_model is None:
            print("Warning: PPO model not loaded")
            return None
        
        # Reset LSTM states if requested
        if reset_lstm and self.is_recurrent_ppo:
            self.lstm_states = None
        
        # Check if we have any tracks
        if len(tracked_states) == 0:
            return None  # Signal for random walk
        
        # ========== NEAREST TRACK LOGIC ==========
        agent_pos = agent_state[:2]
        agent_vel = agent_state[3:5]
        
        min_distance = float('inf')
        nearest_estimate = None
        
        # Find nearest track
        for track_state in tracked_states:
            target_pos = track_state[:2]
            distance = np.linalg.norm(target_pos - agent_pos)
            
            if distance < min_distance:
                min_distance = distance
                # Ensure we have [x, y, vx, vy] format
                if len(track_state) >= 4:
                    nearest_estimate = track_state[:4]
                else:
                    nearest_estimate = np.concatenate([track_state[:2], [0.0, 0.0]])
        
        # ========== BUILD OBSERVATION ==========
        target_pos = nearest_estimate[:2]
        target_vel = nearest_estimate[2:4]
        
        rel_pos = target_pos - agent_pos
        rel_vel = target_vel - agent_vel
        
        obs = np.array([
            agent_pos[0], agent_pos[1],
            agent_vel[0], agent_vel[1],
            rel_pos[0], rel_pos[1],
            rel_vel[0], rel_vel[1]
        ], dtype=np.float32)
        
        # ========== GET ACTION (RECURRENT vs STANDARD) ==========
        if self.is_recurrent_ppo:
            # RecurrentPPO: Pass LSTM states
            action, self.lstm_states = self.ppo_model.predict(
                obs,
                state=self.lstm_states,  # Pass previous LSTM state
                deterministic=True,
                episode_start=np.array([reset_lstm])  # Signal if episode just started
            )
        else:
            action, _states = self.ppo_model.predict(obs, deterministic=True)
        
        return action

    # Keep the old method for backward compatibility but mark as deprecated
    def get_ppo_action(self, agent_state, target_estimates):
        """
        DEPRECATED: Use get_ppo_action_nearest_track() instead.
        This method assumes you're passing the correct format.
        """
        print("Warning: get_ppo_action() is deprecated. Use get_ppo_action_nearest_track() for automatic nearest selection.")
        
        if self.ppo_model is None:
            print("Warning: PPO model not loaded, returning zero action")
            return np.array([0.0, 0.0, 0.0])
        
        # Build observation for PPO
        agent_pos = agent_state[:2]
        agent_vel = agent_state[3:5]
        
        obs = [agent_pos[0], agent_pos[1], agent_vel[0], agent_vel[1]]
        # Add relative target states
        for estimate in target_estimates:
            target_pos = estimate[:2]
            rel_pos = target_pos - agent_pos
            
            if len(estimate) >= 4:
                target_vel = estimate[2:4]
            else:
                target_vel = np.array([0.0, 0.0])
            
            rel_vel = target_vel - agent_vel
            obs.extend([rel_pos[0], rel_pos[1], rel_vel[0], rel_vel[1]])
        
        obs = np.array(obs, dtype=np.float32)
        
        # Get action from PPO policy
        action, _states = self.ppo_model.predict(obs, deterministic=True)
        
        return action

    def current_entropy(self, event=None) -> None:
        """Compute the entropy of the current distribution of particles
        Equation changes if there is a measurement available. See paper for reference
        We sample the particle distribution to reduce computation time.
        Output: Hp_t (float) - entropy of the current distribution"""

        prev_Hp = np.copy(self.Hp_t)


        # Normalize weights
        self.sampled_weights = (
            self.sampled_weights / np.sum(self.sampled_weights)
            if np.sum(self.sampled_weights) > 0
            else self.sampled_weights
        )
        sampled_prev_weights = np.copy(self.filter.prev_weights[self.sampled_index])

        if self.filter.is_update:
            Hp_t = self.entropy_particle(
                self.sampled_particles[-2],
                np.copy(sampled_prev_weights),
                self.sampled_particles[-1],
                np.copy(self.sampled_weights),
                self.noisy_turtle_pose,
            )
        else:
            Hp_t = self.entropy_particle(
                self.sampled_particles[-2],
                np.copy(
                    self.sampled_weights
                ),  # current weights are the (t-1) weights because no update
                self.sampled_particles[-1],
            )

        Hp_t = (
            prev_Hp if not np.isfinite(Hp_t) else Hp_t
        )  # if really bad value, keep the previous one
        if not np.array_equal(self.prev_Hp, np.ones((5, 1))):
            self.Hp_t = self.reject_spikes(
                self.prev_Hp,
                Hp_t,
                threshold_factor=3,
            )

        # update entropy history
        self.prev_Hp = np.roll(self.prev_Hp, -1, axis=0)
        self.prev_Hp[-1, :2] = Hp_t

    def propagate_particles(self, future_parts, dt=0.33):
        prediction_method = self.filter.prediction_method
        if prediction_method in {"NN", "Transformer"}:
            future_parts = self.filter.predict_mml(
                np.copy(future_parts), np.ones(future_parts.shape[0]) * dt
            )
        elif prediction_method == "Unicycle":
            future_parts = self.filter.predict(
                future_parts,
                dt,
                angular_velocity=self.angular_velocity,
                linear_velocity=self.linear_velocity,
            )
        elif prediction_method == "Velocity":
            future_parts = self.filter.predict(future_parts, dt)
        return future_parts

    def information_driven_guidance(self, future_parts):
        """Compute the current entropy and future entropy using particles
        to then compute the expected entropy reduction (EER) over predicted
        measurements. The next action is the  position of the particle
        that maximizes the EER.
        Output:
        eer_particle: the index of the particle that maximizes the EER which
        we propagate to choose the goal position
        """
        # Initialize variables
        future_weight = np.zeros((self.N_s, self.N_s))
        Hp_k = np.zeros(self.N_s)  # partial entropy
        EER = np.zeros(self.N_s)  # Expected Entropy Reduction

        # Future possible measurements
        # TODO: implement N_m sampled measurements (assumption: N_m = 1)
        z_hat = self.filter.add_noise(
            future_parts[-1, :, : self.filter.measurement_covariance.shape[0]],
            self.filter.measurement_covariance,
        )

        # p(z_{k+K} | x_{k+K})
        likelihood = self.filter.likelihood(future_parts[-1], z_hat)
        # TODO: implement N_m sampled measurements (double loop)
        for jj in range(self.N_s):
            k_fov = self.construct_FOV(z_hat[jj])  # assuming we are on top of measurement
            # checking for measurement outside of fov or in occlusion
            if self.is_in_FOV(z_hat[jj], k_fov) and not self.occlusions.in_occlusion(
                z_hat[jj]
            ):
                future_weight[:, jj] = self.filter.update(
                    self.sampled_weights, future_parts, z_hat[jj]
                )

                # H (x_{k+K} | \hat{z}_{k+K})
                Hp_k[jj] = self.entropy_particle(
                    future_parts[-2],
                    np.copy(self.sampled_weights),
                    future_parts[-1],
                    future_weight[:, jj],
                    z_hat[jj],
                )
            else:
                Hp_k[jj] = self.entropy_particle(
                    future_parts[-2],
                    np.copy(self.sampled_weights),
                    future_parts[-1],
                )

            # Information Gain
            EER[jj] = self.Hp_t - (Hp_k[jj] )* likelihood[jj]

        self.eer_particle = np.argmax(EER)

    def info_driven_guidance_multi_kf(self, future_estimates):
        """Compute the current entropy and future entropy using future gaussian estimates"""
        assert len(future_estimates) > 0, "No future estimates provided for info-driven guidance"
        if len(future_estimates) == 1:
            return self.multi_filter.get_track_states(tracks=future_estimates)[0][:2]
        Hp_k = [0.0] * len(future_estimates)  # partial entropy
        for ii, estimate in enumerate(future_estimates):
            # possible future measurement
            estimate_pos = self.multi_filter.get_track_states(tracks=[estimate])[0][:2]
            z_hat = self.filter.add_noise(
                estimate_pos, self.filter.measurement_covariance[:2, :2],
            )

            if self.is_in_FOV(z_hat, self.construct_FOV(estimate_pos)) and not self.occlusions.in_occlusion(z_hat):
                # posterior tracks after update with possible measurement
                future_posterior_tracks = self.multi_filter.process_detections([z_hat], confidences=None, future_tracks=future_estimates)
            else:
                future_posterior_tracks = future_estimates
            if len(future_posterior_tracks) > 0:
                track_states = self.multi_filter.get_track_states(tracks=future_posterior_tracks)
                track_covs = self.multi_filter.get_track_covariances(tracks=future_posterior_tracks)
                for jj in range(len(track_states)):
                    Hp_k[ii] += self.entropy_gaussian(track_states[jj], track_covs[jj])
            else:
                Hp_k[ii] = 1e6  # very high entropy if no tracks
        
        # choose the estimate that minimizes the entropy
        eer_track = np.argmin(Hp_k)  # I do not think I need to multiply by likelihood because it is constant wrt the action
        print("entropies:", [f"{float(e):.3f}" for e in Hp_k])
        print(f"goal track: {eer_track}")
        for ii, estimate in enumerate(future_estimates):
            if ii == eer_track:
                return self.multi_filter.get_track_states(tracks=[estimate])[0][:2]

    def info_driven_guidance_multi_pf(self, future_tracks):
        """Compute the current entropy and future entropy using future particle filter tracks"""
        assert len(future_tracks) > 0, "No future tracks provided for info-driven guidance"
        if len(future_tracks) == 1:
            # Only one track, follow it
            return future_tracks[0].weighted_mean[:2]
        
        Hp_k = [0.0] * len(future_tracks)  # partial entropy
        for ii, track in enumerate(future_tracks):
            # possible future measurement
            estimate_pos = track.weighted_mean[:2]
            z_hat = self.filter.add_noise(
                estimate_pos, self.filter.measurement_covariance[:2, :2],
            )

            if self.is_in_FOV(z_hat, self.construct_FOV(estimate_pos)) and not self.occlusions.in_occlusion(z_hat):
                # Simulate update with possible measurement
                # Create a copy of the track for future prediction
                future_track_copy = copy.deepcopy(track)
                future_track_copy.pf.weights = future_track_copy.pf.update( \
                    future_track_copy.pf.weights, future_track_copy.pf.particles, z_hat)
                
                # Compute entropy based on particle spread after update
                particles = future_track_copy.pf.particles[-1, :, :2]
                weights = future_track_copy.pf.weights
                
                # Weighted covariance
                mean = np.average(particles, weights=weights, axis=0)
                diff = particles - mean
                cov = np.average(diff[:, :, np.newaxis] * diff[:, np.newaxis, :], 
                               weights=weights, axis=0)
                
                # Entropy = 0.5 * log((2*pi*e)^n * |Sigma|)
                det_cov = np.linalg.det(cov + 1e-6 * np.eye(2))
                Hp_k[ii] = 0.5 * np.log((2 * np.pi * np.e) ** 2 * max(det_cov, 1e-6))
            else:
                # No measurement update, use current entropy
                particles = track.pf.particles[-1, :, :2]
                weights = track.pf.weights
                
                mean = np.average(particles, weights=weights, axis=0)
                diff = particles - mean
                cov = np.average(diff[:, :, np.newaxis] * diff[:, np.newaxis, :], 
                               weights=weights, axis=0)
                
                det_cov = np.linalg.det(cov + 1e-6 * np.eye(2))
                Hp_k[ii] = 0.5 * np.log((2 * np.pi * np.e) ** 2 * max(det_cov, 1e-6))
        
        # Choose the track that minimizes the entropy
        eer_track = np.argmin(Hp_k)
        # print("particle filter entropies:", [f"{float(e):.3f}" for e in Hp_k])
        # print(f"selected track: {eer_track}")

        return future_tracks[eer_track].weighted_mean[:2]

    def entropy_gaussian(self, mean, cov):
        """Compute the entropy of a gaussian distribution
        H = 0.5 * ln((2*pi*e)^n * |Sigma|)
        where n is the dimension of the gaussian and Sigma is the covariance matrix
        Output: entropy (float)"""
        n = mean.shape[0]
        det_cov = np.linalg.det(cov)
        if det_cov <= 0:
            det_cov = 1e-6  # numerical stability
        entropy = 0.5 * np.log((2 * np.pi * np.e) ** n * det_cov)
        return entropy

    def entropy_particle(
        self,
        prev_particles,
        prev_wgts,
        particles,
        wgts=np.array([]),
        z_meas=np.array([]),
    ) -> np.int64:
        """Compute the entropy of the particle distribution based on the equation in the
        paper: Y. Boers, H. Driessen, A. Bagchi, and P. Mandal, 'Particle filter based entropy'
        There are two computations, one for the case where the measurement is inside the fov
        and there is an update step before, and one where the measurement is outside the fov.
        Output:
            entropy (numpy.int64)
        """
        if wgts.size > 0:
            # likelihod of measurement p(zt|xt)
            # (how likely is each of the particles in the gaussian of the measurement)
            like_meas = stats.multivariate_normal.pdf(
                x=particles[:, :2],
                mean=z_meas[:2],
                cov=self.filter.measurement_covariance[:2, :2],
            )

            # likelihood of particle p(xt|xt-1)
            part_len, _ = particles.shape
            process_part_like = np.zeros(part_len)

            for ii in range(part_len):
                like_particle = stats.multivariate_normal.pdf(
                    x=prev_particles[:, :2],
                    mean=particles[ii, :2],
                    cov=self.filter.process_covariance[:2, :2],
                )
                process_part_like[ii] = np.sum(like_particle * prev_wgts[ii])

            # Numerical stability
            cutoff = 1e-4
            like_meas[like_meas < cutoff] = np.nan
            prev_wgts[prev_wgts < cutoff] = np.nan
            # remove the nans from the likelihoods
            # like_meas = like_meas[~np.isnan(like_meas)]
            process_part_like[process_part_like < cutoff] = np.nan

            product = like_meas * prev_wgts
            notnans = product[~np.isnan(product)]
            notnans[notnans < cutoff * 0.01] = np.nan
            product[~np.isnan(product)] = notnans
            first_term = np.log(np.nansum(product))
            # first_term = first_term if np.isfinite(first_term) else 0.0
            # second_term = np.nansum(np.log(prev_wgts)*weights)
            # third_term = np.nansum(weights*np.log(like_meas))
            # fourth_term = np.nansum(weights*np.log(process_part_like))

            entropy = (
                first_term
                - np.nansum(np.log(prev_wgts) * wgts)
                - np.nansum(wgts * np.log(like_meas))
                - np.nansum(wgts * np.log(process_part_like))
            )

            # if np.abs(entropy) > 30:  # debugging
            # print("\nEntropy term went bad :(")
            # print("second term: ", np.nansum(np.log(prev_wgts) * wgts))
            # print("third term: ", np.nansum(wgts * np.log(like_meas)))
            # print("fourth term: ", np.nansum(wgts * np.log(process_part_like)))

            if np.isinf(first_term):
                print("first term of entropy is -inf. Likelihood is very small")
        else:
            # likelihood of particle p(xt|xt-1)
            part_len2, _ = prev_particles.shape
            process_part_like = np.zeros(part_len2)
            for ii in range(part_len2):
                like_particle = stats.multivariate_normal.pdf(
                    x=prev_particles[:, :2],
                    mean=particles[ii, :2],
                    cov=self.filter.process_covariance[:2, :2],
                )
                process_part_like[ii] = np.sum(like_particle * prev_wgts)

            # Numerical stability
            cutoff = 1e-4
            process_part_like[process_part_like < cutoff] = np.nan
            prev_wgts[prev_wgts < cutoff] = np.nan

            entropy = -np.nansum(prev_wgts * np.log(process_part_like))

        return entropy

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
        """Check if the particles are in the FOV of the camera."""
        if sparticles.ndim == 1:
            return np.all(
                [
                    sparticles[0] > fov[0],
                    sparticles[0] < fov[1],
                    sparticles[1] > fov[2],
                    sparticles[1] < fov[3],
                ]
            )
        else:
            return np.logical_and.reduce(
                [
                    sparticles[:, 0] > fov[0],
                    sparticles[:, 0] < fov[1],
                    sparticles[:, 1] > fov[2],
                    sparticles[:, 1] < fov[3],
                ]
            )

    def construct_FOV(self, fov_center=np.array([0, 0])) -> np.ndarray:
        """Construct the FOV of the camera given the center
        of the FOV and the dimensions of the FOV
        Input: Center of the FOV
        Output: FOV dimensions in the form (x_min, x_max, y_min, y_max)
        """
        fov = np.array(
            [
                fov_center[0] - self.FOV_dims[0] / 2,
                fov_center[0] + self.FOV_dims[0] / 2,
                fov_center[1] - self.FOV_dims[1] / 2,
                fov_center[1] + self.FOV_dims[1] / 2,
            ]
        )
        return fov

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
                    track.pf.particles = self.propagate_particles(track.pf.particles, dt)
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
        self.noisy_turtle_pose[:2] = self.filter.add_noise(
            self.actual_turtle_pose[:2],
            self.filter.measurement_covariance[:2, :2],
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
        elif self.prediction_method in {"MultiKF", "MultiPFTra", "MultiPFVel"}:
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


class Occlusions:
    def __init__(self, positions, widths):
        """List of occlusions with helper functions"""
        print("Occlusions initialized with centers:", positions)
        print("Occlusions initialized with widths:", widths)
        self.occlusions = [(positions[ii], widths[ii]) for ii in range(len(positions))]

    def in_occlusion(self, pos):
        """Return true if the position measurement is in occlusion zones for a single
        position but if it is an array of positions, return an array of booleans for
        particles inside occlusion
        Inputs: pos: position to check - numpy.array of shape (2,)
        """
        if pos.ndim == 1:
            for position, width in self.occlusions:
                if np.all(
                    [
                        pos[0] > position[0] - width / 2,
                        pos[0] < position[0] + width / 2,
                        pos[1] > position[1] - width / 2,
                        pos[1] < position[1] + width / 2,
                    ]
                ):
                    return True
            return False
        else:
            return np.array(
                [
                    any(
                        np.all(
                            [
                                pos[ii, 0] > position[0] - width / 2,
                                pos[ii, 0] < position[0] + width / 2,
                                pos[ii, 1] > position[1] - width / 2,
                                pos[ii, 1] < position[1] + width / 2,
                            ]
                        )
                        for position, width in self.occlusions
                    )
                    for ii in range(pos.shape[0])
                ]
            )