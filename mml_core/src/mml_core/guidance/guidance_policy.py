"""PPO observation construction, action selection, and recurrent policy state."""
import numpy as np


class GuidancePolicy:
    def __init__(self):
        self.ppo_model = None
        self.is_recurrent_ppo = False
        self.lstm_states = None

    def load(self, guidance_mode, ppo_model_path, PPO_AVAILABLE, PPO=None, RecurrentPPO=None):
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
                                   lstm_states=None, episode_start=None,
                                   visibility_flags=None):
        """
        Get PPO action using ALL tracked targets with Transformer/Attention.
        Observation format matches gym_pursuit_env.py training schema exactly:
          targets slot: [rel_x, rel_y, rel_vx, rel_vy, conf]  (5-dim)
          conf = 1.0 if visible, 0.0 if lost (slot zeroed)

        Args:
            agent_state      : [x, y, theta, vx, vy, omega]
            tracked_states   : list of track states [x, y, vx, vy, ...]
            max_targets      : slots in obs tensor (must match training)
            lstm_states      : LSTM hidden states (RecurrentPPO only)
            episode_start    : np.array([True/False]) for LSTM reset
            visibility_flags : optional list[bool] per tracked_state;
                               None means all treated as visible (conf=1.0)

        Returns:
            If RecurrentPPO: (action, lstm_states)
            If standard PPO: action
            If no tracks:    (None, lstm_states) or None
        """
        if self.ppo_model is None:
            return (None, lstm_states) if self.is_recurrent_ppo else None

        if len(tracked_states) == 0:
            return (None, lstm_states) if self.is_recurrent_ppo else None

        agent_pos = agent_state[:2]
        agent_vel = agent_state[3:5]
        agent_obs = np.array([agent_pos[0], agent_pos[1],
                              agent_vel[0], agent_vel[1]], dtype=np.float32)

        targets_obs = np.zeros((max_targets, 5), dtype=np.float32)
        mask        = np.zeros(max_targets,      dtype=np.float32)

        target_data = []
        for idx, track_state in enumerate(tracked_states):
            target_pos = track_state[:2]
            distance   = np.linalg.norm(target_pos - agent_pos)
            target_vel = track_state[2:4] if len(track_state) >= 4 else np.zeros(2)
            vis = (visibility_flags[idx] if visibility_flags is not None else True)
            target_data.append((distance, target_pos, target_vel, vis))

        target_data.sort(key=lambda x: x[0])

        for slot, (_, target_pos, target_vel, vis) in enumerate(
                target_data[:max_targets]):
            if vis:
                rel_pos = target_pos - agent_pos
                rel_vel = target_vel - agent_vel
                targets_obs[slot] = [rel_pos[0], rel_pos[1],
                                     rel_vel[0], rel_vel[1], 1.0]
            else:
                targets_obs[slot] = [0., 0., 0., 0., 0.]
            mask[slot] = 1.0

        obs = {
            'agent':   agent_obs,
            'targets': targets_obs,
            'mask':    mask,
        }

        if self.is_recurrent_ppo:
            if episode_start is None:
                episode_start = np.array([lstm_states is None])
            action, lstm_states = self.ppo_model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_start,
                deterministic=True,
            )
            return action, lstm_states
        else:
            action, _ = self.ppo_model.predict(obs, deterministic=True)
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
