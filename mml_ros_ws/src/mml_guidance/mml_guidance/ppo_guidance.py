"""
PPO-based guidance for multi-target tracking.
Integrates trained PPO policy with particle filter tracking.
"""
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os


class PPOGuidance:
    """
    PPO-based guidance that uses a trained policy to select goal positions.
    
    The PPO model observes particle filter estimates and outputs velocity commands,
    which are converted to goal positions for the agent.
    """
    
    def __init__(self, model_path, vecnorm_path=None, num_targets=1, drone_height=2.0):
        """
        Initialize PPO guidance.
        
        Args:
            model_path: Path to trained PPO model (.zip file)
            vecnorm_path: Path to VecNormalize stats (.pkl file)
            num_targets: Number of targets being tracked
            drone_height: Height of drone (for FOV calculations)
        """
        self.num_targets = num_targets
        self.drone_height = drone_height
        
        # Load PPO model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"PPO model not found: {model_path}")
        
        print(f"Loading PPO model from: {model_path}")
        self.model = PPO.load(model_path)
        
        # Load normalization if available
        self.vec_normalize = None
        if vecnorm_path and os.path.exists(vecnorm_path):
            print(f"Loading VecNormalize from: {vecnorm_path}")
            # Create dummy env for VecNormalize (won't actually step it)
            dummy_env = DummyVecEnv([lambda: None])
            self.vec_normalize = VecNormalize.load(vecnorm_path, dummy_env)
            self.vec_normalize.training = False
            self.vec_normalize.norm_reward = False
        else:
            print("⚠ No VecNormalize found, using raw observations")
        
        # State tracking
        self.last_obs = None
        self.last_goal_position = np.array([0.0, 0.0])
        self.dt = 0.1  # Time step for velocity integration
        
    def compute_observation(self, agent_position, tracked_states, tracked_covs):
        """
        Compute observation for PPO model.
        
        Observation format: [dx1, dy1, log_cov_det1, dx2, dy2, log_cov_det2, ...]
        
        Args:
            agent_position: [x, y] position of agent (meters)
            tracked_states: List of [x, y, ...] tracked target states
            tracked_covs: List of 2x2 covariance matrices
            
        Returns:
            obs: Observation array for PPO model
        """
        obs = []
        
        for i in range(self.num_targets):
            if i < len(tracked_states):
                # Target is tracked
                target_pos = tracked_states[i][:2]
                cov = tracked_covs[i] if i < len(tracked_covs) else np.eye(2)
                
                # Relative position
                dx = target_pos[0] - agent_position[0]
                dy = target_pos[1] - agent_position[1]
                
                # Log determinant of covariance (uncertainty measure)
                cov_det = np.linalg.det(cov)
                log_cov_det = np.log(cov_det + 1e-6)
                
                obs.extend([dx, dy, log_cov_det])
            else:
                # No track for this target - use placeholder
                obs.extend([0.0, 0.0, 10.0])  # High uncertainty
        
        return np.array(obs, dtype=np.float32)
    
    def predict_velocity_command(self, agent_position, tracked_states, tracked_covs, 
                              deterministic=True):
        """
        Use PPO policy to predict velocity command DIRECTLY.
        
        Returns:
            tau: [vx, vy, omega] velocity command (NOT goal position!)
        """
        # Compute observation
        obs = self.compute_observation(agent_position, tracked_states, tracked_covs)
        
        # Normalize if VecNormalize is available
        if self.vec_normalize is not None:
            obs = self.vec_normalize.normalize_obs(obs)
        
        # Get action from PPO model
        action, _ = self.model.predict(obs, deterministic=deterministic)
        
        # ✅ CORRECT: Return velocity command directly
        # action is [vx, vy, omega] - this IS tau!
        self.last_action = action
        self.last_obs = obs
        
        return action  # [vx, vy, omega] in m/s and rad/s
    
    def get_info(self):
        """Get diagnostic information about PPO guidance."""
        return {
            "model_loaded": self.model is not None,
            "normalized": self.vec_normalize is not None,
            "last_goal": self.last_goal_position,
            "num_targets": self.num_targets
        }