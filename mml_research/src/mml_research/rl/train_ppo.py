"""
PPO Training Script for Multi-Target Tracking Task
"""

from mml_core.config import project_root, motion_resource_dir, research_data_dir

import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import gymnasium as gym

from environments.gym_pursuit_env import GymPursuitEnv
from datetime import datetime

def make_env(rank, seed=0, render_mode=None, use_pf=True, track_nearest=True):
    """
    Create environment factory function.
    
    Args:
        rank: Environment index for parallel training
        seed: Random seed
        render_mode: 'human' for visualization, None for headless
        use_pf: Whether to use particle filter (True) or perfect info (False)
        track_nearest: Track only nearest target (True) or all targets (False)
    """
    def _init():
        env = GymPursuitEnv(
            render_mode=render_mode, 
            max_steps=1000,
            use_particle_filter=use_pf,
            num_particles=100,
            prediction_method="Velocity",
            track_nearest_only=track_nearest  
        )
        env.reset(seed=seed + rank)
        return env
    return _init


def train_ppo(
    total_timesteps=100_000,
    n_envs=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    save_freq=50000,
    eval_freq=10000,
    log_dir="./logs",
    model_dir="./models",
    seed=42,
    use_pf=True,
    track_nearest=True,  
):
    """
    Train a PPO agent for multi-target tracking.
    """
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print("=" * 60)
    print("PPO Training for Multi-Target Tracking")
    print("=" * 60)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Number of environments: {n_envs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"PPO epochs: {n_epochs}")
    print(f"Entropy coefficient: {ent_coef}")
    print(f"Particle Filter: {use_pf}")
    print(f"Track nearest only: {track_nearest}")  
    print("=" * 60)
    
    # Create vectorized environments
    print("\nCreating environments...")
    env = DummyVecEnv([make_env(i, seed, use_pf=use_pf, track_nearest=track_nearest) for i in range(n_envs)])
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(0, seed + 1000, use_pf=use_pf, track_nearest=track_nearest)])
    
    # Configure logger
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    
    # Define PPO model
    print("\nInitializing PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        verbose=1,
        tensorboard_log=log_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=seed,
    )
    
    # Set custom logger
    model.set_logger(new_logger)
    
    print(f"\nUsing device: {model.device}")
    print(f"Policy architecture:\n{model.policy}")
    
    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(save_freq // n_envs, 1),
        save_path=model_dir,
        name_prefix="ppo_tracking",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, "best_model"),
        log_path=log_dir,
        eval_freq=max(eval_freq // n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )
    
    callback = CallbackList([checkpoint_callback, eval_callback])
    
    
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_model_path = os.path.join(model_dir, f"ppo_tracking_final_{timestamp}")
        model.save(final_model_path)
        print(f"\n\nFinal model saved to: {final_model_path}")
        
        env.close()
        eval_env.close()
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    
    return model


def test_trained_model(model_path, n_episodes=5, render=True, track_nearest=True):
    """
    Test a trained model.
    """
    print("\n" + "=" * 60)
    print(f"Testing trained model: {model_path}")
    print("=" * 60)
    
    
    model = PPO.load(model_path)
    
    
    env = GymPursuitEnv(
        render_mode="human" if render else None, 
        max_steps=1000,
        use_particle_filter=True,
        num_particles=100,
        prediction_method="Velocity",
        track_nearest_only=track_nearest
    )
    
    if render:
        env.env.render_init()
    
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        truncated = False
        
        print(f"\nEpisode {episode + 1}/{n_episodes}")
        
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"  Reward: {episode_reward:.2f}, Length: {episode_length}")
    
    env.close()
    
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print(f"  Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Mean length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train PPO for multi-target tracking")
    parser.add_argument("--mode", type=str, default="test", choices=["train", "test"],
                        help="Mode: train or test")
    parser.add_argument("--timesteps", type=int, default=10_000,
                        help="Total training timesteps")
    parser.add_argument("--n_envs", type=int, default=1,
                        help="Number of parallel environments (use 1 on Windows)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--n_epochs", type=int, default=10,
                        help="Number of PPO epochs")
    parser.add_argument("--ent_coef", type=float, default=0.01,
                        help="Entropy coefficient")
    parser.add_argument("--use_pf", action="store_true", default=True,
                        help="Use particle filter (default: True)")
    parser.add_argument("--no_pf", dest="use_pf", action="store_false",
                        help="Disable particle filter (perfect information)")
    parser.add_argument("--track_nearest", action="store_true", default=True,
                        help="Track only nearest target (default: True)")
    parser.add_argument("--track_all", dest="track_nearest", action="store_false",
                        help="Track all targets")
    parser.add_argument("--model_path", type=str, default="./models/ppo_tracking_final.zip",
                        help="Path to model for testing")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        
        model = train_ppo(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            ent_coef=args.ent_coef,
            seed=args.seed,
            use_pf=args.use_pf,
            track_nearest=args.track_nearest,  
        )
    else:
        
        test_trained_model(
            args.model_path, 
            n_episodes=5, 
            render=True,
            track_nearest=args.track_nearest  
        )