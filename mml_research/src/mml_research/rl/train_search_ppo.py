"""
Train Recurrent PPO for SEARCH task.

This script trains a separate policy for searching when particle filter loses tracks.
Uses the same PygameEnvironment core but with different observation and reward.
"""

from mml_core.config import project_root, motion_resource_dir, research_data_dir

import os
import sys
import argparse
import torch
import numpy as np

# Add project root to path

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.logger import configure
from environments.gym_search_env import GymSearchEnv
from datetime import datetime


def make_env(rank, seed=0, render_mode=None):
    """Create environment factory function."""
    def _init():
        env = GymSearchEnv(
            render_mode=render_mode,
            max_steps=500
        )
        env.reset(seed=seed + rank)
        return env
    return _init


def train_search_ppo(
    total_timesteps=500_000,
    n_envs=4,
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
    log_dir="./logs/search",
    model_dir="./models/search",
    seed=42,
    lstm_hidden_size=256,
    n_lstm_layers=1,
    enable_critic_lstm=True,
):
    """
    Train Recurrent PPO for search task.
    """
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print("=" * 60)
    print("Recurrent PPO Search Training")
    print("=" * 60)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Number of environments: {n_envs}")
    print(f"Learning rate: {learning_rate}")
    print(f"LSTM hidden size: {lstm_hidden_size}")
    print("=" * 60)
    
    # TODO: Implement training logic
    # 1. Create vectorized environments
    # 2. Configure RecurrentPPO with LSTM
    # 3. Set up callbacks
    # 4. Train the model
    # 5. Save the model
    
    print("\nTODO: Complete implementation of training loop")
    print("See train_tracking_ppo.py for reference")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Recurrent PPO for Search")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    train_search_ppo(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        learning_rate=args.lr,
        seed=args.seed
    )
