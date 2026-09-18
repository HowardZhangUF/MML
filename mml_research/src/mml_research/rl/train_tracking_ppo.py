"""
Recurrent PPO Training Script with LSTM + Attention/Transformer
Adds temporal modeling through LSTM hidden states
"""

from mml_core.config import project_root, motion_resource_dir, research_data_dir

import os
import numpy as np
import torch
from sb3_contrib import RecurrentPPO  # ← NEW: Use RecurrentPPO instead of PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.logger import configure

from environments.gym_pursuit_env import GymPursuitEnv
from datetime import datetime


def make_env(rank, seed=0, render_mode=None, use_pf=True, track_nearest=False, 
             use_attention=False, max_targets=10):
    """
    Create environment factory function.
    """
    def _init():
        env = GymPursuitEnv(
            render_mode=render_mode, 
            max_steps=1000,
            use_particle_filter=use_pf,
            num_particles=100,
            prediction_method="Velocity",
            track_nearest_only=track_nearest,
            use_attention=use_attention,
            max_tracked_targets=max_targets
        )
        env.reset(seed=seed + rank)
        return env
    return _init


def train_recurrent_ppo(
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
    log_dir="./logs_recurrent",  # ← Different log dir
    model_dir="./models_recurrent",  # ← Different model dir
    seed=42,
    use_pf=True,
    track_nearest=False,
    use_attention=True,  # ← Should be True for best results
    max_targets=10,
    # ========== NEW LSTM PARAMETERS ==========
    lstm_hidden_size=256,  # LSTM hidden state dimension
    n_lstm_layers=1,       # Number of LSTM layers
    enable_critic_lstm=True,  # Use LSTM for critic network too
    shared_lstm=False,     # Separate LSTMs for actor and critic
):
    """
    Train a Recurrent PPO agent with LSTM for temporal modeling.
    """
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print("=" * 60)
    print("Recurrent PPO Training (LSTM + Attention)")
    print("=" * 60)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Number of environments: {n_envs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"PPO epochs: {n_epochs}")
    print(f"Entropy coefficient: {ent_coef}")
    print(f"Particle Filter: {use_pf}")
    print(f"Track nearest only: {track_nearest}")
    print(f"Use Attention: {use_attention}")
    print(f"LSTM hidden size: {lstm_hidden_size}")  # NEW
    print(f"LSTM layers: {n_lstm_layers}")  # NEW
    if use_attention:
        print(f"Max tracked targets: {max_targets}")
    print("=" * 60)
    
    # Create vectorized environments
    print("\nCreating environments...")
    env = DummyVecEnv([
        make_env(i, seed, use_pf=use_pf, track_nearest=track_nearest, 
                 use_attention=use_attention, max_targets=max_targets) 
        for i in range(n_envs)
    ])
    
    # Create evaluation environment
    eval_env = DummyVecEnv([
        make_env(0, seed + 1000, use_pf=use_pf, track_nearest=track_nearest,
                 use_attention=use_attention, max_targets=max_targets)
    ])
    
    # Configure logger
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    
    # ========== RECURRENT POLICY KWARGS ==========
    policy_kwargs = None
    policy_type = None
    
    if use_attention:
        # Import attention feature extractor
        from attention_policy import AttentionFeatureExtractor
        
        policy_kwargs = dict(
            # Spatial feature extraction (your existing attention code)
            features_extractor_class=AttentionFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=128),
            
            # ========== LSTM CONFIGURATION ==========
            lstm_hidden_size=lstm_hidden_size,  # Hidden state size
            n_lstm_layers=n_lstm_layers,        # Number of LSTM layers
            enable_critic_lstm=enable_critic_lstm,  # Use LSTM for value network
            shared_lstm=shared_lstm,            # Share LSTM between actor/critic
            
            # Actor/Critic network architecture (after LSTM)
            net_arch=dict(
                pi=[256, 256],  # Actor: LSTM → 256 → 256 → action
                vf=[256, 256]   # Critic: LSTM → 256 → 256 → value
            ),
        )
        policy_type = "MultiInputLstmPolicy"  # ← LSTM policy for Dict observations
        print("\n✓ Using Recurrent Policy: Attention + LSTM")
    
    else:
        # Standard LSTM without attention
        policy_kwargs = dict(
            lstm_hidden_size=lstm_hidden_size,
            n_lstm_layers=n_lstm_layers,
            enable_critic_lstm=enable_critic_lstm,
            shared_lstm=shared_lstm,
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
        )
        policy_type = "MlpLstmPolicy"  # ← LSTM policy for flat observations
        print("\n✓ Using Recurrent Policy: MLP + LSTM")
    
    # ========== INITIALIZE RECURRENT PPO ==========
    print("\nInitializing RecurrentPPO model...")
    model = RecurrentPPO(  # ← Use RecurrentPPO instead of PPO!
        policy_type,
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
        policy_kwargs=policy_kwargs,
    )
    
    # Set custom logger
    model.set_logger(new_logger)
    
    print(f"\nUsing device: {model.device}")
    print(f"Policy architecture:\n{model.policy}")
    
    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(save_freq // n_envs, 1),
        save_path=model_dir,
        name_prefix="recurrent_ppo_tracking",  # ← Different prefix
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
    
    # Train the agent
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
        final_model_path = os.path.join(model_dir, f"recurrent_ppo_tracking_final_{timestamp}")
        model.save(final_model_path)
        print(f"\n\nFinal model saved to: {final_model_path}")
        
        env.close()
        eval_env.close()
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    
    return model


def test_trained_model(model_path, n_episodes=5, render=True, track_nearest=False, 
                       use_attention=True, max_targets=10):
    """
    Test a trained recurrent model.
    """
    print("\n" + "=" * 60)
    print(f"Testing trained recurrent model: {model_path}")
    print("=" * 60)
    
    # Load recurrent model
    model = RecurrentPPO.load(model_path)  # ← Use RecurrentPPO
    
    # Create environment with same settings as training
    env = GymPursuitEnv(
        render_mode="human" if render else None, 
        max_steps=1000,
        use_particle_filter=True,
        num_particles=100,
        prediction_method="Velocity",
        track_nearest_only=track_nearest,
        use_attention=use_attention,
        max_tracked_targets=max_targets
    )
    
    if render:
        env.env.render_init()
    
    # Test episodes
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        truncated = False
        
        # ========== RESET LSTM STATE ==========
        lstm_states = None  # Initialize LSTM hidden state
        
        print(f"\nEpisode {episode + 1}/{n_episodes}")
        
        while not (done or truncated):
            # ========== PREDICT WITH LSTM STATE ==========
            action, lstm_states = model.predict(
                obs, 
                state=lstm_states,  # ← Pass LSTM state
                deterministic=True,
                episode_start=np.array([done or truncated])  # Reset on episode start
            )
            
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"  Reward: {episode_reward:.2f}, Length: {episode_length}")
    
    env.close()
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Test Results:")
    print(f"  Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Mean length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Recurrent PPO with LSTM")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--timesteps", type=int, default=500_000)  # ← More timesteps for LSTM
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--use_pf", action="store_true", default=True)
    parser.add_argument("--no_pf", dest="use_pf", action="store_false")
    parser.add_argument("--track_nearest", action="store_true", default=False)
    parser.add_argument("--use_attention", action="store_true", default=True)
    parser.add_argument("--max_targets", type=int, default=10)
    
    # ========== NEW LSTM ARGUMENTS ==========
    parser.add_argument("--lstm_hidden_size", type=int, default=256,
                        help="LSTM hidden state dimension")
    parser.add_argument("--n_lstm_layers", type=int, default=1,
                        help="Number of LSTM layers")
    parser.add_argument("--enable_critic_lstm", action="store_true", default=True,
                        help="Use LSTM for critic network")
    parser.add_argument("--shared_lstm", action="store_true", default=False,
                        help="Share LSTM between actor and critic")
    
    parser.add_argument("--model_path", type=str, 
                        default="./models_recurrent/recurrent_ppo_tracking_final.zip")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    if args.mode == "train":
        # Train recurrent model
        model = train_recurrent_ppo(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            ent_coef=args.ent_coef,
            seed=args.seed,
            use_pf=args.use_pf,
            track_nearest=args.track_nearest,
            use_attention=args.use_attention,
            max_targets=args.max_targets,
            # LSTM parameters
            lstm_hidden_size=args.lstm_hidden_size,
            n_lstm_layers=args.n_lstm_layers,
            enable_critic_lstm=args.enable_critic_lstm,
            shared_lstm=args.shared_lstm,
        )
    else:
        # Test recurrent model
        test_trained_model(
            args.model_path, 
            n_episodes=5, 
            render=True,
            track_nearest=args.track_nearest,
            use_attention=args.use_attention,
            max_targets=args.max_targets,
        )