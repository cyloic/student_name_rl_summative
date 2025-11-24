# training/dqn_training.py

import sys
import os
# FIX: Add the project root (one level up) to the system path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from environment.custom_env import SmartSortEnv
import os

def train_dqn(timesteps=50000, lr=1e-4, gamma=0.99, exploration_fraction=0.1, model_name="DQN_baseline"):
    # Create the environment. Use vectorized env for SB3 compatibility
    env = make_vec_env(SmartSortEnv, n_envs=1)
    
    # Define model storage path
    model_path = os.path.join("models", "dqn", model_name)
    os.makedirs(model_path, exist_ok=True)
    
    print(f"--- Training {model_name} (DQN) ---")
    
    # Initialize the DQN agent
    model = DQN("MlpPolicy", 
                env, 
                learning_rate=lr, 
                gamma=gamma, 
                exploration_fraction=exploration_fraction,
                verbose=1, 
                tensorboard_log="./tensorboard_logs/dqn")
    
    # Train the agent
    model.learn(total_timesteps=timesteps, log_interval=4)
    
    # Save the trained model
    model.save(os.path.join(model_path, "final_model.zip"))
    print(f"Model saved to {model_path}")
    
    # Clean up the environment
    env.close()
    return model

if __name__ == '__main__':
    # Hyperparameter Tuning Grid (12 runs)
    lrs = [1e-4, 5e-4]
    gammas = [0.95, 0.99, 0.999]
    exploration_fractions = [0.1, 0.3]
    
    run_counter = 1
    
    for lr in lrs:
        for gamma in gammas:
            for expl_frac in exploration_fractions:
                # Create a unique name for logging and saving
                model_name = f"DQN_Run_{run_counter}_LR{lr:.0e}_G{gamma}_E{expl_frac}"
                print(f"\n--- Starting {model_name} ---")
                train_dqn(
                    timesteps=50000, 
                    lr=lr, 
                    gamma=gamma, 
                    exploration_fraction=expl_frac,
                    model_name=model_name
                )
                run_counter += 1
    print("\n\nDQN Hyperparameter Tuning Complete.")