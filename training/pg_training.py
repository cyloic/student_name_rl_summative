# training/pg_training.py

import sys
import os
# FIX: Add the project root (one level up) to the system path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from environment.custom_env import SmartSortEnv
import os

def train_pg(algorithm, timesteps=50000, lr=3e-4, gamma=0.99, n_steps=2048, model_name="PG_baseline"):
    # Create the environment
    env = make_vec_env(SmartSortEnv, n_envs=1)
    
    # Define model storage path
    alg_dir = "ppo" if algorithm == PPO else "a2c"
    model_path = os.path.join("models", "pg", alg_dir, model_name)
    os.makedirs(model_path, exist_ok=True)
    
    print(f"--- Training {algorithm.__name__} ({model_name}) ---")
    
    # Initialize the Policy Gradient agent
    if algorithm == PPO:
        model = PPO("MlpPolicy", 
                    env, 
                    learning_rate=lr, 
                    gamma=gamma, 
                    n_steps=n_steps,
                    verbose=1, 
                    tensorboard_log="./tensorboard_logs/pg")
    elif algorithm == A2C:
        model = A2C("MlpPolicy", 
                    env, 
                    learning_rate=lr, 
                    gamma=gamma, 
                    n_steps=n_steps,
                    verbose=1, 
                    tensorboard_log="./tensorboard_logs/pg")
    else:
        raise ValueError("Invalid algorithm specified.")
    
    # Train the agent
    model.learn(total_timesteps=timesteps, log_interval=4)
    
    # Save the trained model
    model.save(os.path.join(model_path, "final_model.zip"))
    print(f"Model saved to {model_path}")
    
    # Clean up the environment
    env.close()
    return model

if __name__ == '__main__':
    # PPO and A2C Tuning Grid (8 runs each = 16 runs total)
    lrs = [3e-4, 1e-3]
    gammas = [0.95, 0.99]
    n_steps_list = [512, 1024]
    
    # --- PPO TUNING ---
    ppo_run_counter = 1
    for lr in lrs:
        for gamma in gammas:
            for n_steps in n_steps_list:
                model_name = f"PPO_Run_{ppo_run_counter}_LR{lr:.0e}_G{gamma}_N{n_steps}"
                print(f"\n--- Starting {model_name} ---")
                train_pg(algorithm=PPO, timesteps=50000, lr=lr, gamma=gamma, n_steps=n_steps, model_name=model_name)
                ppo_run_counter += 1
    
    # --- A2C TUNING ---
    a2c_run_counter = 1
    for lr in lrs:
        for gamma in gammas:
            for n_steps in n_steps_list:
                model_name = f"A2C_Run_{a2c_run_counter}_LR{lr:.0e}_G{gamma}_N{n_steps}"
                print(f"\n--- Starting {model_name} ---")
                train_pg(algorithm=A2C, timesteps=50000, lr=lr, gamma=gamma, n_steps=n_steps, model_name=model_name)
                a2c_run_counter += 1
                
    print("\n\nPPO and A2C Hyperparameter Tuning Complete.")