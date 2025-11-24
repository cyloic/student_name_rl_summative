# main.py

import sys
import os
import time
# Fix path for sub-module imports: Adds the project root to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))) 

from stable_baselines3 import DQN, PPO, A2C # Import all possible algorithms
from environment.custom_env import SmartSortEnv
# time is already imported

# --- FINAL CONFIGURATION FOR CHAMPION AGENT (Mean Reward: 924.9) ---
# This specific DQN run performed best in your hyperparameter tuning.
ALGORITHM = DQN      
ALGORITHM_DIR = "dqn" 
MODEL_NAME = "DQN_Run_10_LR5e-04_G0.99_E0.3" 
# -------------------------------------------------------------------

def get_model_path(model_name):
    """Dynamically determines the path and checks for the final_model.zip file."""
    
    # Determine the folder for the algorithm type
    if ALGORITHM == DQN:
        alg_folder = "dqn"
    elif ALGORITHM in [PPO, A2C]:
        alg_folder = "pg"
    else:
        return None, "Invalid ALGORITHM specified in main.py."

    # Construct the full expected path
    path = os.path.join("models", alg_folder, model_name, "final_model.zip")
    
    if os.path.exists(path):
        return path, None
    else:
        return None, f"Model file 'final_model.zip' not found in expected location: {path}"


def run_best_agent():
    """
    Loads the best-performing trained agent and runs it for 10 episodes
    for the required video demonstration.
    """
    
    model_path, error = get_model_path(MODEL_NAME)

    if error:
        print(f"FATAL ERROR: {error}")
        print("Please check that the folder 'models/dqn/DQN_Run_10_LR5e-04_G0.99_E0.3' exists and contains 'final_model.zip'.")
        return

    # Load the trained agent
    print(f"Loading champion model: {MODEL_NAME}...")
    try:
        model = ALGORITHM.load(model_path)
    except Exception as e:
        print(f"FATAL ERROR LOADING MODEL: {e}")
        print("The file may be corrupted. Try re-running the training script for this run.")
        return
    
    # Create the environment for visualization
    env = SmartSortEnv(max_steps=15)
    
    print("\n--- Running Trained Agent Simulation (Look for Pygame Window) ---")
    print("Agent should consistently classify in 2-3 steps.")
    
    # Run the simulation for 10 episodes
    for episode in range(1, 11):
        obs, info = env.reset()
        done = False
        steps = 0
        total_reward = 0
        
        true_class_name = 'Plastic' if info['true_class'] == 0 else 'Paper'
        
        print(f"\nEpisode {episode}: True Class is {true_class_name}")

        while not done:
            # Agent takes the optimal action (deterministic=True)
            action, _ = model.predict(obs, deterministic=True) 
            obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            steps += 1
            total_reward += reward
            
            # Render the environment frame
            env.render()
            time.sleep(0.3) 

        # The final reward should be high (900+) and steps should be low (2-3)
        print(f"Result: Classified in {steps} steps. Final Reward: {total_reward:.2f}")

    env.close()

if __name__ == '__main__':
    run_best_agent()