# environment/custom_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
# The renderer import should be relative
from .rendering import SmartSortRenderer 

class SmartSortEnv(gym.Env):
    """
    RL Environment for Sequential Waste Classification (SmartSort).
    The agent chooses feature refinement steps (Action 0 or 1) before making a final classification (Action 2 or 3).
    """
    def __init__(self, true_class=0, max_steps=10):
        super(SmartSortEnv, self).__init__()

        # 0: Plastic, 1: Paper (The ground truth for this episode)
        self.true_class = true_class 
        self.max_steps = max_steps
        self.current_step = 0
        
        # Define the 4 actions: 2 refinement actions, 2 terminal classification actions
        # 0: Focus Texture, 1: Focus Shape, 2: Classify Plastic, 3: Classify Paper
        self.action_space = spaces.Discrete(4)
        
        # Observation Space: [Texture Feature, Shape Feature, Confidence_Plastic, Confidence_Paper]
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        
        self.state = None
        self.metadata['render_modes'] = ['human']
        self.renderer = None 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # MOCK INITIAL AMBIGUOUS STATE: features are evenly split
        # Randomly choose the true class (0 or 1)
        self.true_class = self.np_random.choice([0, 1]) 
        
        # State: [Texture, Shape, Conf_Plastic, Conf_Paper]
        # Start confidences high and ambiguous (0.5)
        self.state = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        
        info = {"true_class": self.true_class}
        return self.state, info

    def step(self, action):
        self.current_step += 1
        reward = 0
        terminated = False
        info = {"true_class": self.true_class}

        # --- Refinement Actions (0 and 1) ---
        if action < 2:
            reward = -10 # Small penalty for time/latency

            # MOCK FEATURE UPDATE LOGIC: True class confidence gets a boost, false class degrades
            boost = 0.1 * self.np_random.uniform(0.5, 1.5)
            degrade = 0.05 * self.np_random.uniform(0.5, 1.5)

            # True class confidence/feature improves
            true_conf_idx = 2 + self.true_class
            self.state[true_conf_idx] = min(1.0, self.state[true_conf_idx] + boost)
            
            # False class confidence/feature degrades
            false_conf_idx = 2 + (1 - self.true_class)
            self.state[false_conf_idx] = max(0.0, self.state[false_conf_idx] - degrade)
            
            # Small reward for improving confidence
            reward += 5 
            
            self.state = np.clip(self.state, 0.0, 1.0)
            
        # --- Terminal Classification Actions (2 and 3) ---
        else:
            predicted_class = action - 2
            terminated = True
            
            if predicted_class == self.true_class:
                # Correct Classification
                reward = 1000 
            else:
                # Incorrect Classification
                reward = -1000

        # --- Max Steps Termination ---
        if self.current_step >= self.max_steps and not terminated:
            terminated = True
            reward -= 50 # Penalty for failing to classify in time

        return self.state, reward, terminated, False, info

    def render(self):
        if self.renderer is None:
            self.renderer = SmartSortRenderer()
        
        self.renderer.render_frame(self.state, self.true_class, self.current_step)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None