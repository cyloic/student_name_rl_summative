#!/usr/bin/env python3
"""
SmartSort Environment - Interactive GUI Visualization
Demonstrates real-time agent decision-making and environment dynamics
"""

import tkinter as tk
from tkinter import ttk
import threading
import numpy as np
from stable_baselines3 import DQN
from environment.custom_env import SmartSortEnv


class SmartSortGUI:
    """Interactive GUI for visualizing SmartSort environment and DQN agent"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("SmartSort - RL Environment Visualization")
        self.root.geometry("1200x600")
        
        # Initialize environment and model
        self.env = SmartSortEnv()
        try:
            self.model = DQN.load("models/dqn/DQN_Run_10_LR5e-04_G0.99_E0.3/final_model")
        except:
            self.model = DQN.load("models/dqn/DQN_50k_baseline/final_model")
        
        # State variables
        self.obs = None
        self.episode_num = 0
        self.step_num = 0
        self.total_reward = 0.0
        self.episode_history = []
        self.running = False
        self.speed = 1.0
        self.episode_count = 0
        self.total_episodes = 0
        self.total_reward_sum = 0.0
        self.best_reward = -float('inf')
        self.worst_reward = float('inf')
        
        self.setup_ui()
        self.reset_episode()
        
    def setup_ui(self):
        """Create the GUI layout with 3 panels"""
        
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # LEFT PANEL - Visualization
        left_frame = ttk.LabelFrame(main_frame, text="Visualization", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.canvas = tk.Canvas(left_frame, bg='white', height=300)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # MIDDLE PANEL - Controls
        middle_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        middle_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Episode info
        info_frame = ttk.Frame(middle_frame)
        info_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(info_frame, text="Episode:").pack(side=tk.LEFT)
        self.episode_label = ttk.Label(info_frame, text="0", font=("Arial", 12, "bold"))
        self.episode_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(info_frame, text="Step:").pack(side=tk.LEFT, padx=(20, 0))
        self.step_label = ttk.Label(info_frame, text="0", font=("Arial", 12, "bold"))
        self.step_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(info_frame, text="Reward:").pack(side=tk.LEFT, padx=(20, 0))
        self.reward_label = ttk.Label(info_frame, text="0.0", font=("Arial", 12, "bold"), foreground="green")
        self.reward_label.pack(side=tk.LEFT, padx=5)
        
        # State display
        state_frame = ttk.LabelFrame(middle_frame, text="Current State", padding=5)
        state_frame.pack(fill=tk.X, pady=10)
        
        self.state_labels = []
        state_names = ["Texture", "Shape", "Plastic Conf", "Paper Conf"]
        for i, name in enumerate(state_names):
            label_frame = ttk.Frame(state_frame)
            label_frame.pack(fill=tk.X, pady=3)
            ttk.Label(label_frame, text=f"{name}:", width=15).pack(side=tk.LEFT)
            self.state_labels.append(ttk.Label(label_frame, text="0.00", font=("Arial", 10, "bold")))
            self.state_labels[-1].pack(side=tk.LEFT, padx=5)
        
        # Action display
        action_frame = ttk.Frame(middle_frame)
        action_frame.pack(fill=tk.X, pady=10)
        ttk.Label(action_frame, text="Action:").pack(side=tk.LEFT)
        self.action_label = ttk.Label(action_frame, text="-", font=("Arial", 11, "bold"), foreground="blue")
        self.action_label.pack(side=tk.LEFT, padx=5)
        
        # Buttons
        button_frame = ttk.Frame(middle_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="▶ Start Episode", command=self.start_episode).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="⏯ Auto-Run (10)", command=self.auto_run_episodes).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="↻ Reset", command=self.reset_stats).pack(side=tk.LEFT, padx=2)
        
        # Speed control
        speed_frame = ttk.Frame(middle_frame)
        speed_frame.pack(fill=tk.X, pady=10)
        ttk.Label(speed_frame, text="Speed:").pack(side=tk.LEFT)
        self.speed_label = ttk.Label(speed_frame, text="1.0x")
        self.speed_slider = ttk.Scale(speed_frame, from_=0.1, to=2.0, orient=tk.HORIZONTAL, command=self.set_speed)
        self.speed_slider.set(1.0)
        self.speed_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.speed_label.pack(side=tk.LEFT)
        
        # RIGHT PANEL - Statistics
        right_frame = ttk.LabelFrame(main_frame, text="Statistics", padding=10)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Stats display
        stats_frame = ttk.Frame(right_frame)
        stats_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(stats_frame, text="Total Episodes:").pack(fill=tk.X)
        self.total_episodes_label = ttk.Label(stats_frame, text="0", font=("Arial", 11, "bold"))
        self.total_episodes_label.pack(fill=tk.X, padx=20)
        
        ttk.Label(stats_frame, text="Avg Reward:").pack(fill=tk.X, pady=(10, 0))
        self.avg_reward_label = ttk.Label(stats_frame, text="0.0", font=("Arial", 11, "bold"), foreground="green")
        self.avg_reward_label.pack(fill=tk.X, padx=20)
        
        ttk.Label(stats_frame, text="Best Reward:").pack(fill=tk.X, pady=(10, 0))
        self.best_reward_label = ttk.Label(stats_frame, text="0.0", font=("Arial", 11, "bold"), foreground="darkgreen")
        self.best_reward_label.pack(fill=tk.X, padx=20)
        
        ttk.Label(stats_frame, text="Worst Reward:").pack(fill=tk.X, pady=(10, 0))
        self.worst_reward_label = ttk.Label(stats_frame, text="0.0", font=("Arial", 11, "bold"), foreground="red")
        self.worst_reward_label.pack(fill=tk.X, padx=20)
        
        ttk.Label(stats_frame, text="Success Rate:").pack(fill=tk.X, pady=(10, 0))
        self.success_rate_label = ttk.Label(stats_frame, text="0%", font=("Arial", 11, "bold"), foreground="blue")
        self.success_rate_label.pack(fill=tk.X, padx=20)
        
        # Episode history
        history_frame = ttk.LabelFrame(right_frame, text="Episode History", padding=5)
        history_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create scrollbar and listbox
        scrollbar = ttk.Scrollbar(history_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.history_listbox = tk.Listbox(history_frame, yscrollcommand=scrollbar.set, height=10)
        self.history_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.history_listbox.yview)
        
    def reset_episode(self):
        """Reset environment for new episode"""
        self.obs, _ = self.env.reset()
        self.step_num = 0
        self.total_reward = 0.0
        self.update_visualization()
        
    def reset_stats(self):
        """Reset all statistics"""
        self.episode_num = 0
        self.total_episodes = 0
        self.total_reward_sum = 0.0
        self.best_reward = -float('inf')
        self.worst_reward = float('inf')
        self.episode_history = []
        self.history_listbox.delete(0, tk.END)
        self.update_statistics()
        self.reset_episode()
        
    def set_speed(self, value):
        """Set animation speed"""
        self.speed = float(value)
        self.speed_label.config(text=f"{self.speed:.1f}x")
        
    def update_visualization(self):
        """Draw the 4 state bars on canvas"""
        if self.obs is None:
            return
            
        self.canvas.delete("all")
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        
        if width == 1:  # Canvas not yet rendered
            width = 300
        if height == 1:
            height = 300
        
        # Bar dimensions
        bar_width = width // 5
        bar_height = height * 0.7
        colors = ["#4169E1", "#9370DB", "#32CD32", "#FF6347"]  # Blue, Purple, Green, Red
        names = ["Texture", "Shape", "Plastic", "Paper"]
        
        for i, (obs_val, color, name) in enumerate(zip(self.obs, colors, names)):
            x = 50 + i * bar_width
            y_max = height - 50
            bar_height_val = max(10, obs_val * bar_height)
            
            # Draw bar
            self.canvas.create_rectangle(
                x, y_max - bar_height_val, x + bar_width - 20, y_max,
                fill=color, outline="black", width=2
            )
            
            # Draw value
            self.canvas.create_text(
                x + (bar_width - 20) / 2, y_max - bar_height_val - 10,
                text=f"{obs_val:.2f}", font=("Arial", 10, "bold")
            )
            
            # Draw label
            self.canvas.create_text(
                x + (bar_width - 20) / 2, y_max + 20,
                text=name, font=("Arial", 9)
            )
        
        # Update state labels
        for i, label in enumerate(self.state_labels):
            label.config(text=f"{self.obs[i]:.2f}")
            
    def run_episode_step(self):
        """Execute one step in the episode"""
        action, _ = self.model.predict(self.obs, deterministic=True)
        self.obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.step_num += 1
        self.total_reward += reward
        
        # Update action label
        action_names = ["Focus Texture", "Focus Shape", "Classify Plastic", "Classify Paper"]
        self.action_label.config(text=f"{action_names[action]}")
        
        # Update reward display
        color = "green" if reward > 0 else "red"
        self.reward_label.config(text=f"{self.total_reward:.1f}", foreground=color)
        
        # Update labels
        self.episode_label.config(text=str(self.episode_num))
        self.step_label.config(text=str(self.step_num))
        
        self.update_visualization()
        
        return terminated or truncated
        
    def run_full_episode(self):
        """Run a complete episode"""
        self.reset_episode()
        done = False
        
        while not done:
            done = self.run_episode_step()
            wait_time = int(500 / self.speed)  # Milliseconds
            self.root.update()
            self.root.after(wait_time)
        
        # Record episode
        success = "✓" if self.total_reward > 500 else "✗"
        self.episode_history.append({
            'episode': self.episode_num,
            'reward': self.total_reward,
            'steps': self.step_num,
            'success': success
        })
        
        self.history_listbox.insert(tk.END, 
            f"Ep {self.episode_num}: R={self.total_reward:.1f} | Steps={self.step_num} {success}")
        self.history_listbox.see(tk.END)
        
        # Update stats
        self.total_episodes += 1
        self.total_reward_sum += self.total_reward
        self.best_reward = max(self.best_reward, self.total_reward)
        self.worst_reward = min(self.worst_reward, self.total_reward)
        
        self.update_statistics()
        
    def start_episode(self):
        """Start a single episode in a thread"""
        if not self.running:
            self.running = True
            self.episode_num += 1
            thread = threading.Thread(target=self.run_full_episode, daemon=True)
            thread.start()
            self.running = False
            
    def auto_run_episodes(self):
        """Auto-run 10 episodes"""
        if not self.running:
            self.running = True
            
            def run_auto():
                for _ in range(10):
                    self.episode_num += 1
                    self.run_full_episode()
                self.running = False
                
            thread = threading.Thread(target=run_auto, daemon=True)
            thread.start()
            
    def update_statistics(self):
        """Update statistics display"""
        self.total_episodes_label.config(text=str(self.total_episodes))
        
        if self.total_episodes > 0:
            avg_reward = self.total_reward_sum / self.total_episodes
            self.avg_reward_label.config(text=f"{avg_reward:.1f}")
            self.best_reward_label.config(text=f"{self.best_reward:.1f}")
            self.worst_reward_label.config(text=f"{self.worst_reward:.1f}")
            
            # Success rate
            success_count = sum(1 for ep in self.episode_history if ep['success'] == "✓")
            success_rate = (success_count / self.total_episodes) * 100
            self.success_rate_label.config(text=f"{success_rate:.1f}%")
        else:
            self.avg_reward_label.config(text="0.0")
            self.best_reward_label.config(text="0.0")
            self.worst_reward_label.config(text="0.0")
            self.success_rate_label.config(text="0%")


def main():
    """Main entry point"""
    root = tk.Tk()
    gui = SmartSortGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
