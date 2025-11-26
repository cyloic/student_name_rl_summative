#  SmartSort RL Agent

> A Reinforcement Learning approach to intelligent waste classification with low-latency decision-making

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Stable-Baselines3](https://img.shields.io/badge/stable--baselines3-latest-green.svg)](https://stable-baselines3.readthedocs.io/)
[![Gymnasium](https://img.shields.io/badge/gymnasium-latest-orange.svg)](https://gymnasium.farama.org/)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Environment Details](#environment-details)
- [Training Results](#training-results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

SmartSort RL Agent is a reinforcement learning project that addresses the challenge of **low-latency, high-accuracy waste classification** in resource-constrained environments like Rwanda's waste management system. Instead of building a traditional supervised learning classifier, this project uses RL to learn an **optimal decision-making policy** that balances:

- ⚡ **Latency**: Minimizing feature refinement steps
- 🎯 **Accuracy**: Achieving high-confidence correct classifications
- 🧠 **Intelligence**: Learning when to refine features vs. when to classify

The system compares three RL algorithms: **DQN** (Value-Based), **PPO**, and **A2C** (Policy Gradient methods).

---

## ✨ Features

- 🔄 **Sequential Decision-Making**: Agent learns when to refine features or make final classification
- 📊 **Real-time Visualization**: Pygame-based visual feedback of agent's decision process
- 🏆 **Multiple RL Algorithms**: Comparison of DQN, PPO, and A2C implementations
- 📈 **Comprehensive Logging**: TensorBoard integration for training metrics
- 🎮 **Interactive Demo**: Watch trained agents classify waste in real-time
- 💾 **Model Persistence**: Save and load trained models for evaluation

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/cyloic/student_name_rl_summative.git
   cd cyusa_loic_rl_summative
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # Linux/Mac
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Required Packages

```txt
gymnasium>=0.29.0
stable-baselines3>=2.0.0
pygame>=2.5.0
numpy>=1.24.0
tensorboard>=2.13.0
torch>=2.0.0
```

---

## ⚡ Quick Start

### Run Pre-trained Agent Demo

```bash
python main.py
```

This will:
1. Load the champion DQN model
2. Open a Pygame window showing real-time classification
3. Run 10 test episodes with visual feedback
4. Display classification results and rewards

### Train Your Own Agent

```bash
# Train DQN agent
python training/dqn_training.py --timesteps 50000

# Train PPO agent
python training/pg_training.py --algorithm ppo --timesteps 50000

# Train A2C agent
python training/pg_training.py --algorithm a2c --timesteps 50000
```

### View Training Metrics

```bash
tensorboard --logdir=./logs
```

Then open `http://localhost:6006` in your browser.

---

## 📁 Project Structure

```
cyusa_loic_rl_summative/
│
├── environment/
│   ├── custom_env.py            # Custom Gymnasium environment implementation
│   └── rendering.py             # Visualization GUI components (Pygame)
│
├── training/
│   ├── dqn_training.py          # Training script for DQN using Stable-Baselines3
│   └── pg_training.py           # Training script for PPO/A2C using Stable-Baselines3
│
├── models/
│   ├── dqn/                     # Saved DQN models
│   │   └── DQN_Run_10_LR5e-04_G0.99_E0.3.zip
│   └── pg/                      # Saved policy gradient models
│       ├── PPO_Run_1_Best.zip
│       └── A2C_Run_1_Best.zip
│
├── main.py                      # Entry point for running best performing model
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

---

## 🎮 Environment Details

### Agent
The **SmartSort Classifier Decision Module** interprets intermediate feature vectors and decides between:
- Further refinement (focus on texture/shape)
- Final classification (plastic or paper)

### Action Space (Discrete - 4 actions)
1. `Focus on Texture` - Refinement action
2. `Focus on Shape/Edges` - Refinement action
3. `Classify as Plastic` - Terminal action
4. `Classify as Paper` - Terminal action

### Observation Space (Continuous - 4D vector)
```
[Texture Feature, Shape Feature, Confidence_Plastic, Confidence_Paper]
```

### Reward Structure
| Event | Reward | Purpose |
|-------|--------|---------|
| ✅ Correct Classification | +1000 | Encourage accuracy |
| ❌ Incorrect Classification | -1000 | Penalize errors |
| 🔄 Refinement Action | -10 | Discourage latency |
| 📈 Confidence Increase | +5 | Reward learning progress |

---

## 📊 Training Results

### Algorithm Comparison

| Algorithm | Mean Reward | Mean Episode Length | Training Stability |
|-----------|-------------|--------------------|--------------------|
| **PPO** | **955.4** ⭐ | 3.0 steps | High ✅ |
| **A2C** | 948.7 | 3.4 steps | Medium ⚠️ |
| **DQN** | 924.9 | 3.5 steps | Low ⚠️ |

### Champion Model Performance

The DQN champion model (`DQN_Run_10_LR5e-04_G0.99_E0.3`) achieved:
- **100% Accuracy** on test episodes
- **3 steps** for Paper classification (optimal)
- **6-8 steps** for Plastic classification
- **Average reward: 983.0** across 20 test episodes

### Key Findings
- 📄 **Paper items** are classified faster (3 steps) due to clearer features
- 🥤 **Plastic items** require more refinement (6-8 steps) due to initial ambiguity
- 🎯 PPO showed best training stability with highest peak performance
- 🔄 DQN demonstrated excellent generalization despite training volatility

---

## Performance Analysis

### Convergence Speed
- PPO converged to 95%+ accuracy in ~20k timesteps
- A2C required ~30k timesteps
- DQN showed high variance, stabilizing after 35k timesteps

### Exploration-Exploitation Analysis
- DQN: ε-greedy (ε=0.1-0.3) provided consistent exploration
- PPO: Entropy regularization (0.0001) balanced exploration naturally
- A2C: Advantage estimation led to adaptive exploration

### Algorithm Insights
1. PPO's stability advantage due to clipped objective
2. DQN's value function approach vulnerable to overestimation
3. A2C's advantage estimation provided good gradients


## 💻 Usage

### Basic Usage

```python
from environment.custom_env import SmartSortEnv
from stable_baselines3 import DQN

# Create environment
env = SmartSortEnv()

# Load trained model
model = DQN.load("models/dqn/DQN_Run_10_LR5e-04_G0.99_E0.3")

# Run inference
obs, info = env.reset()
for _ in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        print(f"Episode finished with reward: {reward}")
        obs, info = env.reset()
```

### Advanced Training

```python
from environment.custom_env import SmartSortEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

# Create environment
env = SmartSortEnv()

# Configure model
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=1e-3,
    gamma=0.99,
    n_steps=1024,
    verbose=1,
    tensorboard_log="./logs/PPO"
)

# Setup evaluation callback
eval_callback = EvalCallback(
    env,
    best_model_save_path="./models/pg/",
    log_path="./logs/",
    eval_freq=5000
)

# Train
model.learn(total_timesteps=50000, callback=eval_callback)
```

---

## 🎓 Academic Context

**Course**: Reinforcement Learning Summative Assignment  
**Student**: Cyusa Loic  
**Project Focus**: Comparing Value-Based (DQN) vs Policy Gradient (PPO, A2C) methods for sequential decision-making in classification tasks

### Research Questions
1. Which RL algorithm learns the most efficient classification policy?
2. Can RL agents balance latency and accuracy in ambiguous classification scenarios?
3. How do different algorithms handle the exploration-exploitation trade-off?

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---


## 🙏 Acknowledgments

- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) for RL implementations
- [Gymnasium](https://gymnasium.farama.org/) for environment framework
- [Pygame](https://www.pygame.org/) for visualization
- Rwanda's waste management challenges as project inspiration

---

## 📧 Contact

**Cyusa Loic**  
- GitHub: [cyloic](https://github.com/cyloic)
- Email: l.cyusa@alustudent.com
- Project Link: [https://github.com/cyloic/cyusa_loic_rl_summative](https://github.com/yourusername/cyusa_loic_rl_summative)
- Demo Video : https://www.youtube.com/watch?v=BDUgbt6hHSE
---

## 🎥 Demo

[Link to your 3-minute video demonstration]

---




