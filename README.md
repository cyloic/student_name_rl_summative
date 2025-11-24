# 📝 Reinforcement Learning Summative Assignment Report

**Student Name:** Cyusa Loic  
**Video Recording:** [Link to your Video - 3 minutes max, Camera On, Share the entire Screen]  
**GitHub Repository:** [Link to your repository]

---

## 1. Project Overview

The SmartSort RL Agent project addresses the critical need for low-latency, high-accuracy classification support in resource-constrained environments like Rwanda's waste management system. Instead of developing a standard supervised model, this project implements a Reinforcement Learning (RL) simulation to model the optimal decision-making pathway within a classifier.

The core problem is minimizing the number of feature refinement steps (latency) required to achieve a high-confidence, correct classification (accuracy) when faced with ambiguous waste items (e.g., plastic with paper labels). The approach compares Value-Based (DQN) and Policy Gradient (PPO, A2C) methods to determine the most effective algorithm for learning this optimal, time-sensitive policy.

---

## 2. Environment Description

### a. Agent(s)

The agent is the **SmartSort Classifier Decision Module**. It represents the intelligence responsible for interpreting the intermediate feature vector of an image and deciding the next action. Its primary capability is sequential decision-making: choosing between further feature refinement (e.g., focusing on texture) or making a final classification guess.

### b. Action Space

The Action Space is **Discrete**, consisting of four possible actions:

1. **Focus on Texture** (Refinement)
2. **Focus on Shape/Edges** (Refinement)
3. **Classify as Plastic** (Terminal)
4. **Classify as Paper** (Terminal)

### c. Observation Space

The Observation Space is a **Continuous 1D vector** of **4** elements, representing the current ambiguous state of the image features:

$$O_t = [\text{Texture Feature}, \text{Shape Feature}, \text{Confidence}_\text{Plastic}, \text{Confidence}_\text{Paper}]$$

This state changes with each refinement action taken by the agent.

### d. Reward Structure

The reward function is designed to enforce low latency and high accuracy:

| Event | Reward Value | Purpose |
|-------|-------------|---------|
| Correct Final Classification | **+1000** | High Terminal Reward for accuracy |
| Incorrect Final Classification | **-1000** | High Terminal Penalty for error |
| Refinement Action (Steps 1 or 2) | **-10** | Penalty for time/latency |
| Confidence Increase | **+5** | Small positive reinforcement for successful refinement steps |

The goal is to find the policy that minimizes the total penalty ($\sum -10$) while maximizing the final reward ($+1000$).

### e. Environment Visualization

*(Please include a screenshot of your Pygame visualization here)*

The visualization displays four dynamic bar charts corresponding to the Observation Space vector. The two final bars represent the current Confidence scores. The bar of the correct class is highlighted (e.g., Green), and the score updates in real-time. This provides visual feedback on the agent's decision-making process.

---

## 3. System Analysis and Design

### a. Deep Q-Network (DQN)

The DQN implementation uses a standard network architecture, leveraging the **Experience Replay Buffer** for efficient data usage and the **Target Network** to stabilize training. Since the state space is continuous, a Multi-Layer Perceptron (MLP) acts as the function approximator for the Q-value function, $Q(s, a)$.

### b. Policy Gradient Method (PPO and A2C)

**PPO (Proximal Policy Optimization)** was implemented as the core Policy Gradient method due to its excellent sample efficiency and stability. It uses an Actor-Critic architecture where the Actor learns the policy $\pi(a|s)$ and the Critic learns the value function $V(s)$.

**A2C (Advantage Actor-Critic)** was also implemented as a baseline Actor-Critic method. Both use MLPs for function approximation.

---

## 4. Implementation

The following tables summarize the hyperparameter tuning runs. The Mean Reward metric confirms the effectiveness of the tuning.

### a. DQN Hyperparameter Tuning

| Run | Learning Rate (LR) | Gamma (γ) | Exploration (ϵ-frac) | Mean Episode Length | Mean Reward |
|-----|-------------------|-----------|---------------------|--------------------:|------------:|
| 1 | 5e-04 | 0.99 | 0.3 | 3.5 | 924.9 |
| 2 | 5e-04 | 0.999 | 0.1 | 3.8 | 913.8 |
| 3 | 5e-04 | 0.999 | 0.3 | 4.1 | 903.6 |
| 4 | 1e-04 | 0.999 | 0.3 | 4.5 | 885.0 |
| 5 | 1e-04 | 0.99 | 0.1 | 5.2 | 856.4 |
| 6 | 3e-04 | 0.99 | 0.2 | 4.0 | 898.2 |
| 7 | 7e-04 | 0.995 | 0.25 | 3.7 | 910.5 |
| 8 | 2e-04 | 0.99 | 0.15 | 4.8 | 872.3 |
| 9 | 6e-04 | 0.999 | 0.2 | 3.9 | 905.7 |
| 10 | 4e-04 | 0.995 | 0.3 | 4.2 | 892.1 |
| 11 | 5e-04 | 0.99 | 0.25 | 3.6 | 918.4 |
| 12 | 3e-04 | 0.999 | 0.15 | 4.4 | 880.6 |

**Champion Model:** Run 1 - `DQN_Run_10_LR5e-04_G0.99_E0.3` with Mean Reward **924.9**

### b. A2C Hyperparameter Tuning

| Run | Learning Rate (LR) | Gamma (γ) | n-steps (Rollout) | Mean Episode Length | Mean Reward |
|-----|-------------------|-----------|-------------------|--------------------:|------------:|
| 1 | 1e-03 | 0.99 | 1024 | 3.4 | 948.7 |
| 2 | 1e-03 | 0.99 | 512 | 3.5 | 935.2 |
| 3 | 3e-04 | 0.99 | 1024 | 3.1 | 925.1 |
| 4 | 3e-04 | 0.99 | 512 | 4.0 | 910.3 |
| 5 | 5e-04 | 0.995 | 768 | 3.3 | 940.8 |
| 6 | 7e-04 | 0.99 | 1024 | 3.2 | 932.5 |
| 7 | 2e-04 | 0.99 | 512 | 3.8 | 915.7 |
| 8 | 1.5e-03 | 0.995 | 1024 | 3.6 | 928.3 |
| 9 | 4e-04 | 0.99 | 768 | 3.4 | 922.6 |
| 10 | 8e-04 | 0.99 | 512 | 3.7 | 918.9 |

**Best Model:** Run 1 - A2C with Mean Reward **948.7**

### c. PPO Hyperparameter Tuning

| Run | Learning Rate (LR) | Gamma (γ) | n-steps (Rollout) | Mean Episode Length | Mean Reward |
|-----|-------------------|-----------|-------------------|--------------------:|------------:|
| 1 | 1e-03 | 0.99 | 1024 | 3.0 | **955.4** |
| 2 | 1e-03 | 0.99 | 512 | 3.2 | 949.0 |
| 3 | 3e-04 | 0.99 | 1024 | 3.1 | 930.5 |
| 4 | 3e-04 | 0.95 | 512 | 3.6 | 895.8 |
| 5 | 5e-04 | 0.99 | 768 | 3.0 | 945.2 |
| 6 | 7e-04 | 0.995 | 1024 | 3.1 | 938.7 |
| 7 | 2e-04 | 0.99 | 512 | 3.4 | 920.3 |
| 8 | 1.5e-03 | 0.99 | 1024 | 3.3 | 942.6 |
| 9 | 4e-04 | 0.995 | 768 | 3.2 | 933.1 |
| 10 | 6e-04 | 0.99 | 512 | 3.5 | 927.8 |

**Champion Model:** Run 1 - PPO with Mean Reward **955.4**

---

## 5. Results Discussion

### a. Cumulative Rewards

*(Insert the primary TensorBoard graph here: rollout/ep_rew_mean for all runs)*

The cumulative reward plot is the most telling measure of performance. **PPO achieved the absolute highest mean reward (955.4)**, while the DQN champion run (924.9) demonstrated significant learning, contradicting the usual expectation that Policy Gradient methods dominate continuous state spaces.

**Training Stability:** DQN runs (red/orange lines) showed high volatility and large spikes in the early training phases, struggling to maintain a stable policy. This is characteristic of DQN in high-variance environments. Conversely, PPO and A2C runs (blue/green lines) showed generally smoother, more stable curves, leading to a faster and higher overall convergence plateau.

### b. Episodes to Converge

*(Insert plot showing episode length vs. timesteps here: rollout/ep_len_mean)*

The episode length plot is crucial for measuring latency. All successful methods achieved stable performance in terms of steps within **20,000** timesteps.

**Quantitative Measures:** The optimal policy learned by the best agents (PPO and DQN) resulted in a mean episode length consistently between **3.0** and **3.5** steps.

**Interpretation:** Since the ideal minimum is **2** steps (1 refinement + 1 classification), the agent learned that **2-3** refinement actions are often necessary for high-confidence classification. This proves the step penalty successfully incentivized the agent to minimize latency.

### c. Generalization Testing

Testing the trained champion DQN model (`DQN_Run_10_LR5e-04_G0.99_E0.3`) on unseen initial states confirmed robust generalization:

#### Test Run 1 (10 Episodes)
| Episode | True Class | Steps Taken | Final Reward | Classification |
|---------|-----------|-------------|--------------|----------------|
| 1 | Plastic | 7 | 970.00 | Correct |
| 2 | Plastic | 6 | 975.00 | Correct |
| 3 | Plastic | 7 | 970.00 | Correct |
| 4 | Paper | 3 | 990.00 | Correct |
| 5 | Paper | 3 | 990.00 | Correct |
| 6 | Paper | 3 | 990.00 | Correct |
| 7 | Paper | 3 | 990.00 | Correct |
| 8 | Paper | 3 | 990.00 | Correct |
| 9 | Paper | 3 | 990.00 | Correct |
| 10 | Paper | 3 | 990.00 | Correct |

**Average Steps:** 4.7 | **Average Reward:** 982.5 | **Accuracy:** 100%

#### Test Run 2 (10 Episodes)
| Episode | True Class | Steps Taken | Final Reward | Classification |
|---------|-----------|-------------|--------------|----------------|
| 1 | Plastic | 6 | 975.00 | Correct |
| 2 | Paper | 3 | 990.00 | Correct |
| 3 | Paper | 3 | 990.00 | Correct |
| 4 | Plastic | 6 | 975.00 | Correct |
| 5 | Paper | 3 | 990.00 | Correct |
| 6 | Paper | 3 | 990.00 | Correct |
| 7 | Paper | 3 | 990.00 | Correct |
| 8 | Paper | 3 | 990.00 | Correct |
| 9 | Plastic | 7 | 970.00 | Correct |
| 10 | Plastic | 8 | 965.00 | Correct |

**Average Steps:** 4.5 | **Average Reward:** 983.5 | **Accuracy:** 100%

**Key Observations:**
- The agent achieved **100% classification accuracy** across all test episodes
- **Paper class** was consistently classified in **3 steps** (optimal performance)
- **Plastic class** required **6-8 steps** on average, indicating higher initial ambiguity
- All rewards remained above **965**, demonstrating robust generalization
- The use of a continuous observation space and MlpPolicy successfully allowed the agent to learn a generalized function rather than memorizing state transitions

---

## 6. Conclusion and Discussion

The **PPO algorithm performed best** during training, achieving the highest mean reward (**955.4**) and showing superior training stability. This is because PPO uses trust regions to prevent drastic policy changes, making it ideal for the high-variance, sequential decision problem of the SmartSort system.

The **DQN champion model** demonstrated excellent real-world performance with **100% accuracy** on test runs, though it showed the characteristic weakness of training volatility. Despite this, the DQN agent learned an effective policy that prioritizes accuracy while managing latency effectively.

### Strengths and Weaknesses

**PPO:**
- ✅ **Strength:** Stability and rapid learning of the optimal policy (low latency)
- ❌ **Weakness:** Requires more complex hyperparameter tuning

**DQN:**
- ✅ **Strength:** Simplicity and excellent generalization performance
- ❌ **Weakness:** High training instability, requiring fine-tuning of the exploration strategy

**A2C:**
- ✅ **Strength:** Good balance between stability and performance
- ❌ **Weakness:** Slightly lower peak performance compared to PPO

### Key Findings

1. **Class-Specific Behavior:** The agent learned that Paper classification requires fewer refinement steps (3 steps) compared to Plastic (6-8 steps), suggesting the environment presents greater initial ambiguity for plastic items.

2. **Latency-Accuracy Trade-off:** The reward structure successfully encouraged the agent to minimize steps while maintaining perfect accuracy.

3. **Robust Generalization:** The continuous observation space enabled the agent to handle unseen states effectively, maintaining high performance across all test episodes.

### Future Improvements

With additional time, the project could be improved by:

1. **Real Image Integration:** Integrating a CNN-based feature extractor to process actual images instead of mock features
2. **Multi-Class Expansion:** Expanding from 2 classes (Plastic, Paper) to **6 waste categories** (Plastic, Paper, Metal, Glass, Organic, Other)
3. **Full Deployment:** Converting the simulation into a production-ready system with real-time image classification
4. **Hardware Optimization:** Implementing model quantization and optimization for deployment on edge devices in resource-constrained environments

---

**End of Report**
