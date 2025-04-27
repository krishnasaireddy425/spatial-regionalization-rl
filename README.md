# Spatial Regionalization using Reinforcement Learning (DQN)

This project explores **spatial regionalization** — the division of space into contiguous, similar regions — using **Deep Q-Networks (DQN)**. It compares a Reinforcement Learning (RL)-based approach against the Region2Vec baseline.

---

## 📂 Project Structure
- `notebook/` — Jupyter Notebook implementing the DQN model.
- `data/` — Dataset used (feature matrix from Region2Vec).
- `report/` — Detailed project report (PDF).

---

##  Project Overview

Spatial regionalization is essential for fields like urban planning and GIS (Geographic Information Systems).  
The goal: Partition a space into a fixed number of **contiguous, balanced** regions while optimizing certain **attribute similarities**.

Traditional methods struggle with dynamic or large datasets.  
This project uses **Deep Reinforcement Learning (DQN)** to dynamically learn partitioning strategies, aiming for better adaptability and scalability.

---

## ⚙️ Dataset
- Feature matrix derived from Region2Vec study.
- File: `data/feature_matrix_f1.csv`
- Scaled using `MinMaxScaler` to range [0, 1].

---

## 🏗️ Approach

- Build a DQN model with two hidden layers (64 neurons each, ReLU activation).
- Implement an **epsilon-greedy strategy** for balancing exploration and exploitation.
- Define a simple simulation environment to train the agent.
- Evaluate model using cluster quality metrics.

---

## 🎯 Epsilon in Reinforcement Learning

In Reinforcement Learning, **epsilon (ε)** controls the trade-off between:
- **Exploration** (trying new actions to discover better strategies).
- **Exploitation** (using known actions to maximize reward).

**In this project:**
```python
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
