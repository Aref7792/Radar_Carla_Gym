# 🚗 A Standardized Multi-Modal Reinforcement Learning Benchmark for Autonomous Driving with Explicit Dynamic Sensing

<p align="center">
  <b>CVPR URVIS Workshop Submission</b><br>
  A reproducible benchmark for multi-modal RL with explicit dynamic sensing in CARLA
</p>

---

## 🎥 Demo

<p align="center">
  <img src="assets/demo.gif" width="700">
</p>

> Example rollout using radar-enhanced observations (BEV + dynamic sensing)

---

## 📌 Overview

This repository provides the official implementation of:

> **A Standardized Multi-Modal Reinforcement Learning Benchmark for Autonomous Driving with Explicit Dynamic Sensing**

We introduce a **Gym-compatible CARLA benchmark (v0.9.13)** for systematic evaluation of multi-modal RL algorithms.

### 🔑 Key Contributions

- **Explicit dynamic sensing via radar**
- **Unified multi-modal observation interface**
- **Controlled benchmarking protocol**
- **Reproducible evaluation across RL methods**

---

## 🧠 Observation Pipeline

<p align="center">
  <img src="assets/pipeline.png" width="800">
</p>

The environment provides:

- **BEV semantic rendering** (scene structure)
- **Radar** (dynamic motion, velocity-aware)
- **LiDAR** (geometry)
- **Radar/LiDAR fusion**
- **Ego-state vector**

### Design Principle

Unlike prior work relying on **temporal stacking**, this benchmark:

- Encodes **motion explicitly (radar)**
- Reduces ambiguity in velocity estimation
- Enables **physically grounded perception**

---

## 📊 Benchmark Setup

We evaluate:

- **DQN** (value-based)
- **PPO** (on-policy actor-critic)
- **SAC** (off-policy actor-critic)

All methods:

- Use **identical observation space**
- Share **same encoder architecture**
- Are trained under **identical environment settings**
- Results averaged over **3 random seeds**

---

## 📈 Results

<p align="center">
  <img src="assets/results.png" width="700">
</p>

### Key Findings

- Dynamic sensing (**radar**) improves:
  - Policy stability
  - Robustness to traffic variation
  - Generalization across scenarios
- Reduces reliance on:
  - Frame stacking
  - Implicit motion inference

---

## 🔍 Ablation Study

| Setting | Description |
|--------|------------|
| BEV only | Static perception baseline |
| BEV + stacking | Implicit temporal modeling |
| BEV + radar | Explicit dynamic sensing |
| BEV + radar + LiDAR | Full multi-modal |

**Observation:**  
Radar-based dynamic information consistently improves performance over static-only inputs.

---

## 🧠 Model Architecture

| Component | Specification |
|----------|-------------|
| Input | BEV + Radar/LiDAR |
| Fusion | Cross-attention |
| Latent size | 64 |
| Attention heads | 8 |
| Final layer | 512 |
| Activation | ReLU / GELU |
| Normalization | LayerNorm |

Encoders:
- Channels: 16 → 32 → 64  
- Kernels: 5×5, 3×3, 3×3  
- Strides: 2, 2, 1  

---

## ⚙️ Training Hyperparameters

| Parameter | DQN | PPO | SAC |
|----------|-----|-----|-----|
| Steps | 1e5 | 5.24e5 | 1e5 |
| γ | 0.99 | 0.99 | 0.99 |
| LR | 1e-4 | multi | multi |
| Batch | 64 | 256/64 | 16 |
| Replay | 5e4 | — | 1e5 |
| Target | 100 | — | τ=0.005 |

---

## ⚙️ Installation

```bash
conda create -n carla913 python=3.8 -y
conda activate carla913

pip install -U "pip<24.1"
pip install -U "setuptools<66" "wheel<0.41"

mkdir -p ~/carla
cd ~/carla
wget https://github.com/carla-simulator/carla/releases/download/0.9.13/CARLA_0.9.13.tar.gz
tar -xvzf CARLA_0.9.13.tar.gz

export CARLA_ROOT=~/carla/CARLA_0.9.13
export PYTHONPATH=$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.13-py3.8-linux-x86_64.egg:$PYTHONPATH

git clone https://github.com/cjy1992/gym-carla.git
cd gym-carla
pip install -r requirements.txt
pip install -e .
