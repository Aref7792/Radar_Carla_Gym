# 🚗 A Standardized Multi-Modal Reinforcement Learning Benchmark for Autonomous Driving with Explicit Dynamic Sensing

An OpenAI Gym-compatible CARLA environment for **multi-modal reinforcement learning (RL)** in urban autonomous driving, with a focus on **explicit dynamic sensing via radar**.

---

## 📌 Overview

This repository provides the official implementation of:

> **A Standardized Multi-Modal Reinforcement Learning Benchmark for Autonomous Driving with Explicit Dynamic Sensing**  
> *(Submitted to CVPR URVIS Workshop)*

We introduce a **reproducible, Gym-compatible benchmark** built on CARLA (v0.9.13), designed for systematic evaluation of multi-modal RL algorithms under controlled conditions.

### Key Features

- **Multi-modal observation space**
  - Radar (dynamic sensing)
  - LiDAR (geometric structure)
  - Joint Radar/LiDAR fusion
  - Bird’s-Eye View (BEV) semantic rendering
  - Ego-state features

- **Explicit dynamic modeling**
  - Radar directly encodes motion (velocity-aware perception)
  - Reduces reliance on implicit temporal stacking

- **Temporal representation**
  - Frame stacking supported across all modalities

- **Benchmarking capability**
  - Enables controlled comparison between:
    - Static perception (BEV / LiDAR)
    - Dynamic sensing (radar-enhanced observations)

This framework supports **robust, reproducible evaluation of perception–decision pipelines** in RL-based autonomous driving.

---

## ⚙️ Requirements

- Ubuntu 20.04 / 22.04  
- Python 3.8 (Conda recommended)  
- CARLA 0.9.13  
- NVIDIA GPU (recommended)

---

# 🚀 Installation


### Create environment

```bash
conda create -n carla913 python=3.8 -y
conda activate carla913
```
### Ensure compatibility with CARLA dependencies
```
pip install -U "pip<24.1"
pip install -U "setuptools<66" "wheel<0.41"
```
### Download CARLA

```
mkdir -p ~/carla
cd ~/carla
wget https://github.com/carla-simulator/carla/releases/download/0.9.13/CARLA_0.9.13.tar.gz
tar -xvzf CARLA_0.9.13.tar.gz
```

### Install system dependencies
```
sudo apt update
sudo apt install -y \
    libtiff5 libpng16-16 libjpeg-dev libglu1-mesa \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1
```

### Configure CARLA Python API

```
export CARLA_ROOT=~/carla/CARLA_0.9.13
export PYTHONPATH=$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.13-py3.8-linux-x86_64.egg:$PYTHONPATH
export PYTHONPATH=$CARLA_ROOT/PythonAPI/carla:$PYTHONPATH
export PYTHONPATH=$CARLA_ROOT/PythonAPI/carla/agents:$PYTHONPATH
```

### Install gym-carla
```
git clone https://github.com/cjy1992/gym-carla.git
cd gym-carla
pip install -r requirements.txt
pip install -e .
```

### 🖥️ Running CARLA

```
cd /path/to/CARLA_0.9.13
./CarlaUE4.sh -RenderOffScreen -carla-port=2000
```
### ✅ Verification

```
python - <<'PY'
import carla
c = carla.Client('localhost', 2000)
c.set_timeout(5.0)
print("Client:", c.get_client_version())
print("Server:", c.get_server_version())
PY
```

Expected:

```
Client: 0.9.13
Server: 0.9.13
```

### Quick Test
```
python test.py
```

### 📊 Benchmark

We evaluate three representative RL algorithms:

DQN — value-based
PPO — on-policy actor-critic
SAC — off-policy actor-critic

All methods operate on the same multi-modal observation space (BEV + radar/LiDAR) and share a consistent encoder architecture.




