# 🚗 gym-carla (CARLA 0.9.13)

An OpenAI Gym-compatible environment for the CARLA simulator, designed for multi-modal reinforcement learning in autonomous driving for urban driving scenarios. This version is configured for CARLA 0.9.13 (updated from the original 0.9.6 setup).

---

# Overview

This repository introduces a Gym-compatible CARLA benchmark for multi-modal reinforcement learning in autonomous driving, with a focus on explicit dynamic sensing through radar.

The environment provides a unified observation space including radar, joint radar/LiDAR, LiDAR, BEV, and ego-state features, where radar encodes motion and the joint radar/LiDAR representation combines dynamic and geometric information. All modalities are returned as stacked frames, enabling temporal reasoning while remaining compatible with standard RL methods.

The benchmark supports controlled comparisons between explicit dynamic sensing and temporal stacking, allowing systematic evaluation of their impact on policy performance, robustness, and generalization.

---

# Requirements

- Ubuntu 20.04 / 22.04  
- Python 3.8 (via conda)  
- CARLA 0.9.13  
- NVIDIA GPU (recommended)

---

# Installation

```bash
# create environment
conda create -n carla913 python=3.8 -y
conda activate carla913

# fix build tools (important for older deps)
pip install -U "pip<24.1"
pip install -U "setuptools<66" "wheel<0.41"

# download and extract CARLA
mkdir -p ~/carla
cd ~/carla
wget https://github.com/carla-simulator/carla/releases/download/0.9.13/CARLA_0.9.13.tar.gz
tar -xvzf CARLA_0.9.13.tar.gz

# install system dependencies
sudo apt update
sudo apt install -y     libtiff5     libpng16-16     libjpeg-dev     libglu1-mesa     libglib2.0-0     libsm6     libxext6     libxrender1     libgomp1

# set CARLA python API
export CARLA_ROOT=~/carla/CARLA_0.9.13
export PYTHONPATH=$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.13-py3.8-linux-x86_64.egg:$PYTHONPATH
export PYTHONPATH=$CARLA_ROOT/PythonAPI/carla:$PYTHONPATH
export PYTHONPATH=$CARLA_ROOT/PythonAPI/carla/agents:$PYTHONPATH

# clone and install gym-carla
git clone https://github.com/cjy1992/gym-carla.git
cd gym-carla
pip install -r requirements.txt
pip install -e .
```

Note: Do NOT install `carla` via pip.

---

# Running CARLA

```bash
cd /path/to/CARLA_0.9.13
./CarlaUE4.sh -RenderOffScreen -carla-port=2000
```

---

# Verification

```bash
python - <<'PY'
import carla
c = carla.Client('localhost', 2000)
c.set_timeout(5.0)
print("Client:", c.get_client_version())
print("Server:", c.get_server_version())
PY
```

Expected:
Client: 0.9.13  
Server: 0.9.13  

```bash
python - <<'PY'
import carla
c = carla.Client('localhost', 2000)
c.set_timeout(5.0)
w = c.get_world()
print([bp.id for bp in w.get_blueprint_library().filter("*radar*")])
PY
```

Expected:
['sensor.other.radar']

---

# Test

Verify the setup:

```bash
python test.py
```

---

# Project Structure

```bash
Radar_Carla_Gym/
├── gym_carla/
├── test.py
├── requirements.txt
└── setup.py
```

---

# Important Notes

- CARLA server and Python API must both be 0.9.13  
- Do not mix CARLA versions  
- Do not install CARLA via pip  
- Supported Python version: 3.7–3.8  

---

# License

MIT License

---

# Acknowledgement

Original repository:  
https://github.com/cjy1992/gym-carla
