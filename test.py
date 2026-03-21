import gym
import numpy as np
import time
import gym
import gym_carla
import carla

import matplotlib.pyplot as plt

def main():
  # parameters for the gym_carla environment
  params = {
    'number_of_vehicles': 150,
    'number_of_walkers': 0,
    'display_size': 256,          # screen size of bird-eye render
    'max_past_step': 1,           # the number of past steps to draw
    'dt': 0.1,                    # time interval between two frames

    'discrete': False,
    'discrete_acc': [
    (0.00, 0.00),  # coast
    (0.25, 0.00),
    (0.40, 0.00),
    (0.60, 0.00),
    (0.80, 0.00),
    (1.00, 0.00),
    (0.00, 0.20),  # light brake
    (0.00, 0.50),  # strong brake (keep only one strong)
],
    'discrete_steer': [-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20],
    'continuous_accel_range': [-3.0, 3.0],
    'continuous_steer_range': [-0.3, 0.3],

    'ego_vehicle_filter': 'vehicle.lincoln*',
    'port': 2000,
    'town': 'Town03',
    'task_mode': 'random',
    'max_time_episode': 200,
    'max_waypt': 12,

    'obs_range': 32
    ,              # meters
    'lidar_bin': .125 ,           # meters
    'd_behind': 12,               # meters
    'out_lane_thres': 2.0,
    'desired_speed': 8,
    'max_ego_spawn_times': 200,
    'display_route': True,

    'pixor_size': 64,
    'pixor': False,

    # ---------------------------
    # NEW: Radar parameters
    # ---------------------------
    'use_radar': True,            # enable/disable radar
    'radar_height': 1.0,          # meters
    'radar_x': 2.0,               # meters forward of ego origin
    'radar_hfov': 60,             # deg
    'radar_vfov': 20,             # deg
    'radar_range': 32,            # meters (match obs_range recommended)
    'radar_pps': 3000,            # points per second
    'radar_vmax': 30.0,           # m/s clamp for visualization
    'render_panels': 4,           # 4-panel display: birdeye, lidar, radar, camera
    "render": True,
    "enable_pygame": True,
    "frame_stack" : 4
  }

  env = gym.make('carla-v0', params=params)
  obs = env.reset()
  print(obs['radar'].shape)
  print(env.action_space.sample())

  # Basic sanity check: radar is present
  if 'radar' not in obs:
    raise RuntimeError("Radar not found in observation. Ensure CarlaEnv returns obs['radar'].")

  while True:
    print(env.action_space.sample())

    action = [1.0, 0]

    obs, r, done, info = env.step(action)


    if done:
      obs = env.reset()

if __name__ == '__main__':
  main()
