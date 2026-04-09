#!/usr/bin/env python

# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from __future__ import division

import copy
import numpy as np
import pygame
import random
import time
from skimage.transform import resize

import gym
from gym import spaces
from gym.utils import seeding
import carla

from gym_carla.envs.render import BirdeyeRender
from gym_carla.envs.route_planner import RoutePlanner
from gym_carla.envs.misc import *
from collections import deque


class CarlaEnv(gym.Env):
  """An OpenAI gym wrapper for CARLA simulator."""

  def __init__(self, params):
    # parameters
    self.display_size = params['display_size']  # rendering screen size
    self.max_past_step = params['max_past_step']
    self.number_of_vehicles = params['number_of_vehicles']
    self.number_of_walkers = params['number_of_walkers']
    self.dt = params['dt']
    self.task_mode = params['task_mode']
    self.max_time_episode = params['max_time_episode']
    self.max_waypt = params['max_waypt']
    self.obs_range = params['obs_range']
    self.lidar_bin = params['lidar_bin']
    self.d_behind = params['d_behind']
    self.obs_size = int(self.obs_range/self.lidar_bin)
    self.out_lane_thres = params['out_lane_thres']
    self.desired_speed = params['desired_speed']
    self.max_ego_spawn_times = params['max_ego_spawn_times']
    self.display_route = params['display_route']
    self.enable_pygame = params.get("enable_pygame", True)  # default keeps current behavior
    self.frame_stack = params.get('frame_stack', 1)
    self.sync = False
    self.radar_lidar_frames = deque(maxlen=self.frame_stack)
    self.radar_frames = deque(maxlen=self.frame_stack)
    self.lidar_frames = deque(maxlen=self.frame_stack)
    self.birdeye_frames = deque(maxlen=self.frame_stack)
    self.camera_frames = deque(maxlen=self.frame_stack)

    if 'pixor' in params.keys():
      self.pixor = params['pixor']
      self.pixor_size = params['pixor_size']
    else:
      self.pixor = False

    # Destination
    if params['task_mode'] == 'roundabout':
      self.dests = [[4.46, -61.46, 0], [-49.53, -2.89, 0], [-6.48, 55.47, 0], [35.96, 3.33, 0]]
    else:
      self.dests = None

    # action and observation spaces
    self.discrete = params['discrete']
    self.discrete_act = [params['discrete_acc'], params['discrete_steer']] # acc, steer
    self.n_acc = len(self.discrete_act[0])
    self.n_steer = len(self.discrete_act[1])
    if self.discrete:
      self.action_space = spaces.Discrete(self.n_acc*self.n_steer)
    else:
      self.action_space = spaces.Box(np.array([params['continuous_accel_range'][0], 
      params['continuous_steer_range'][0]]), np.array([params['continuous_accel_range'][1],
      params['continuous_steer_range'][1]]), dtype=np.float32)  # acc, steer
    observation_space_dict = {
              'camera': spaces.Box(
                  low=0, high=255,
                  shape=(self.obs_size, self.obs_size, 3 * self.frame_stack),
                  dtype=np.uint8
              ),
              'lidar': spaces.Box(
                  low=0, high=255,
                  shape=(self.obs_size, self.obs_size, 3 * self.frame_stack),
                  dtype=np.uint8
              ),
              'radar': spaces.Box(
                  low=0, high=255,
                  shape=(self.obs_size, self.obs_size, 3 * self.frame_stack),
                  dtype=np.uint8
              ),
              'birdeye': spaces.Box(
                  low=0, high=255,
                  shape=(self.obs_size, self.obs_size, 3 * self.frame_stack),
                  dtype=np.uint8
              ),
              'state': spaces.Box(
                  np.array([-2, -1, -5, 0], dtype=np.float32),
                  np.array([2, 1, 30, 1], dtype=np.float32),
                  dtype=np.float32
              )
            }
    if self.pixor:
      observation_space_dict.update({
        'roadmap': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
        'vh_clas': spaces.Box(low=0, high=1, shape=(self.pixor_size, self.pixor_size, 1), dtype=np.float32),
        'vh_regr': spaces.Box(low=-5, high=5, shape=(self.pixor_size, self.pixor_size, 6), dtype=np.float32),
        'pixor_state': spaces.Box(np.array([-1000, -1000, -1, -1, -5]), np.array([1000, 1000, 1, 1, 20]), dtype=np.float32)
        })
    self.observation_space = spaces.Dict(observation_space_dict)

    # Connect to carla server and get world object
    print('connecting to Carla server...')
    client = carla.Client('localhost', params['port'])
    client.set_timeout(60.0)
    self.world = client.load_world(params['town'])
    print('Carla server connected!')

    # Set weather
    self.world.set_weather(carla.WeatherParameters.ClearNoon)

    # Get spawn points
    self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())
    self.walker_spawn_points = []
    for i in range(self.number_of_walkers):
      spawn_point = carla.Transform()
      loc = self.world.get_random_location_from_navigation()
      if (loc != None):
        spawn_point.location = loc
        self.walker_spawn_points.append(spawn_point)

    # Create the ego vehicle blueprint
    self.ego_bp = self._create_vehicle_bluepprint(params['ego_vehicle_filter'], color='49,8,8')

    # Collision sensor
    self.collision_hist = [] # The collision history
    self.collision_hist_l = 1 # collision history length
    self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')

    # Lidar sensor
    self.lidar_data = None
    self.lidar_height = 2.1
    self.lidar_trans = carla.Transform(carla.Location(x=0.0, z=self.lidar_height))
    self.lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
    self.lidar_bp.set_attribute('channels', '32')
    self.lidar_bp.set_attribute('range', '50.0')  # NOT 5000
    self.lidar_bp.set_attribute('points_per_second', '100000')
    self.lidar_bp.set_attribute('rotation_frequency', str(int(1/self.dt)))
    self.lidar_bp.set_attribute('upper_fov', '5.0')
    self.lidar_bp.set_attribute('lower_fov', '-40.0')
    self.lidar_bp.set_attribute('sensor_tick', str(self.dt))

    self.lidar_bp.set_attribute('dropoff_general_rate', '0.0')
    self.lidar_bp.set_attribute('dropoff_intensity_limit', '1.0')
    self.lidar_bp.set_attribute('dropoff_zero_intensity', '0.0')

    
    # Radar sensor (NEW)
    self.radar_data = {"front": None, "left": None, "right": None, "back": None}
    self.radar_height = 0.5  # bumper-ish; use 2.1 if you want roof mount
    self.radar_lib = self.world.get_blueprint_library()

    self.radar_bp_front = self.radar_lib.find('sensor.other.radar')
    self.radar_bp_left  = self.radar_lib.find('sensor.other.radar')
    self.radar_bp_right = self.radar_lib.find('sensor.other.radar')
    self.radar_bp_back  = self.radar_lib.find('sensor.other.radar')

    # Shared radar attributes (you can tune side radars separately if desired)
    for bp in [self.radar_bp_front, self.radar_bp_left, self.radar_bp_right, self.radar_bp_back]:
        bp.set_attribute('horizontal_fov', '120')                 # deg
        bp.set_attribute('vertical_fov', '30')                   # deg
        bp.set_attribute('range', str(self.obs_range))           # meters
        bp.set_attribute('points_per_second', '300000')
        # bp.set_attribute('sensor_tick', str(self.dt))          # optional, recommended in sync mode

    # Transforms (vehicle coord: +x forward, +y right, +z up)
    # front radar
    self.radar_trans_front = carla.Transform(
      carla.Location(x=2, y=0.0, z=self.radar_height),
      carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
    )

    # left radar (negative y) looking to left (yaw=-90)
    self.radar_trans_left = carla.Transform(
      carla.Location(x=-0.0, y=-0.9, z=self.radar_height),
      carla.Rotation(pitch=0.0, yaw=-90.0, roll=0.0)
    )

    # right radar (positive y) looking to right (yaw=+90)
    self.radar_trans_right = carla.Transform(
      carla.Location(x=0.0, y=0.9, z=self.radar_height),
      carla.Rotation(pitch=0.0, yaw=+90.0, roll=0.0)
    )

    # back radar (looking backward): yaw=180
    self.radar_trans_back = carla.Transform(
      carla.Location(x=-2.0, y=0.0, z=self.radar_height),
      carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0)
    )
    # self.radar_bp.set_attribute('sensor_tick', '0.05')    # optional

    # Camera sensor
    self.camera_img = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)
    self.camera_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
    self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
    # Modify the attributes of the blueprint to set image resolution and field of view.
    self.camera_bp.set_attribute('image_size_x', str(self.obs_size))
    self.camera_bp.set_attribute('image_size_y', str(self.obs_size))
    self.camera_bp.set_attribute('fov', '110')
    # Set the time in seconds between sensor captures
    self.camera_bp.set_attribute('sensor_tick', '0.02')

    # Set fixed simulation step for synchronous mode
    self.settings = self.world.get_settings()
    self.settings.fixed_delta_seconds = self.dt

    # Record the time of total steps and resetting steps
    self.reset_step = 0
    self.total_step = 0
    
    # Initialize the renderer
    self._init_renderer()

    # Get pixel grid points
    if self.pixor:
      x, y = np.meshgrid(np.arange(self.pixor_size), np.arange(self.pixor_size)) # make a canvas with coordinates
      x, y = x.flatten(), y.flatten()
      self.pixel_grid = np.vstack((x, y)).T

  def reset(self):
    # Clear sensor objects
    self.collision_sensor = None
    self.lidar_sensor = None
    self.radar_sensor_front = None
    self.radar_sensor_left  = None
    self.radar_sensor_right = None
    self.radar_sensor_back  = None
    self.camera_sensor = None

    self.radar_frames.clear()
    self.birdeye_frames.clear()
    self.camera_frames.clear()
    self.lidar_frames.clear()

    # Delete sensors, vehicles and walkers
    self._clear_all_actors([
      'sensor.other.collision',
      'sensor.lidar.ray_cast',
      'sensor.other.radar',
      'sensor.camera.rgb',
      'vehicle.*',
      'controller.ai.walker',
      'walker.*'
    ])

    # Disable sync mode
    self._set_synchronous_mode(False)

    # Spawn surrounding vehicles
    random.shuffle(self.vehicle_spawn_points)
    count = self.number_of_vehicles
    if count > 0:
      for spawn_point in self.vehicle_spawn_points:
        if self._try_spawn_random_vehicle_at(spawn_point, number_of_wheels=[4]):
          count -= 1
        if count <= 0:
          break
    while count > 0:
      if self._try_spawn_random_vehicle_at(random.choice(self.vehicle_spawn_points), number_of_wheels=[4]):
        count -= 1

    # Spawn pedestrians
    random.shuffle(self.walker_spawn_points)
    count = self.number_of_walkers
    if count > 0:
      for spawn_point in self.walker_spawn_points:
        if self._try_spawn_random_walker_at(spawn_point):
          count -= 1
        if count <= 0:
          break
    while count > 0:
      if self._try_spawn_random_walker_at(random.choice(self.walker_spawn_points)):
        count -= 1

    # Get actors polygon list
    self.vehicle_polygons = []
    vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
    self.vehicle_polygons.append(vehicle_poly_dict)

    self.walker_polygons = []
    walker_poly_dict = self._get_actor_polygons('walker.*')
    self.walker_polygons.append(walker_poly_dict)

    # Spawn the ego vehicle
    ego_spawn_times = 0
    while True:
      if ego_spawn_times > self.max_ego_spawn_times:
        return self.reset()

      if self.task_mode == 'random':
        transform = random.choice(self.vehicle_spawn_points)
      elif self.task_mode == 'roundabout':
        self.start = [52.1 + np.random.uniform(-5, 5), -4.2, 178.66]
        transform = set_carla_transform(self.start)
      else:
        transform = random.choice(self.vehicle_spawn_points)

      if self._try_spawn_ego_vehicle_at(transform):
        break
      else:
        ego_spawn_times += 1
        time.sleep(0.1)

    # Add collision sensor
    self.collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.ego)

    def get_collision_hist(event):
      impulse = event.normal_impulse
      intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
      self.collision_hist.append(intensity)
      if len(self.collision_hist) > self.collision_hist_l:
        self.collision_hist.pop(0)

    self.collision_hist = []
    self.collision_sensor.listen(get_collision_hist)

    # Add lidar sensor
    self.lidar_sensor = self.world.spawn_actor(self.lidar_bp, self.lidar_trans, attach_to=self.ego)

    def get_lidar_data(data):
      self.lidar_data = data

    self.lidar_sensor.listen(get_lidar_data)

    # Add radar sensors
    self.radar_sensor_front = self.world.spawn_actor(self.radar_bp_front, self.radar_trans_front, attach_to=self.ego)
    self.radar_sensor_left  = self.world.spawn_actor(self.radar_bp_left,  self.radar_trans_left,  attach_to=self.ego)
    self.radar_sensor_right = self.world.spawn_actor(self.radar_bp_right, self.radar_trans_right, attach_to=self.ego)
    self.radar_sensor_back  = self.world.spawn_actor(self.radar_bp_back,  self.radar_trans_back,  attach_to=self.ego)

    def get_radar_data_front(data):
      self.radar_data["front"] = data

    def get_radar_data_left(data):
      self.radar_data["left"] = data

    def get_radar_data_right(data):
      self.radar_data["right"] = data

    def get_radar_data_back(data):
      self.radar_data["back"] = data

    self.radar_sensor_front.listen(get_radar_data_front)
    self.radar_sensor_left.listen(get_radar_data_left)
    self.radar_sensor_right.listen(get_radar_data_right)
    self.radar_sensor_back.listen(get_radar_data_back)

    # Add camera sensor
    self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.ego)

    def get_camera_img(data):
      array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
      array = np.reshape(array, (data.height, data.width, 4))
      array = array[:, :, :3]
      array = array[:, :, ::-1]
      self.camera_img = array

    self.camera_sensor.listen(get_camera_img)

    # Update timesteps
    self.time_step = 0
    self.reset_step += 1

    # Enable sync mode
    self.settings.synchronous_mode = True
    self.world.apply_settings(self.settings)

    self.routeplanner = RoutePlanner(self.ego, self.max_waypt)
    self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

    # Set ego information for render
    self.birdeye_render.set_hero(self.ego, self.ego.id)

    return self._get_obs()

  def _init_frame_buffers(self, obs):
      """Fill frame buffers at reset using the first observation repeated."""
      self.radar_lidar_frames.clear()
      self.radar_frames.clear()
      self.lidar_frames.clear()
      self.birdeye_frames.clear()

      for _ in range(self.frame_stack):
          self.radar_lidar_frames.append(obs['radar/lidar'].copy())
          self.radar_frames.append(obs['radar'].copy())
          self.lidar_frames.append(obs['lidar'].copy())
          self.birdeye_frames.append(obs['birdeye'].copy())


  def _update_frame_buffers(self, obs):
      """Append newest frame to all frame buffers."""
      self.radar_lidar_frames.append(obs['radar/lidar'].copy())
      self.radar_frames.append(obs['radar'].copy())
      self.lidar_frames.append(obs['lidar'].copy())
      self.birdeye_frames.append(obs['birdeye'].copy())


  def _get_stacked_obs(self, state):
      """Return stacked multi-modal observation."""
      radar_lidar_stack = np.concatenate(list(self.radar_lidar_frames), axis=2)
      radar_stack = np.concatenate(list(self.radar_frames), axis=2)
      lidar_stack = np.concatenate(list(self.lidar_frames), axis=2)
      birdeye_stack = np.concatenate(list(self.birdeye_frames), axis=2)

      return {
          'radar/lidar': radar_lidar_stack.astype(np.uint8),
          'radar': radar_stack.astype(np.uint8),
          'lidar': lidar_stack.astype(np.uint8),
          'birdeye': birdeye_stack.astype(np.uint8),
          'state': state.astype(np.float32),
      }


  def step(self, action):

    # ----------------------------
    # 0) decode action -> control
    # ----------------------------
    if self.discrete:
        long_id  = action // self.n_steer
        steer_id = action %  self.n_steer

        throttle, brake = self.discrete_act[0][long_id]
        steer           = self.discrete_act[1][steer_id]
    else:
        acc   = float(action[0])
        steer = float(action[1])

        if acc >= 0:
            throttle = np.clip(acc / 2, 0.0, 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = np.clip((-acc) / 4.0, 0.0, 1.0)

    throttle = float(throttle)
    brake    = float(brake)
    steer    = float(steer)

    # never throttle + brake
    if throttle > 0.0:
        brake = 0.0

    # prevent brake-at-rest “stuck”
    v = self.ego.get_velocity()
    speed = float(np.sqrt(v.x**2 + v.y**2))
    if speed < 0.3 and brake > 0.0:
        brake = 0.0

    # -------------------------------------------------
    # 1) Adaptive K (LOW SPEED -> LARGER K)
    # -------------------------------------------------
    K_prev = int(getattr(self, "K_current", 3))

    if K_prev == 4:
        K = 3 if speed > 1.8 else 4
    elif K_prev == 3:
        if speed > 3.8:
            K = 2
        elif speed < 1.2:
            K = 4
        else:
            K = 3
    elif K_prev == 2:
        if speed > 5.8:
            K = 1
        elif speed < 3.2:
            K = 3
        else:
            K = 2
    else:  # K_prev == 1
        K = 2 if speed < 5.0 else 1

    self.K_current = K


    # -------------------------------------------------
    # 2) Action repeat loop
    # -------------------------------------------------
    reward_sum = 0.0
    done = False
    repeats_executed = 0

    act = carla.VehicleControl(
        throttle=throttle,
        steer=steer,
        brake=brake,
        hand_brake=False,
        reverse=False,
        manual_gear_shift=False
    )

    for k in range(K):
        self.ego.apply_control(act)
        self.world.tick()   # synchronous mode

        # update polygons
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)
        while len(self.vehicle_polygons) > self.max_past_step:
            self.vehicle_polygons.pop(0)

        walker_poly_dict = self._get_actor_polygons('walker.*')
        self.walker_polygons.append(walker_poly_dict)
        while len(self.walker_polygons) > self.max_past_step:
            self.walker_polygons.pop(0)

        # update route
        self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

        # accumulate per-tick reward
        r = float(self._get_reward()[0])
        reward_sum += r

        repeats_executed += 1

        done = bool(self._terminal())
        if done:
            break
    reward_sum = reward_sum / K

    # -------------------------------------------------
    # 3) Bookkeeping
    # -------------------------------------------------
    info = {
        #'waypoints': self.waypoints,
        'vehicle_front': self.vehicle_front,
        'action_repeat': K,
        'repeats_executed': repeats_executed,
        'speed_mps': speed,
        'departure': self._get_reward()[2]
    }

    self.time_step += 1
    self.total_step += 1

    obs = self._get_obs()

    # IMPORTANT: return sum (not average)
    return (obs, reward_sum, done, copy.deepcopy(info))

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def render(self, mode):
    pass

  def _create_vehicle_bluepprint(self, actor_filter, color=None, number_of_wheels=[4]):
    """Create the blueprint for a specific actor type.

    Args:
      actor_filter: a string indicating the actor type, e.g, 'vehicle.lincoln*'.

    Returns:
      bp: the blueprint object of carla.
    """
    blueprints = self.world.get_blueprint_library().filter(actor_filter)
    blueprint_library = []
    for nw in number_of_wheels:
      blueprint_library = blueprint_library + [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == nw]
    bp = random.choice(blueprint_library)
    if bp.has_attribute('color'):
      if not color:
        color = random.choice(bp.get_attribute('color').recommended_values)
      bp.set_attribute('color', color)
    return bp

  def _init_renderer(self):
    if not pygame.get_init():
      pygame.init()
    if not pygame.font.get_init():
      pygame.font.init()

    # Headless: create a hidden 1x1 display so Surface.convert() works
    if not getattr(self, "enable_pygame", True):
      if not pygame.display.get_init():
        pygame.display.init()
      pygame.display.set_mode((1, 1), flags=pygame.HIDDEN)  # <-- key line
      self.display = pygame.Surface((self.display_size * 5, self.display_size))
    else:
      self.display = pygame.display.set_mode(
        (self.display_size * 5, self.display_size),
        pygame.HWSURFACE | pygame.DOUBLEBUF
      )

    pixels_per_meter = self.display_size / self.obs_range
    pixels_ahead_vehicle = (self.obs_range / 2 - self.d_behind) * pixels_per_meter
    birdeye_params = {
      'screen_size': [self.display_size, self.display_size],
      'pixels_per_meter': pixels_per_meter,
      'pixels_ahead_vehicle': pixels_ahead_vehicle
    }
    self.birdeye_render = BirdeyeRender(self.world, birdeye_params)

  def _set_synchronous_mode(self, synchronous = True):
    """Set whether to use the synchronous mode.
    """
    self.settings.synchronous_mode = synchronous
    self.world.apply_settings(self.settings)

  def _try_spawn_random_vehicle_at(self, transform, number_of_wheels=[4]):
    """Try to spawn a surrounding vehicle at specific transform with random bluprint.

    Args:
      transform: the carla transform object.

    Returns:
      Bool indicating whether the spawn is successful.
    """
    blueprint = self._create_vehicle_bluepprint('vehicle.*', number_of_wheels=number_of_wheels)
    blueprint.set_attribute('role_name', 'autopilot')
    vehicle = self.world.try_spawn_actor(blueprint, transform)
    if vehicle is not None:
      vehicle.set_autopilot()
      return True
    return False

  def _try_spawn_random_walker_at(self, transform):
    """Try to spawn a walker at specific transform with random bluprint.

    Args:
      transform: the carla transform object.

    Returns:
      Bool indicating whether the spawn is successful.
    """
    walker_bp = random.choice(self.world.get_blueprint_library().filter('walker.*'))
    # set as not invencible
    if walker_bp.has_attribute('is_invincible'):
      walker_bp.set_attribute('is_invincible', 'false')
    walker_actor = self.world.try_spawn_actor(walker_bp, transform)

    if walker_actor is not None:
      walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
      walker_controller_actor = self.world.spawn_actor(walker_controller_bp, carla.Transform(), walker_actor)
      # start walker
      walker_controller_actor.start()
      # set walk to random point
      walker_controller_actor.go_to_location(self.world.get_random_location_from_navigation())
      # random max speed
      walker_controller_actor.set_max_speed(1 + random.random())    # max speed between 1 and 2 (default is 1.4 m/s)
      return True
    return False

  def _try_spawn_ego_vehicle_at(self, transform):
    """Try to spawn the ego vehicle at specific transform.
    Args:
      transform: the carla transform object.
    Returns:
      Bool indicating whether the spawn is successful.
    """
    vehicle = None
    # Check if ego position overlaps with surrounding vehicles
    overlap = False
    for idx, poly in self.vehicle_polygons[-1].items():
      poly_center = np.mean(poly, axis=0)
      ego_center = np.array([transform.location.x, transform.location.y])
      dis = np.linalg.norm(poly_center - ego_center)
      if dis > 8:
        continue
      else:
        overlap = True
        break

    if not overlap:
      vehicle = self.world.try_spawn_actor(self.ego_bp, transform)

    if vehicle is not None:
      self.ego=vehicle
      return True
      
    return False

  def _get_actor_polygons(self, filt):
    """Get the bounding box polygon of actors.

    Args:
      filt: the filter indicating what type of actors we'll look at.

    Returns:
      actor_poly_dict: a dictionary containing the bounding boxes of specific actors.
    """
    actor_poly_dict={}
    for actor in self.world.get_actors().filter(filt):
      # Get x, y and yaw of the actor
      trans=actor.get_transform()
      x=trans.location.x
      y=trans.location.y
      yaw=trans.rotation.yaw/180*np.pi
      # Get length and width
      bb=actor.bounding_box
      l=bb.extent.x
      w=bb.extent.y
      # Get bounding box polygon in the actor's local coordinate
      poly_local=np.array([[l,w],[l,-w],[-l,-w],[-l,w]]).transpose()
      # Get rotation matrix to transform to global coordinate
      R=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
      # Get global bounding box polygon
      poly=np.matmul(R,poly_local).transpose()+np.repeat([[x,y]],4,axis=0)
      actor_poly_dict[actor.id]=poly
    return actor_poly_dict

  @staticmethod
  def radar_to_bev_points(dets, xy_mode: str, vel_sign: str, z_gate=0.5, n_steps=5, step=0.25):
    """
    dets: iterable of CARLA radar detections (each has depth, azimuth, altitude, velocity)
    xy_mode:
        'front' -> [ y, -x, v1]
        'left'  -> [-x, -y, v1]
        'right' -> [ x,  y, v1]
        'back'  -> [-y,  x, v1]
    vel_sign:
        'neg' -> keep only vr < 0
        'pos' -> keep only vr > 0
    Returns: (N*n_steps, 3) float array
    """
    if len(dets) == 0:
      return np.empty((0, 3), dtype=np.float32)

    r = np.array([d.depth for d in dets], dtype=np.float32)[:, None]
    az = np.array([d.azimuth for d in dets], dtype=np.float32)[:, None]
    alt = np.array([d.altitude for d in dets], dtype=np.float32)[:, None]
    vr = np.array([d.velocity for d in dets], dtype=np.float32)[:, None]

    R = (np.arange(n_steps, dtype=np.float32) * step)[None, :]  # (1,n_steps)
    rr = r + R  # (N,n_steps)

    ca, sa = np.cos(alt), np.sin(alt)
    cz, sz = np.cos(az), np.sin(az)

    x = rr * ca * cz
    y = rr * ca * sz
    z = rr * sa

    if vel_sign == "neg":
      vmask = (vr < 0)
    elif vel_sign == "pos":
      vmask = (vr > 0)
    else:
      raise ValueError("vel_sign must be 'neg' or 'pos'")

    v1 = (z > z_gate) * vmask * np.abs(vr)

    if xy_mode == "front":
      pts = np.stack([y, -x, v1], axis=-1)
    elif xy_mode == "left":
      pts = np.stack([-x, -y, v1], axis=-1)
    elif xy_mode == "right":
      pts = np.stack([x, y, v1], axis=-1)
    elif xy_mode == "back":
      pts = np.stack([-y, x, v1], axis=-1)
    else:
      raise ValueError("xy_mode must be one of: front, left, right, back")

    return pts.reshape(-1, 3)

  def _get_obs(self):
    """Get the observations."""
    ## Birdeye rendering
    self.birdeye_render.vehicle_polygons = self.vehicle_polygons
    self.birdeye_render.walker_polygons = self.walker_polygons
    self.birdeye_render.waypoints = self.waypoints

    # birdeye view with roadmap and actors
    birdeye_render_types = ['roadmap', 'actors']
    if self.display_route:
      birdeye_render_types.append('waypoints')
    self.birdeye_render.render(self.display, birdeye_render_types)
    birdeye = pygame.surfarray.array3d(self.display)
    birdeye = birdeye[0:self.display_size, :, :]
    birdeye = display_to_rgb(birdeye, self.obs_size)

    # Roadmap
    if self.pixor:
      roadmap_render_types = ['roadmap']
      if self.display_route:
        roadmap_render_types.append('waypoints')
      self.birdeye_render.render(self.display, roadmap_render_types)
      roadmap = pygame.surfarray.array3d(self.display)
      roadmap = roadmap[0:self.display_size, :, :]
      roadmap = display_to_rgb(roadmap, self.obs_size)
      # Add ego vehicle
      for i in range(self.obs_size):
        for j in range(self.obs_size):
          if abs(birdeye[i, j, 0] - 255)<20 and abs(birdeye[i, j, 1] - 0)<20 and abs(birdeye[i, j, 0] - 255)<20:
            roadmap[i, j, :] = birdeye[i, j, :]

    # Display birdeye image
    birdeye_surface = rgb_to_display_surface(birdeye, self.display_size)
    self.display.blit(birdeye_surface, (0, 0))

    ## Lidar image generation
    point_cloud = []
    # Get point cloud data
    for location in self.lidar_data:
      point_cloud.append([location.point.y, -location.point.x, location.point.z])
    point_cloud = np.array(point_cloud)

    # Separate the 3D space to bins for point cloud, x and y is set according to self.lidar_bin,
    # and z is set to be two bins.
    y_bins = np.arange(-(self.obs_range - self.d_behind), self.d_behind+self.lidar_bin, self.lidar_bin)
    x_bins = np.arange(-self.obs_range/2, self.obs_range/2+self.lidar_bin, self.lidar_bin)
    #z_bins = [-self.lidar_height-1, -self.lidar_height+0.25, 1]
    z_bins = [-1.5, 0, 1]
    #z_bins = [-3.0, 1.5, 2.0]
    #print(z_bins)
    # Get lidar image according to the bins
    lidar, _ = np.histogramdd(point_cloud, bins=(x_bins, y_bins, z_bins))
    lidar[:,:,0] = np.array(lidar[:,:,0]>0, dtype=np.uint8)
    lidar[:,:,1] = np.array(lidar[:,:,1]>0, dtype=np.uint8)
    # Add the waypoints to lidar image
    if self.display_route:
      wayptimg = (birdeye[:,:,0] <= 10) * (birdeye[:,:,1] <= 10) * (birdeye[:,:,2] >= 240)
    else:
      wayptimg = birdeye[:,:,0] < 0  # Equal to a zero matrix
    wayptimg = np.expand_dims(wayptimg, axis=2)
    wayptimg = np.fliplr(np.rot90(wayptimg, 3))

    lidar1 = lidar






    # Get the final lidar image
    lidar = np.concatenate((lidar, wayptimg), axis=2)
    lidar = np.flip(lidar, axis=1)
    lidar = np.rot90(lidar, 1)
    lidar = lidar * 255


    # Display lidar image
    # lidar_surface = rgb_to_display_surface(lidar, self.display_size)
    # self.display.blit(lidar_surface, (self.display_size, 0))

    # ## Radar image generation (NEW)

    point_cloudr = []

    R = (np.arange(10, dtype=np.float32) * 0.01)[None, :]


    # =============== FRONT ===============

    dets = self.radar_data["front"]

    if len(dets):
        r  = np.array([d.depth    for d in dets], dtype=np.float32)[:, None]
        az = np.array([d.azimuth  for d in dets], dtype=np.float32)[:, None]
        al = np.array([d.altitude for d in dets], dtype=np.float32)[:, None]
        vr = np.array([d.velocity for d in dets], dtype=np.float32)[:, None]  # relative radial vel (m/s)

        # LOS unit direction in radar frame
        ca, sa = np.cos(al), np.sin(al)
        cz, sz = np.cos(az), np.sin(az)
        rhat_x = ca * cz
        rhat_y = ca * sz
        rhat_z = sa

        # Ego velocity projected into radar/ego frame
        v = self.ego.get_velocity()
        v_world = np.array([v.x, v.y, v.z], dtype=np.float32)

        T = self.ego.get_transform()
        fwd = T.get_forward_vector()
        rgt = T.get_right_vector()
        up  = T.get_up_vector()

        fwd_vec = np.array([fwd.x, fwd.y, fwd.z], dtype=np.float32)
        rgt_vec = np.array([rgt.x, rgt.y, rgt.z], dtype=np.float32)
        up_vec  = np.array([up.x,  up.y,  up.z ], dtype=np.float32)

        # components of ego velocity expressed in ego/radar axes
        v_ego_x = float(v_world @ fwd_vec)  # forward
        v_ego_y = float(v_world @ rgt_vec)  # right
        v_ego_z = float(v_world @ up_vec)   # up

        # ego LOS component (N,1)
        v_ego_los = (v_ego_x * rhat_x + v_ego_y * rhat_y + v_ego_z * rhat_z).astype(np.float32)

        # "absolute" target radial component (world-fixed reference)
        vr_abs = vr + v_ego_los

        # range smear (unchanged, but use vr_abs if you want smear to reflect corrected vel)
        rr = r + np.matmul(vr, R)

        # coordinates (unchanged)
        x = rr * ca * cz
        y = rr * ca * sz
        z = rr * sa

        # use corrected velocity for channels
        v1 = (z > 0.5) * (z < 1.5) * (vr_abs < 0) * np.abs(vr_abs) + (z > 0.5) * (z < 1.5) * (vr_abs > 0) * np.abs(vr_abs)
        va = (z > 0.5) * (z < 1.5) * (vr_abs > 0) * np.abs(vr_abs)

        pts = np.stack([y, -x, v1, va], axis=-1)
        point_cloudr.extend(pts.reshape(-1, 4))



    dets = self.radar_data["left"]

    if len(dets):
        r  = np.array([d.depth    for d in dets], dtype=np.float32)[:, None]
        az = np.array([d.azimuth  for d in dets], dtype=np.float32)[:, None]
        al = np.array([d.altitude for d in dets], dtype=np.float32)[:, None]
        vr = np.array([d.velocity for d in dets], dtype=np.float32)[:, None]  # relative radial vel (m/s)

        # LOS unit direction in LEFT RADAR frame
        ca, sa = np.cos(al), np.sin(al)
        cz, sz = np.cos(az), np.sin(az)
        rhat_x = ca * cz
        rhat_y = ca * sz
        rhat_z = sa

        # Ego velocity in world
        v = self.ego.get_velocity()
        v_world = np.array([v.x, v.y, v.z], dtype=np.float32)

        # Ego basis vectors in world
        T = self.ego.get_transform()
        fwd = T.get_forward_vector()
        rgt = T.get_right_vector()
        up  = T.get_up_vector()

        fwd_vec = np.array([fwd.x, fwd.y, fwd.z], dtype=np.float32)
        rgt_vec = np.array([rgt.x, rgt.y, rgt.z], dtype=np.float32)
        up_vec  = np.array([up.x,  up.y,  up.z ], dtype=np.float32)

        # ---- LEFT RADAR axes expressed in world ----
        # Assuming left radar yaw = +90 deg relative to ego:
        # radar_x (forward) = ego_left = -ego_right
        # radar_y (right)   = ego_forward
        # radar_z (up)      = ego_up
        radar_x = -rgt_vec
        radar_y =  fwd_vec
        radar_z =  up_vec

        # Ego velocity components in LEFT RADAR frame
        v_ego_x = float(v_world @ radar_x)
        v_ego_y = float(v_world @ radar_y)
        v_ego_z = float(v_world @ radar_z)

        # Ego LOS component (N,1)
        v_ego_los = (v_ego_x * rhat_x + v_ego_y * rhat_y + v_ego_z * rhat_z).astype(np.float32)

        # Corrected radial velocity (world-fixed reference)
        vr_abs = vr + v_ego_los

        # Range smear (keep using vr or switch to vr_abs depending on intent)
        rr = r + np.matmul(vr, R)

        # Coordinates in left radar frame
        x = rr * ca * cz
        y = rr * ca * sz
        z = rr * sa

        # Use corrected velocity for channels
        v1 = (z > 0.5) * (z < 1.5) * (vr_abs < 0) * np.abs(vr_abs) + (z > 0.5) * (z < 1.5) * (vr_abs > 0) * np.abs(vr_abs)
        va = (z > 0.5) * (z < 1.5) * (vr_abs > 0) * np.abs(vr_abs)

        # LEFT mapping (your original)
        pts = np.stack([-x, -y, v1, va], axis=-1)

        point_cloudr.extend(pts.reshape(-1, 4))

    dets = self.radar_data["right"]

    if len(dets):
        r  = np.array([d.depth    for d in dets], dtype=np.float32)[:, None]
        az = np.array([d.azimuth  for d in dets], dtype=np.float32)[:, None]
        al = np.array([d.altitude for d in dets], dtype=np.float32)[:, None]
        vr = np.array([d.velocity for d in dets], dtype=np.float32)[:, None]  # relative radial vel (m/s)

        # LOS unit direction in RIGHT RADAR frame
        ca, sa = np.cos(al), np.sin(al)
        cz, sz = np.cos(az), np.sin(az)
        rhat_x = ca * cz
        rhat_y = ca * sz
        rhat_z = sa

        # Ego velocity in world
        v = self.ego.get_velocity()
        v_world = np.array([v.x, v.y, v.z], dtype=np.float32)

        # Ego basis vectors in world
        T = self.ego.get_transform()
        fwd = T.get_forward_vector()
        rgt = T.get_right_vector()
        up  = T.get_up_vector()

        fwd_vec = np.array([fwd.x, fwd.y, fwd.z], dtype=np.float32)
        rgt_vec = np.array([rgt.x, rgt.y, rgt.z], dtype=np.float32)
        up_vec  = np.array([up.x,  up.y,  up.z ], dtype=np.float32)


        radar_x =  rgt_vec
        radar_y = -fwd_vec
        radar_z =  up_vec


        v_ego_x = float(v_world @ radar_x)
        v_ego_y = float(v_world @ radar_y)
        v_ego_z = float(v_world @ radar_z)

        # Ego LOS component (N,1)
        v_ego_los = (v_ego_x * rhat_x + v_ego_y * rhat_y + v_ego_z * rhat_z).astype(np.float32)

        # Corrected radial velocity (world-fixed reference)
        vr_abs = vr + v_ego_los

        # Range smear (keep vr, or switch to vr_abs if you want smear to reflect corrected vel)
        rr = r + np.matmul(vr, R)

        # Coordinates in right radar frame
        x = rr * ca * cz
        y = rr * ca * sz
        z = rr * sa

        # Use corrected velocity for channels
        v1 = (z > 0.5) * (z < 1.5) * (vr_abs < 0) * np.abs(vr_abs) + (z > 0.5) * (z < 1.5) * (vr_abs > 0) * np.abs(vr_abs)
        va = (z > 0.5) * (z < 1.5) * (vr_abs > 0) * np.abs(vr_abs)

        # RIGHT mapping (your original)
        pts = np.stack([x, y, v1, va], axis=-1)

        point_cloudr.extend(pts.reshape(-1, 4))


    dets = self.radar_data["back"]

    if len(dets):
        r  = np.array([d.depth    for d in dets], dtype=np.float32)[:, None]
        az = np.array([d.azimuth  for d in dets], dtype=np.float32)[:, None]
        al = np.array([d.altitude for d in dets], dtype=np.float32)[:, None]
        vr = np.array([d.velocity for d in dets], dtype=np.float32)[:, None]  # relative radial vel (m/s)

        # LOS unit direction in BACK RADAR frame
        ca, sa = np.cos(al), np.sin(al)
        cz, sz = np.cos(az), np.sin(az)
        rhat_x = ca * cz
        rhat_y = ca * sz
        rhat_z = sa

        # Ego velocity in world
        v = self.ego.get_velocity()
        v_world = np.array([v.x, v.y, v.z], dtype=np.float32)

        # Ego basis vectors in world
        T = self.ego.get_transform()
        fwd = T.get_forward_vector()
        rgt = T.get_right_vector()
        up  = T.get_up_vector()

        fwd_vec = np.array([fwd.x, fwd.y, fwd.z], dtype=np.float32)
        rgt_vec = np.array([rgt.x, rgt.y, rgt.z], dtype=np.float32)
        up_vec  = np.array([up.x,  up.y,  up.z ], dtype=np.float32)

        # ---- BACK RADAR axes expressed in world ----
        # Assuming back radar yaw = 180 deg relative to ego:
        # radar_x (forward) = ego_backward
        # radar_y (right)   = ego_left
        # radar_z (up)      = ego_up
        radar_x = -fwd_vec
        radar_y = -rgt_vec
        radar_z =  up_vec

        # Ego velocity components in BACK RADAR frame
        v_ego_x = float(v_world @ radar_x)
        v_ego_y = float(v_world @ radar_y)
        v_ego_z = float(v_world @ radar_z)

        # Ego LOS component (N,1)
        v_ego_los = (v_ego_x * rhat_x + v_ego_y * rhat_y + v_ego_z * rhat_z).astype(np.float32)

        # Corrected radial velocity (world-fixed reference)
        vr_abs = vr + v_ego_los

        # Range smear: keep vr or switch to vr_abs depending on what you want
        rr = r + np.matmul(vr, R)

        # Coordinates in back radar frame
        x = rr * ca * cz
        y = rr * ca * sz
        z = rr * sa

        # Use corrected velocity for channels
        v1 = (z > 0.5) * (z < 1.5) * (vr_abs < 0) * np.abs(vr_abs) + (z > 0.5) * (z < 1.5) * (vr_abs > 0) * np.abs(vr_abs)
        va = (z > 0.5) * (z < 1.5) * (vr_abs > 0) * np.abs(vr_abs)

        # BACK mapping (your original)
        pts = np.stack([-y, x, v1, va], axis=-1)

        point_cloudr.extend(pts.reshape(-1, 4))



    T = self.ego.get_transform()
    fwd = T.get_forward_vector()

    v = self.ego.get_velocity()
    v_vec = np.array([v.x, v.y, v.z], dtype=np.float32)
    f_vec = np.array([fwd.x, fwd.y, fwd.z], dtype=np.float32)

    ego_speed = float(np.dot(v_vec, f_vec))  # signed forward speed

    bb = self.ego.bounding_box
    c = bb.location
    e = bb.extent

    dx = 0.25
    dy = 0.25

    xs = np.arange(c.x - e.x, c.x + e.x + 1e-6, dx, dtype=np.float32)
    ys = np.arange(c.y - e.y, c.y + e.y + 1e-6, dy, dtype=np.float32)

    X, Y = np.meshgrid(xs, ys, indexing="xy")
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    N = X.shape[0]

    v1_ego = np.zeros(N, dtype=np.float32)

    if ego_speed < 0:
          v1_ego[:] = abs(ego_speed)
    else:
          v1_ego[:] = abs(ego_speed)

    #pts = np.stack([-x, -y, v1])

    ego_pts = np.stack([
        -Y,
        -X,
        v1_ego, v1_ego
    ], axis=-1)

    point_cloudr.extend(ego_pts)

    point_cloudr = np.asarray(point_cloudr)  # (N, 3): x, y, z

    # bins (same as yours)
    y_bins = np.arange(-(self.obs_range - self.d_behind),
                      self.d_behind + self.lidar_bin,
                      self.lidar_bin)

    x_bins = np.arange(-self.obs_range / 2,
                      self.obs_range / 2 + self.lidar_bin,
                      self.lidar_bin)

    # digitize x and y ONLY
    x_idx = np.digitize(point_cloudr[:, 0], x_bins) - 1
    y_idx = np.digitize(point_cloudr[:, 1], y_bins) - 1
    z_val = point_cloudr[:, 2]
    z_val_a = point_cloudr[:, 3]

    # grid size
    H = len(y_bins) - 1
    W = len(x_bins) - 1

    # valid mask
    valid = (
        (x_idx >= 0) & (x_idx < W) &
        (y_idx >= 0) & (y_idx < H)
    )

    x_idx = x_idx[valid]
    y_idx = y_idx[valid]
    z_val = z_val[valid]
    z_val_a = z_val_a[valid]

    # Initialize 2-channel radar map
    radar = np.full((H, W, 2), -np.inf, dtype=np.float32)

    # Channel 0 → z_val
    np.maximum.at(radar[:, :, 0], (y_idx, x_idx), z_val)

    # Channel 1 → z_val_a
    np.maximum.at(radar[:, :, 1], (y_idx, x_idx), z_val_a)

    # Replace -inf with 0
    radar[radar == -np.inf] = 0.0
    # Get the final lidar image

    radar = radar* 255

    #radar = np.expand_dims(radar, axis=-1)

    lidar1 = np.flip(lidar1, axis=1)

    lidar1 = np.rot90(lidar1, 1)

    lidar1= lidar1 * 255

    wayptimg = np.flip(wayptimg, axis=1)
    wayptimg = np.rot90(wayptimg, 1)
    wayptimg = wayptimg * 255

    

    #radar1 = np.concatenate((lidar1, radar), axis=2)
    #lidarn = (lidar1 + wayptimg)/2
    #lidarn = lidar1.mean(axis=2, keepdims=True)
    radar_lidar = np.concatenate((radar, lidar), axis=2)
    #print(radar.shape)
    radarf = np.concatenate((radar, wayptimg), axis=2)

    # Display lidar image

    radar_lidar_surface = rgb_to_display_surface(radar_lidar[:,:,:3], self.display_size)
    self.display.blit(radar_lidar_surface, (self.display_size * 1, 0))

    # radar_surface = rgb_to_display_surface(radarf, self.display_size)
    # self.display.blit(radar_surface, (self.display_size* 2, 0))

    # lidar_surface = rgb_to_display_surface(lidar, self.display_size)
    # self.display.blit(lidar_surface, (self.display_size* 3, 0))

    ## Display camera image
    camera = resize(self.camera_img, (self.obs_size, self.obs_size)) * 255
    camera_surface = rgb_to_display_surface(camera, self.display_size)
    self.display.blit(camera_surface, (self.display_size * 2, 0))

    # Display on pygame
    if getattr(self, "enable_pygame", True):
      pygame.display.flip()

    # State observation
    ego_trans = self.ego.get_transform()
    ego_x = ego_trans.location.x
    ego_y = ego_trans.location.y
    ego_yaw = ego_trans.rotation.yaw/180*np.pi
    lateral_dis, w = get_preview_lane_dis(self.waypoints, ego_x, ego_y)
    delta_yaw = np.arcsin(np.cross(w, 
      np.array(np.array([np.cos(ego_yaw), np.sin(ego_yaw)]))))
    v = self.ego.get_velocity()
    speed = np.sqrt(v.x**2 + v.y**2)
    state = np.array([lateral_dis, - delta_yaw, speed, self.vehicle_front])

    if self.pixor:
      ## Vehicle classification and regression maps (requires further normalization)
      vh_clas = np.zeros((self.pixor_size, self.pixor_size))
      vh_regr = np.zeros((self.pixor_size, self.pixor_size, 6))

      # Generate the PIXOR image. Note in CARLA it is using left-hand coordinate
      # Get the 6-dim geom parametrization in PIXOR, here we use pixel coordinate
      for actor in self.world.get_actors().filter('vehicle.*'):
        x, y, yaw, l, w = get_info(actor)
        x_local, y_local, yaw_local = get_local_pose((x, y, yaw), (ego_x, ego_y, ego_yaw))
        if actor.id != self.ego.id:
          if abs(y_local)<self.obs_range/2+1 and x_local<self.obs_range-self.d_behind+1 and x_local>-self.d_behind-1:
            x_pixel, y_pixel, yaw_pixel, l_pixel, w_pixel = get_pixel_info(
              local_info=(x_local, y_local, yaw_local, l, w),
              d_behind=self.d_behind, obs_range=self.obs_range, image_size=self.pixor_size)
            cos_t = np.cos(yaw_pixel)
            sin_t = np.sin(yaw_pixel)
            logw = np.log(w_pixel)
            logl = np.log(l_pixel)
            pixels = get_pixels_inside_vehicle(
              pixel_info=(x_pixel, y_pixel, yaw_pixel, l_pixel, w_pixel),
              pixel_grid=self.pixel_grid)
            for pixel in pixels:
              vh_clas[pixel[0], pixel[1]] = 1
              dx = x_pixel - pixel[0]
              dy = y_pixel - pixel[1]
              vh_regr[pixel[0], pixel[1], :] = np.array(
                [cos_t, sin_t, dx, dy, logw, logl])

      # Flip the image matrix so that the origin is at the left-bottom
      vh_clas = np.flip(vh_clas, axis=0)
      vh_regr = np.flip(vh_regr, axis=0)

      # Pixor state, [x, y, cos(yaw), sin(yaw), speed]
      pixor_state = [ego_x, ego_y, np.cos(ego_yaw), np.sin(ego_yaw), speed]

    single_obs = {
      'radar/lidar': radar_lidar.astype(np.uint8),
      'radar' : radarf.astype(np.uint8),
      'lidar' : lidar.astype(np.uint8),
      'birdeye': birdeye.astype(np.uint8),
      'state': state.astype(np.float32),
    }

    if getattr(self, "enable_pygame", True):
      pygame.display.flip()

    if len(self.radar_frames) == 0:
        self._init_frame_buffers(single_obs)
    else:
        self._update_frame_buffers(single_obs)

    stacked_obs = self._get_stacked_obs(single_obs['state'])

    if self.pixor:
        stacked_obs.update({
          'roadmap': roadmap.astype(np.uint8),
          'vh_clas': np.expand_dims(vh_clas, -1).astype(np.float32),
          'vh_regr': vh_regr.astype(np.float32),
          'pixor_state': np.array(pixor_state, dtype=np.float32),
        })

    if getattr(self, "enable_pygame", True):
        pygame.display.flip()

    return stacked_obs

    #return obs

  def _get_reward(self):
    """Calculate the step reward."""
    # reward for speed tracking
    v = self.ego.get_velocity()
    speed = np.sqrt(v.x**2 + v.y**2)
    r_speed = -abs(speed - self.desired_speed)
    
    # reward for collision
    r_collision = 0
    if len(self.collision_hist) > 0:
      r_collision = -1

    # reward for steering:
    r_steer = -self.ego.get_control().steer**2

    # reward for out of lane
    ego_x, ego_y = get_pos(self.ego)
    dis, w = get_lane_dis(self.waypoints, ego_x, ego_y)
    r_out = 0
    if abs(dis) > self.out_lane_thres:
      r_out = -1

    # longitudinal speed
    lspeed = np.array([v.x, v.y])
    lspeed_lon = np.dot(lspeed, w)

    # cost for too fast
    r_fast = 0
    if lspeed_lon > self.desired_speed:
      r_fast = -1

    # cost for lateral acceleration
    r_lat = - abs(self.ego.get_control().steer) * lspeed_lon**2

    r = 200*r_collision + 1*lspeed_lon + 10*r_fast + 1*r_out + r_steer*5 + 0.2*r_lat - 0.1

    return r, lspeed_lon, dis


  def _terminal(self):
    """Calculate whether to terminate the current episode."""
    # Get ego state
    ego_x, ego_y = get_pos(self.ego)

    # If collides
    if len(self.collision_hist)>0: 
      return True

    # If reach maximum timestep
    if self.time_step>self.max_time_episode:
      return True

    # If at destination
    if self.dests is not None: # If at destination
      for dest in self.dests:
        if np.sqrt((ego_x-dest[0])**2+(ego_y-dest[1])**2)<4:
          return True

    # If out of lane
    dis, _ = get_lane_dis(self.waypoints, ego_x, ego_y)
    if abs(dis) > self.out_lane_thres:
      return True

    return False

  def _clear_all_actors(self, actor_filters):
    """Clear specific actors."""
    for actor_filter in actor_filters:
      for actor in self.world.get_actors().filter(actor_filter):
        if actor.is_alive:
          if actor.type_id == 'controller.ai.walker':
            actor.stop()
          actor.destroy()
