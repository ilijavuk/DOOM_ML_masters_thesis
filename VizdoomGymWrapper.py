from vizdoom import DoomGame  
import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
import cv2
from vizdoom.vizdoom import GameVariable
import EnvironmentConfigurations as EnvConfig 

damage_reward_factor = 0.01
distance_reward = 5e-4
distance_penalty = -2.5e-3
distance_reward_threshold = 3.0
ammo_reward_factor = 0.02
ammo_penalty_factor = -0.01
health_reward_factor = 0.02
health_penalty_factor = -0.01
armor_reward_factor = 0.01

AMMO_VARIABLES = [GameVariable.AMMO0, GameVariable.AMMO1,
                  GameVariable.AMMO2, GameVariable.AMMO3,
                  GameVariable.AMMO4, GameVariable.AMMO5,
                  GameVariable.AMMO6, GameVariable.AMMO7,
                  GameVariable.AMMO8, GameVariable.AMMO9]

WEAPON_VARIABLES = [GameVariable.WEAPON0, GameVariable.WEAPON1,
                    GameVariable.WEAPON2, GameVariable.WEAPON3,
                    GameVariable.WEAPON4, GameVariable.WEAPON5, 
                    GameVariable.WEAPON6, GameVariable.WEAPON7,
                    GameVariable.WEAPON8, GameVariable.WEAPON9]

class VizDoomGym(Env):
  def __init__(self, env_config, is_reward_shaping_on=False, is_game_window_visible=False):
    super().__init__()
    self.game = DoomGame()
    self.game.load_config(env_config["scenarioConfigFilePath"])
    self.game.set_window_visible(is_game_window_visible)
    self.game.set_automap_buffer_enabled(True)
    self.game.init()

    self.is_reward_shaping_on = is_reward_shaping_on
    self.action_number = env_config["actionNumber"]
    self.action_space = Discrete(self.action_number)
    self.actions = np.identity(self.action_number, dtype=np.uint8)

    height, width, channels = self.game.get_screen_height(), self.game.get_screen_width(), self.game.get_screen_channels()
    new_height, new_width, new_channels = self.process_frame(np.zeros((height, width, channels))).shape
    self.observation_space = Box(low=0, high=255, shape=(new_height, new_width, new_channels), dtype=np.uint8)

    self.health = 100
    self.x, self.y = self.get_player_position()
    self.weapon_state = self.get_weapon_state()
    self.ammo_state = self.get_ammo_state()
    self.total_reward = self.damage_dealt = self.deaths = self.frags = self.armor = 0

  def get_player_position(self):
    return self.game.get_game_variable(GameVariable.POSITION_X), self.game.get_game_variable(
        GameVariable.POSITION_Y)

  def get_weapon_state(self):
    return np.array([self.game.get_game_variable(WEAPON_VARIABLES[i]) for i in range(len(WEAPON_VARIABLES))])
  
  def get_ammo_state(self):
    return np.array([self.game.get_game_variable(AMMO_VARIABLES[i]) for i in range(len(AMMO_VARIABLES))])

  def close(self):
    self.game.close()
  
  def step(self, action):
    action_reward = self.game.make_action(self.actions[action], 5)

    done = self.game.is_episode_finished()
    state = self.game.get_state()
  
    if not state:
      return np.zeros(self.observation_space.shape), action_reward, done, {}
    
    reward = action_reward
    if self.is_reward_shaping_on:
        reward += self.shape_rewards()

    img = self.process_frame(state.screen_buffer)

    return img, reward, done, {}
  
  def shape_rewards(self):
        return sum([
            self.calculate_damage_reward(),
            self.calculate_ammo_reward(),
            self.calculate_health_reward(),
            self.calculate_armor_reward(),
            self.calculate_distance_reward(),
        ])

  def calculate_damage_reward(self):
      damage_dealt = self.game.get_game_variable(GameVariable.DAMAGECOUNT)
      delta_damage_dealt = damage_dealt - self.damage_dealt
      self.damage_dealt = damage_dealt
      
      reward = damage_reward_factor * delta_damage_dealt
      return reward
  
  def calculate_ammo_reward(self):
      self.weapon_state = self.get_weapon_state()

      new_ammo_state = self.get_ammo_state()
      delta_ammo = np.sum((new_ammo_state - self.ammo_state) * self.weapon_state)
      self.ammo_state = new_ammo_state
      
      ammo_reward = ammo_reward_factor * max(0, delta_ammo)
      ammo_penalty = ammo_penalty_factor * min(0, delta_ammo)
      
      reward = ammo_reward - ammo_penalty
      return reward

  def calculate_health_reward(self):
      health = max(self.game.get_game_variable(GameVariable.HEALTH), 0)
      delta_health = health - self.health
      self.health = health

      health_reward = health_reward_factor * max(0, delta_health)
      health_penalty = health_penalty_factor * min(0, delta_health)

      reward = health_reward - health_penalty
      return reward

  def calculate_armor_reward(self):
      armor = self.game.get_game_variable(GameVariable.ARMOR)
      delta_armor = armor - self.armor
      self.armor = armor
      
      reward = armor_reward_factor * max(0, delta_armor)
      return reward
  
  def calculate_distance_reward(self):
      x, y = self.get_player_position()
      delta_x = self.x - x
      delta_y = self.y - y

      distance_moved = np.sqrt(delta_x ** 2 + delta_y ** 2)
      self.x = x
      self.y = y

      reward = distance_reward if distance_moved > distance_reward_threshold else distance_penalty
      return reward

  def reset(self):
    self.game.new_episode()
    state = self.game.get_state()

    self.health = 100
    self.x, self.y = self.get_player_position()
    self.armor = self.frags = self.total_reward = self.deaths = self.damage_dealt = 0

    return self.process_frame(state.screen_buffer)
    
  def render(self):
    pass

  def process_frame(self, observation):
    if observation.shape != EnvConfig.EXPECTED_IMAGE_SHAPE:
        raise ValueError(f"Unexpected observation shape. Expected {EnvConfig.EXPECTED_IMAGE_SHAPE}, but got {observation.shape}.")
    
    resized = cv2.resize(observation[40:, 4:-4], None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    return resized
