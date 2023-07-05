from vizdoom import DoomGame  
import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
import cv2
from vizdoom.vizdoom import GameVariable

###################
#     REWARDS     #
###################
reward_factor_frag = 1.0
reward_factor_damage = 0.01
reward_factor_distance = 5e-4
penalty_factor_distance = -2.5e-3
reward_threshold_distance = 3.0
reward_factor_ammo_increment = 0.02
reward_factor_ammo_decrement = -0.01
reward_factor_health_increment = 0.02
reward_factor_health_decrement = -0.01
reward_factor_armor_increment = 0.01

AMMO_VARIABLES = [GameVariable.AMMO0, GameVariable.AMMO1, GameVariable.AMMO2, GameVariable.AMMO3, GameVariable.AMMO4,
                  GameVariable.AMMO5, GameVariable.AMMO6, GameVariable.AMMO7, GameVariable.AMMO8, GameVariable.AMMO9]

WEAPON_VARIABLES = [GameVariable.WEAPON0, GameVariable.WEAPON1, GameVariable.WEAPON2, GameVariable.WEAPON3,
                    GameVariable.WEAPON4,
                    GameVariable.WEAPON5, GameVariable.WEAPON6, GameVariable.WEAPON7, GameVariable.WEAPON8,
                    GameVariable.WEAPON9]

class VizDoomGym(Env):
  def __init__(self, env_config, is_reward_shaping_on=False, render=False):
    super().__init__()
    self.game = DoomGame()
    self.game.load_config(env_config["scenarioConfigFilePath"])
    self.game.set_window_visible(render)
    self.game.init()

    self.is_reward_shaping_on = is_reward_shaping_on
    self.action_number = env_config["actionNumber"]
    self.action_space = Discrete(self.action_number)

    h, w, c = self.game.get_screen_height(), self.game.get_screen_width(), self.game.get_screen_channels()
    new_h, new_w, new_c = self.process_frame(np.zeros((h, w, c))).shape
    self.observation_space = Box(low=0, high=255, shape=(new_h, new_w, new_c), dtype=np.uint8)

    self.health = 100
    self.x, self.y = self.get_player_position()
    self.weapon_state = self.get_weapon_state()
    self.ammo_state = self.get_ammo_state()
    self.total_reward = self.damage_dealt = self.deaths = self.frags = self.armor = 0


  def close(self):
    self.game.close()
  
  def step(self, action):
    actions = np.identity(self.action_number, dtype=np.uint8)
    action_reward = self.game.make_action(actions[action], 5)

    done = self.game.is_episode_finished()
    state = self.game.get_state()
  
    if not state:
      return np.zeros(self.observation_space.shape), action_reward, done, {}
    
    reward = action_reward
    if self.is_reward_shaping_on:
        reward += self.shape_rewards()

    img = self.process_frame(state.screen_buffer)
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return img, reward, done, {}
  
  def shape_rewards(self):
      reward_contributions = [
          self.calculate_damage_reward(),
          self.calculate_ammo_reward(),
          self.calculate_health_reward(),
          self.calculate_armor_reward(),
          self.calculate_player_position_reward(*self.get_player_position()),
      ]

      return sum(reward_contributions)

  def calculate_damage_reward(self):
      """Computes a reward based on total damage inflicted to enemies since last update."""
      damage_dealt = self.game.get_game_variable(GameVariable.DAMAGECOUNT)
      reward = reward_factor_damage * (damage_dealt - self.damage_dealt)

      self.damage_dealt = damage_dealt

      return reward
  
  def calculate_ammo_reward(self):
      self.weapon_state = self.get_weapon_state()

      new_ammo_state = self.get_ammo_state()
      ammo_diffs = (new_ammo_state - self.ammo_state) * self.weapon_state
      ammo_reward = reward_factor_ammo_increment * max(0, np.sum(ammo_diffs))
      ammo_penalty = reward_factor_ammo_decrement * min(0, np.sum(ammo_diffs))
      reward = ammo_reward - ammo_penalty
      
      self.ammo_state = new_ammo_state

      return reward

  def calculate_health_reward(self):
      health = max(self.game.get_game_variable(GameVariable.HEALTH), 0)

      health_reward = reward_factor_health_increment * max(0, health - self.health)
      health_penalty = reward_factor_health_decrement * min(0, health - self.health)
      reward = health_reward - health_penalty

      self.health = health

      return reward

  def calculate_armor_reward(self):
      armor = self.game.get_game_variable(GameVariable.ARMOR)
      reward = reward_factor_armor_increment * max(0, armor - self.armor)
      
      self.armor = armor

      return reward
  
  def calculate_player_position_reward(self, x, y):
      dx = self.x - x
      dy = self.y - y

      distance = np.sqrt(dx ** 2 + dy ** 2)

      if distance - reward_threshold_distance > 0:
          reward = reward_factor_distance
      else:
          reward = penalty_factor_distance

      self.x = x
      self.y = y

      return reward

  def get_player_position(self):
    return self.game.get_game_variable(GameVariable.POSITION_X), self.game.get_game_variable(
        GameVariable.POSITION_Y)

  def get_ammo_state(self):
      ammoState = np.zeros(10)

      for i in range(10):
          ammoState[i] = self.game.get_game_variable(AMMO_VARIABLES[i])

      return ammoState

  def get_weapon_state(self):
      weaponState = np.zeros(10)

      for i in range(10):
          weaponState[i] = self.game.get_game_variable(WEAPON_VARIABLES[i])

      return weaponState


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
    expected_shape = (240, 320, 3)
    
    if observation.shape != expected_shape:
        raise ValueError(f"Unexpected observation shape. Expected {expected_shape}, but got {observation.shape}.")
    
    resized = cv2.resize(observation[40:, 4:-4], None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    
    return resized
    # grayscaled = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
    # resized = cv2.resize(grayscaled, (160, 100), cv2.INTER_CUBIC)
    # return np.reshape(resized, (100, 160, 1)
