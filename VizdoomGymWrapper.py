from vizdoom import DoomGame  
import random
import time
import numpy as np
import gym
from gym import Env
from gym.spaces import Discrete, Box
import cv2
from matplotlib import pyplot as plt
import os

class VizDoomGym(Env):
  def __init__(self, envConfig, render=False):
    super().__init__()
    self.game = DoomGame()
    self.game.load_config(envConfig["scenarioConfigFilePath"])
    self.game.set_window_visible(render)
    self.game.init()

    self.action_number = envConfig["actionNumber"]
    self.action_space = Discrete(self.action_number)
    self.observation_space = Box(0, 255, [100, 160, 1], np.uint8)

  def close(self):
    self.game.close()
  
  def step(self, action):
    actions = np.identity(self.action_number, dtype=np.uint8)
    actionReward = self.game.make_action(actions[action], 5)


    done = self.game.is_episode_finished()
    state = self.game.get_state()
  
    if not state:
      return np.zeros(self.observation_space.shape), actionReward, done, {"damage_taken": 0, "hitcount": 0, "ammo": 0}
    
    health, damage_taken, hitcount, fragcount, armor, _, ammo = state.game_variables
    
    deltasObject = {
      'damage_taken': -damage_taken + self.rewardsObject["damage_taken"],
      'hitcount': hitcount - self.rewardsObject["hitcount"],
      'fragcount': fragcount - self.rewardsObject["fragcount"],
      'armor': armor - self.rewardsObject["armor"],
      'ammo': ammo - self.rewardsObject["ammo"]
    }
    
    self.rewardsObject["damage_taken"] = damage_taken
    self.rewardsObject["hitcount"] = hitcount
    self.rewardsObject["fragcount"] = fragcount
    self.rewardsObject["armor"] = armor
    self.rewardsObject["ammo"] = ammo

    reward = actionReward + deltasObject["damage_taken"]*10 + deltasObject["hitcount"]*30 + deltasObject["fragcount"]*100 + deltasObject["ammo"]*5 #+ deltasObject["armor"]*30  

    img = self.grayscale(state.screen_buffer)
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return img, reward, done, deltasObject
  
  def reset(self):
    self.game.new_episode()
    state = self.game.get_state()
    self.rewardsObject = {
      'damage_taken': 0,
      'hitcount': 0,
      'ammo': 52,
      'armor': 0,
      'fragcount': 0
    }
    return self.grayscale(state.screen_buffer)
    
  
  def render():
    pass
  
  def grayscale(self, observation):
    grayscaled = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(grayscaled, (160, 100), cv2.INTER_CUBIC)
    return np.reshape(resized, (100, 160, 1))
  