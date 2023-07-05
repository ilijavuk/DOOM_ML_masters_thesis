import typing as t

from stable_baselines3.common import vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from VizdoomGymWrapper import VizDoomGym

def create_vectorised_environment(n_envs=1, **params) -> vec_env.VecTransposeImage:
    env = VecTransposeImage(DummyVecEnv([lambda: VizDoomGym(**params)] * n_envs))
    return env
