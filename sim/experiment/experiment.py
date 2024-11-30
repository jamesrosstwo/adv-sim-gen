from abc import ABC, abstractmethod

import gymnasium
import numpy as np
from omegaconf import DictConfig
from stable_baselines3.common.vec_env import DummyVecEnv

from definitions import OUT_PATH
from util.string import get_date_string

def make_env(env_config: DictConfig) -> DummyVecEnv:
    return DummyVecEnv([lambda: gymnasium.make(**env_config)])


class BaseExperiment(ABC):
    def __init__(self, name: str):
        self._name = name
        self._date_string = get_date_string()
        self._out_path = OUT_PATH / self._name / self._date_string
        self._out_path.mkdir(parents=True)
        print(f"Created experiment diretory at {self._out_path}")

    @abstractmethod
    def run(self):
        raise NotImplementedError()


class Experiment(BaseExperiment, ABC):
    def __init__(self, name: str, environment: DictConfig):
        super().__init__(name)
        self._env = make_env(environment)

    def _reset_env(self):
        obs = self._env.reset()
        for i in range(50):
            obs, _, _, _ = self._env.step(np.zeros((1, 3)))
        return obs

    @abstractmethod
    def run(self):
        raise NotImplementedError()