from abc import ABC, abstractmethod

import gymnasium
from omegaconf import DictConfig
from stable_baselines3.common.vec_env import DummyVecEnv

from definitions import OUT_PATH
from util.string import get_date_string

def make_env(env_config: DictConfig) -> DummyVecEnv:
    return DummyVecEnv([lambda: gymnasium.make(**env_config)])


class Experiment(ABC):
    def __init__(self, name: str, environment: DictConfig):
        self._name = name
        self._date_string = get_date_string()
        self._out_path = OUT_PATH / self._name / self._date_string
        self._out_path.mkdir(parents=True)
        print(f"Created experiment diretory at {self._out_path}")
        self._env = make_env(environment)

    @abstractmethod
    def run(self):
        raise NotImplementedError()