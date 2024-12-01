from pathlib import Path

from hydra.utils import instantiate
from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import nn

class PPOPolicy(nn.Module):

    @classmethod
    def from_conf(cls, policy: DictConfig, env: DummyVecEnv):
        return cls(ppo=instantiate(policy, env=env))

    @property
    def sb3_ppo(self):
        return self._sb3_ppo

    def __init__(self, ppo: PPO):
        super().__init__()
        self._sb3_ppo = ppo

    def forward(self,x):
        return self._sb3_ppo.policy(x)

    def save(self, path: Path):
        return self._sb3_ppo.save(path)

    def __getattr__(self, name):
        return getattr(self._sb3_ppo, name)