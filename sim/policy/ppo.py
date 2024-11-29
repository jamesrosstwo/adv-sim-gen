from pathlib import Path

from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import nn

class PPOPolicy(nn.Module):
    def __init__(self, policy: DictConfig, env: DummyVecEnv):
        super().__init__()
        self._sb3_ppo = PPO(**policy, env=env)
        self.extractor = self._sb3_ppo.policy.mlp_extractor
        self.policy_net = self._sb3_ppo.policy.mlp_extractor.policy_net
        self.action_net = self._sb3_ppo.policy.action_net

    def forward(self,x):
        x = self.policy_net(x)
        x = self.action_net(x)
        return x

    def save(self, path: Path):
        return self._sb3_ppo.save(path)

    def __getattr__(self, name):
        return getattr(self._sb3_ppo, name)