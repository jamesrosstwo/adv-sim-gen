import torch
from omegaconf import DictConfig
from stable_baselines3 import PPO
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
import gymnasium as gym

from definitions import ROOT_PATH
from experiment import Experiment
from policy.ppo import PPOPolicy


class TrainBaselineExperiment(Experiment):
    def __init__(
            self,
            learner_path: str,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._learner = PPO.load("gail_learner_policy", env=self._env)
        self._reward_net = BasicRewardNet(
            observation_space=self._env.observation_space,
            action_space=self._env.action_space,
            normalize_input_layer=RunningNorm,
        )
        self._reward_net.load_state_dict(torch.load(str(ROOT_PATH / learner_path)))

    def run(self):
        obs = self._env.reset()
        done = False
        while not done:
            action, _ = self._learner.predict(obs)
            obs, reward, done, info = self._env.step(action)
            self._env.render()