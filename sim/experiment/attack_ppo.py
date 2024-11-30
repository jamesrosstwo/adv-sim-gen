import torch
from hydra.utils import instantiate
from imitation.policies.serialize import load_policy
from omegaconf import DictConfig
from stable_baselines3 import PPO
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm

from definitions import ROOT_PATH
from sim.experiment.experiment import BaseExperiment
from perturbation.perturbation import Perturbation


# Idea 0: Train a VAE purely on reconstruction, and freeze the encoder. Reuse this encoder for all subsequent ideas.

# Idea 1: train a VAE that directly attacks the policy. in: image, VAE predicts a masked perturbation added to image.
#   Loss is difference between GT action and the estimated action by passed model

# Idea 2: Train a VAE that estimates the next frame. Takes in action and observation, loss is pixel loss between pred and actual frame?
# Or maybe loss is the difference between encodings?



class AttackExperiment(BaseExperiment):
    def __init__(
            self,
            learner_path: str,
            perturbation: DictConfig,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._perturbation: Perturbation = instantiate(perturbation)
        self._policy = load_policy(
            "ppo-huggingface",
            organization = "igpaub",
            env_name = "CarRacing-v2",
        )

    def run(self):
        obs = self._env.reset()
        done = False
        while not done:
            perturbed = self._perturbation(obs)
            action, _ = self._learner.predict(obs)
            obs, reward, done, info = self._env.step(action)
            self._env.render()