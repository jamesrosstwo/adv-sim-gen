import hydra
import numpy as np
from imitation.policies.serialize import load_policy
from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import CnnPolicy

import os
import torch

from definitions import ROOT_PATH, SEED
from experiment.experiment import Experiment
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout
from imitation.data import serialize
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm


class TrainGAILExperiment(Experiment):
    def __init__(
            self,
            # expert_checkpoint: str,
            n_timesteps: int = 1_000_000,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self._expert = load_policy(
            "ppo-huggingface",
            organization = "igpaub",
            env_name = "CarRacing-v2",
            venv = self._env,
        )
        # checkpoint_path = str(ROOT_PATH / expert_checkpoint / "ppo_car_racing.zip")
        self._n_timesteps = n_timesteps
        self._rollouts_path = self._out_path / "expert_rollouts"


    def run(self):

        print("Creating PPO learner...")
        learner = PPO(
            env=self._env,
            policy="CnnPolicy",  # Suitable for CarRacing's image observations
            batch_size=128,  # Increased for better gradient estimation
            ent_coef=0.01,  # Higher entropy for better exploration
            learning_rate=0.0001,  # Lower learning rate for stable updates
            gamma=0.99,  # Increased for long-term credit assignment
            n_epochs=10,  # More epochs for better PPO training
            seed=SEED,
        )

        # Check if rollouts already exist
        if self._rollouts_path.exists():
            print("Loading expert rollouts from file...")
            rollouts = serialize.load(str(self._rollouts_path))
            print("Expert rollouts loaded.")
        else:
            print("Generating expert rollouts...")
            rollouts = rollout.rollout(
                self._expert,
                learner.get_env(),
                rollout.make_sample_until(min_timesteps=None, min_episodes=60),
                rng=np.random.default_rng(SEED),
                unwrap=False,
            )
            # Save rollouts to disk
            print("Saving expert rollouts to file...")
            serialize.save(str(self._rollouts_path), rollouts)
            print("Expert rollouts saved.")

        print("Creating reward network for GAIL...")
        reward_net = BasicRewardNet(
            observation_space=self._env.observation_space,
            action_space=self._env.action_space,
            normalize_input_layer=RunningNorm,
        )

        print("Initializing GAIL trainer...")
        gail_trainer = GAIL(
            demonstrations=rollouts,
            demo_batch_size=2048,  # Larger batch size for demonstrations
            gen_replay_buffer_capacity=2048,  # Larger replay buffer
            n_disc_updates_per_round=16,  # More updates for adversarial training
            venv=self._env,
            gen_algo=learner,
            reward_net=reward_net,
        )

        # Set the seed for the environment
        self._env.seed(SEED)

        print("Evaluating learner before training...")
        learner_rewards_before_training, _ = evaluate_policy(
            learner, self._env, 100, return_episode_rewards=True,
        )
        print("Mean reward before training:", np.mean(learner_rewards_before_training))

        print("Starting GAIL training (first phase)...")
        gail_trainer.train(50000)  # Adjusted to ~2M environment steps
        print("GAIL training (first phase) completed.")

        # Evaluate PPO learner after first phase of GAIL training
        self._env.seed(SEED)
        print("Evaluating learner after first phase of training...")
        learner_rewards_after_first_phase, _ = evaluate_policy(
            learner, self._env, 100, return_episode_rewards=True,
        )
        print("Mean reward after first phase of training:", np.mean(learner_rewards_after_first_phase))

        print("Starting GAIL training (second phase)...")
        gail_trainer.train(20000)  # Additional training steps
        print("GAIL training (second phase) completed.")

        # Evaluate PPO learner after second phase of GAIL training
        self._env.seed(SEED)
        print("Evaluating learner after second phase of training...")
        learner_rewards_after_second_phase, _ = evaluate_policy(
            learner, self._env, 100, return_episode_rewards=True,
        )
        print("Mean reward after second phase of training:", np.mean(learner_rewards_after_second_phase))

        # Save the trained learner policy
        print("Saving trained learner policy...")
        learner.save("gail_learner_policy")
        print("Learner policy saved.")

        # Save the trained reward network
        print("Saving trained reward network...")
        torch.save(reward_net.state_dict(), "gail_reward_net.pth")
        print("Reward network saved.")

        print("Experiment completed.")


@hydra.main(config_name="train_gail", config_path=str(ROOT_PATH / "config"))
def main(cfg: DictConfig):
    exp = TrainGAILExperiment(**cfg)
    exp.run()


if __name__ == "__main__":
    main()
