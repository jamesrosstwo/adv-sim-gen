import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from definitions import ROOT_PATH, SEED
from experiment import Experiment
from policy.ppo import PPOPolicy

from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm


class TrainGAILExperiment(Experiment):
    def __init__(
            self,
            expert_checkpoint: str,
            n_timesteps: int = 1_000_000,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._expert: PPOPolicy = PPOPolicy(PPO.load(ROOT_PATH / expert_checkpoint / "ppo_car_racing.zip"))
        self._n_timesteps = n_timesteps

    def run(self):

        # Create PPO learner with updated hyperparameters
        learner = PPO(
            env=self._env,
            policy="CnnPolicy",  # Suitable for CarRacing's image observations
            batch_size=128,  # Increased for better gradient estimation
            ent_coef=0.01,  # Higher entropy for better exploration
            learning_rate=1e-4,  # Lower learning rate for stable updates
            gamma=0.99,  # Increased for long-term credit assignment
            n_epochs=10,  # More epochs for better PPO training
            seed=SEED,
            verbose=1
        )

        # Create reward network for GAIL
        reward_net = BasicRewardNet(
            observation_space=self._env.observation_space,
            action_space=self._env.action_space,
            normalize_input_layer=RunningNorm,
        )

        rollouts = rollout.rollout(
            self._expert.sb3_ppo,
            learner.get_env(),
            rollout.make_sample_until(min_timesteps=None, min_episodes=60),
            rng=np.random.default_rng(SEED),
        )

        # Define GAIL trainer with updated hyperparameters
        gail_trainer = GAIL(
            demonstrations=rollouts,
            demo_batch_size=2048,  # Larger batch size for demonstrations
            gen_replay_buffer_capacity=2048,  # Larger replay buffer
            n_disc_updates_per_round=16,  # More updates for adversarial training
            venv=self._env,
            gen_algo=learner,
            reward_net=reward_net,
            init_tensorboard = True,
            init_tensorboard_graph = True,
        )

        # Set the seed for the environment
        self._env.seed(SEED)

        # Evaluate PPO learner before training
        learner_rewards_before_training, _ = evaluate_policy(
            learner, self._env, 100, return_episode_rewards=True,
        )

        # Train GAIL with updated steps (increased for complexity)
        gail_trainer.train(50000)  # Adjusted to ~2M environment steps

        # Evaluate PPO learner after GAIL training
        self._env.seed(SEED)
        learner_rewards_after_training, _ = evaluate_policy(
            learner, self._env, 100, return_episode_rewards=True,
        )

        self._env.seed(SEED)
        learner_rewards_before_training, _ = evaluate_policy(
            learner, self._env, 100, return_episode_rewards=True,
        )

        gail_trainer.train(20000)  # Train for 800_000 steps to match expert.
        self._env.seed(SEED)
        learner_rewards_after_training, _ = evaluate_policy(
            learner, self._env, 100, return_episode_rewards=True,
        )
        print("mean reward after training:", np.mean(learner_rewards_after_training))
        print("mean reward before training:", np.mean(learner_rewards_before_training))

        torch.save(reward_net.state_dict(), str(self._out_path / "gail_reward_net.pth"))


@hydra.main(config_name="train_gail", config_path=str(ROOT_PATH / "config"))
def main(cfg: DictConfig):
    exp = TrainGAILExperiment(**cfg)
    exp.run()


if __name__ == "__main__":
    main()
