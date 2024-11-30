import hydra
from imitation.policies.serialize import load_policy
from omegaconf import DictConfig
from stable_baselines3 import PPO

from definitions import ROOT_PATH, RESOURCES_PATH
from sim.experiment.experiment import Experiment


class EvaluateModelExperiment(Experiment):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        # self._env is the car racing v2 env from gymnasium.
        self._expert = load_policy(
            "ppo-huggingface",
            organization = "igpaub",
            env_name = "CarRacing-v2",
            venv = self._env,
        )

        # self._expert = PPO.load("/home/james/Desktop/adv-sim-gen/out/train_ppo_baseline/2024-11-29_19-54-45/ppo_car_racing.zip")

    def run(self):
        num_episodes = 100
        total_reward = 0
        success_rate = 0
        max_steps_per_episode = 1000

        for episode in range(num_episodes):
            obs = self._reset_env()
            done = False
            episode_reward = 0
            step_count = 0

            while not done and step_count < max_steps_per_episode:
                action, _ = self._expert.predict(obs, deterministic=True)
                result = self._env.step(action)

                # Extracting values from result, which is a tuple of four elements
                obs, reward, done, info = result

                episode_reward += reward
                step_count += 1

            total_reward += episode_reward

            # Since `info` is a list containing a dictionary
            if info and isinstance(info, list) and isinstance(info[0], dict):
                info_dict = info[0]
            else:
                info_dict = {}

            if info_dict.get("is_success", False):
                success_rate += 1

            print(f"Episode {episode + 1}: Reward: {episode_reward}")

        avg_reward = total_reward / num_episodes
        success_rate = (success_rate / num_episodes) if success_rate > 0 else None

        print(f"Average Reward over {num_episodes} episodes: {avg_reward}")
        if success_rate is not None:
            print(f"Success Rate: {success_rate * 100:.2f}%")


@hydra.main(config_name="eval", config_path=str(ROOT_PATH / "config"))
def main(cfg: DictConfig):
    exp = EvaluateModelExperiment(**cfg)
    exp.run()

if __name__ == "__main__":
    main()
