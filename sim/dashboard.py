import functools

import gradio as gr
import hydra
from omegaconf import DictConfig
from stable_baselines3 import PPO

from definitions import ROOT_PATH
from experiment.experiment import Experiment

class Dashboard(Experiment):
    def __init__(self, ppo_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ppo = PPO.load(ROOT_PATH / ppo_path)
        self._gail = None

        # Initialize the environment and reset it to get the initial observation
        obs = self._env.reset() # [1, 96, 96, 3]
        timestep = 0
        episode_reward = 0
        done = False

        with gr.Blocks() as self._interface:
            # Create Gradio components
            self.base_image = gr.Image(value=base_image, label="True Observation")
            self.timestep_output = gr.Number(value=timestep, label="Timestep")
            self.reward_output = gr.Number(value=episode_reward, label="Episode Reward")
            self.act_ppo_button = gr.Button("Take PPO Action")

            state = gr.State({
                'obs': obs,
                'timestep': timestep,
                'episode_reward': episode_reward,
                'done': done,
            })

            self.act_ppo_button.click(
                fn=functools.partial(self.next_frame, policy=self._ppo),
                inputs=state,
                outputs=[self.base_image, self.timestep_output, self.reward_output, state]
            )

    def next_frame(self, state, policy):
        obs = state['obs']
        timestep = state['timestep']
        episode_reward = state['episode_reward']
        action, _ = policy.predict(obs, deterministic=True)

        result = self._env.step([action])
        obs, reward, done, info = result
        obs = obs[0]
        reward = reward[0]
        done = done[0]

        episode_reward += reward
        timestep += 1
        result = self._env.step(action)

        # Update the state
        state['obs'] = obs
        state['timestep'] = timestep
        state['episode_reward'] = episode_reward
        state['done'] = done
        return obs_image, timestep, episode_reward, state

    def run(self):
        self._interface.launch()

@hydra.main(config_name="dashboard", config_path=str(ROOT_PATH / "config"))
def main(cfg: DictConfig):
    dashboard = Dashboard(**cfg)
    dashboard.run()

if __name__ == "__main__":
    main()
