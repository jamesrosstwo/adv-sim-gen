import functools

import gradio as gr
import hydra
import numpy as np
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
        obs = self._reset_env()

        timestep = 0
        episode_reward = 0
        done = False


        with gr.Blocks() as demo:
            env_state = gr.State({
                'obs': obs,
                'timestep': timestep,
                'episode_reward': episode_reward,
                'done': done,
            })

            self.base_image = gr.Image(value=self._image_from_obs(obs), label="True Observation")
            self.timestep_output = gr.Number(value=timestep, label="Timestep")
            self.reward_output = gr.Number(value=episode_reward, label="Episode Reward")


            self._policy_update_buttons = [
                self._construct_policy_button(env_state, self._ppo, "PPO")
            ]

            state_change_comps = [self.base_image, self.timestep_output, self.reward_output]
            env_state.change(self._on_state_change, env_state, state_change_comps)
        demo.launch()


    def _on_state_change(self, state):
        new_image = self._image_from_obs(state["obs"])
        tmstp = state['timestep']
        reward = state['episode_reward']

        return [
            new_image,
            tmstp,
            reward,
        ]


    def _construct_policy_button(self, state: gr.State, policy, name: str):
        act_button = gr.Button(f"Take {name} Actions")
        on_click = functools.partial(self.next_frame, policy=policy, k=20)
        act_button.click(on_click, [state], state)
        return act_button


    def _image_from_obs(self, obs):
        image = (obs[0] * 255).astype(np.uint8)
        # Swap channels from RGB to BGR or vice versa
        image = image[..., ::-1]
        return image

    def next_frame(self, state, policy, k):
        done = False
        obs = state['obs']
        timestep = state['timestep']
        episode_reward = state['episode_reward']

        for i in range(k):
            action, _ = policy.predict(obs, deterministic=True)
            print("taking action {}".format(action))
            result = self._env.step(action)
            obs, reward, done, info = result
            reward = reward[0]
            done = done[0]
            episode_reward += reward
            timestep += 1

        # Update the state
        state['obs'] = obs
        state['timestep'] = timestep
        state['episode_reward'] = episode_reward
        state['done'] = done
        return state

    def run(self):
        self._interface.launch()


@hydra.main(config_name="dashboard", config_path=str(ROOT_PATH / "config"))
def main(cfg: DictConfig):
    dashboard = Dashboard(**cfg)
    dashboard.run()


if __name__ == "__main__":
    main()
