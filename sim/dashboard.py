import functools
from typing import Type

import gradio as gr
import hydra
import numpy as np
import torch
from PIL import Image
from omegaconf import DictConfig
from stable_baselines3 import PPO

from definitions import ROOT_PATH
from experiment.experiment import Experiment
from models.ppo import PPOPolicy
from perturbation.perturbation import Perturbation
from perturbation.vae import VAELatentPerturbation, VAEFramePerturbation


class Dashboard(Experiment):
    def __init__(self, ppo_path: str, vae_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ppo = PPOPolicy(PPO.load(ROOT_PATH / ppo_path))
        self._vae_path = ROOT_PATH / vae_path
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

            base_img = self._image_from_obs(obs)
            self._latent_p = VAEFramePerturbation(self._vae_path).cuda()
            latent_perturbed = self.attack_frame(self._ppo, obs, self._latent_p)
            latent_p_img = self._image_from_obs(self._latent_p.postproc_obs(latent_perturbed))
            # frame = self.attack_frame(self._ppo, obs, VAEFramePerturbation)


            self.base_image = gr.Image(value=base_img, label="True Observation", height=400, width=400)
            self.latent_p_image = gr.Image(value=latent_p_img, label="True Observation", height=400, width=400)
            self.timestep_output = gr.Number(value=timestep, label="Timestep")
            self.reward_output = gr.Number(value=episode_reward, label="Episode Reward")


            self._policy_update_buttons = [
                self._construct_policy_button(env_state, self._ppo, "PPO")
            ]

            state_change_comps = [self.base_image, self.timestep_output, self.reward_output]
            env_state.change(self._on_state_change, env_state, state_change_comps)
        demo.launch()


    def _on_state_change(self, state):
        return [
            self._image_from_obs(state["obs"]),
            state['timestep'],
            state['episode_reward'],
        ]


    def _construct_policy_button(self, state: gr.State, policy, name: str):
        act_button = gr.Button(f"Take {name} Actions")
        on_click = functools.partial(self.next_frame, policy=policy.sb3_ppo, k=20)
        act_button.click(on_click, [state], state)
        return act_button


    def _image_from_obs(self, obs):
        image = (obs[0] * 255).astype(np.uint8)
        image = image[..., ::-1]
        pil_image = Image.fromarray(image)
        pil_image_resized = pil_image.resize((400, 400), Image.Resampling.NEAREST)
        image_resized = np.array(pil_image_resized)

        return image_resized

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


    def attack_frame(self, policy, obs, perturbation: Perturbation):
        for i, loss in perturbation.fit(policy, obs):
            print(i, loss)
        return perturbation(perturbation.preproc_obs(obs))


    def run(self):
        pass # this is dumb


@hydra.main(config_name="dashboard", config_path=str(ROOT_PATH / "config"))
def main(cfg: DictConfig):
    dashboard = Dashboard(**cfg)


if __name__ == "__main__":
    main()
