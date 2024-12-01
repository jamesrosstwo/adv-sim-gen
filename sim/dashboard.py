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
from models.vae import ConvVAE
from models.vqvae import VQVAE
from perturbation.perturbation import Perturbation
from perturbation.vae import VAEFramePerturbation, VAELatentPerturbation
from perturbation.vqvae import VQVAEFramePerturbation, VQVAELatentPerturbation


class Dashboard(Experiment):
    def __init__(self, ppo_path: str, vae_path: str, vae_z_size: int, vqvae_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ppo = PPOPolicy(PPO.load(ROOT_PATH / ppo_path))
        self._vae_path = ROOT_PATH / vae_path
        self._vqvae_path = ROOT_PATH / vqvae_path

        self._vae_z_size = vae_z_size
        self._vae = ConvVAE(z_size=self._vae_z_size).cuda()
        vae_state_dict = torch.load(str(self._vae_path))
        self._vae.load_state_dict(vae_state_dict, strict=True)
        self._vqvae = VQVAE().cuda()
        vqvae_state_dict = torch.load(str(self._vqvae_path))
        self._vqvae.load_state_dict(vqvae_state_dict, strict=True)

        self._reconstruction_methods = {
            "VQ-VAE": self._vqvae_recons,
            "VAE": self._vae_recons,
        }

        self._attack_methods = {
            "VQ-VAE Frame Attack": VQVAEFramePerturbation,
            "VQ-VAE Latent Space Attack": VQVAELatentPerturbation,
            "VAE Frame Attack": VAEFramePerturbation,
            "VAE Latent Attack": VAELatentPerturbation
        }

        self._attack_state_paths = {
            "VQ-VAE Frame Attack": self._vqvae_path,
            "VQ-VAE Latent Space Attack": self._vqvae_path,
            "VAE Frame Attack": self._vae_path,
            "VAE Latent Attack": self._vae_path
        }

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
                'timestep_size': 20
            })
            with gr.Tab("Tab A"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=300):
                        self.timestep_output = gr.Number(value=timestep, label="Timestep")
                        self.reward_output = gr.Number(value=episode_reward, label="Episode Reward")
                        self.timestep_k_slider = gr.Slider(1, 100, step=1, value=20, label="Timestep Size")
                        self.timestep_k_slider.release(self._on_slider_change,
                                                       inputs=[self.timestep_k_slider, env_state],
                                                       outputs=env_state,
                                                       api_name="predict")

                        self.policy_selection = gr.Dropdown(["PPO Pseudo-Expert", "GAIL"], value=0, label="Policy")
                        self._construct_policy_button(env_state, self._ppo, "PPO")

                    with gr.Column(scale=2, min_width=300):
                        with gr.Row():
                            self.base_image = self._construct_gr_image(obs, "True Observation")

                with gr.Tab("VAE"):
                    self.vae_recons_image = self._construct_gr_image(self._vae_recons(obs), "VAE Reconstruction")
                with gr.Tab("VQ-VAE"):
                    self.vqvae_recons_image = self._construct_gr_image(self._vqvae_recons(obs), "VQ-VAE Reconstruction")
                with gr.Tab("Perturbations"):
                    with gr.Row():
                        with gr.Column(scale=2, min_width=300):
                            recons_selection = gr.Dropdown(["VQ-VAE", "VAE"], value=0,
                                                                label="Reconstruction Method")

                        with gr.Column(scale=1, min_width=300):
                            @gr.render(inputs=[recons_selection, env_state])
                            def discover_attack(recons_method, state):
                                prog = gr.Progress()
                                o = state["obs"]
                                recons_method = self._reconstruction_methods[recons_method]
                                # TODO: Complete this
                                self._construct_gr_image(p.postproc_obs(perturbed), label="Discovered Attack")

            with gr.Tab("Adversarial Learning"):
                with gr.Row():
                    keys = list(self._attack_methods.keys())
                    with gr.Column():
                        attack_selection = gr.Dropdown(keys, value=keys[0], label="Attack Type")
                        attack_strength = gr.Slider(0, 1)

                    @gr.render(inputs=[attack_selection, env_state])
                    def discover_attack(attack_type, state):
                        prog = gr.Progress()
                        o = state["obs"]
                        attack_method = self._attack_methods[attack_type]
                        state_path = self._attack_state_paths[attack_type]
                        p: Perturbation = attack_method(state_path).cuda()
                        for i, loss in prog.tqdm(p.fit(self._ppo, o)):
                            print(i, loss)
                        perturbed = p(p.preproc_obs(o))
                        self._construct_gr_image(p.postproc_obs(perturbed), label="Discovered Attack")
                    # self.delta_img = self._construct_gr_image(delta_img, "|Adversarial - True|")

                # delta_img = np.abs(obs_p - obs)  # Take the absolute difference for visualization
                # print(delta_img.min(), delta_img.max())
                # delta_img = delta_img / delta_img.max()  # Normalize delta to [0, 1] for display

            with gr.Tab("Augmented Policy Learning"):
                gr.Label("Coming Soon!")

            state_change_comps = [
                self.base_image,
                self.vae_recons_image,
                self.vqvae_recons_image,
                self.timestep_output,
                self.reward_output
            ]
            env_state.change(self._on_state_change, env_state, state_change_comps)
        demo.launch()

    def _on_slider_change(self, timestep_slider, state):
        print(timestep_slider)
        state["timestep_size"] = timestep_slider
        return state

    # perturbations
    def _vae_recons(self, o):
        vae_recons, _mu, _logvar = self._vae(Perturbation.preproc_obs(o))
        return Perturbation.postproc_obs(vae_recons)

    def _vqvae_recons(self, o):
        vqvae_recons, _ = self._vqvae(Perturbation.preproc_obs(o))
        return Perturbation.postproc_obs(vqvae_recons)

    def _construct_gr_image(self, obs, label: str, height=400, width=400):
        # TODO: Overlay the action here.
        im = self._image_from_obs(obs, height=height, width=width)
        return gr.Image(value=im, label=label, height=height, width=width)

    def _on_state_change(self, state):
        obs = state["obs"]
        return [
            self._image_from_obs(obs),
            self._image_from_obs(self._vae_recons(obs)),
            self._image_from_obs(self._vqvae_recons(obs)),
            state['timestep'],
            state['episode_reward'],
        ]

    def _construct_policy_button(self, state: gr.State, policy, name: str):
        act_button = gr.Button(f"Take Actions", size="lg")
        on_click = functools.partial(self.next_frame, policy=policy.sb3_ppo)
        act_button.click(on_click, [state], state)
        return act_button

    def _image_from_obs(self, obs, height: int = 400, width: int = 400):
        image = (obs[0]).astype(np.uint8)
        image = image[..., ::-1]
        pil_image = Image.fromarray(image)
        pil_image_resized = pil_image.resize((width, height), Image.Resampling.NEAREST)
        image_resized = np.array(pil_image_resized)

        return image_resized

    def next_frame(self, state, policy):
        done = False
        obs = state['obs']
        timestep = state['timestep']
        episode_reward = state['episode_reward']

        for i in range(state["timestep_size"]):
            action, _ = policy.predict(obs, deterministic=False)
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
        pass  # this is dumb


@hydra.main(config_name="dashboard", config_path=str(ROOT_PATH / "config"))
def main(cfg: DictConfig):
    dashboard = Dashboard(**cfg)


if __name__ == "__main__":
    main()
