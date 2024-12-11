from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Generator, Tuple, Optional

from models.vqvae import VQVAE, Decoder
from perturbation.perturbation import Perturbation
from util.frame import get_mask
from util.param import randomize_weights


class VQVAEFramePerturbation(Perturbation):
    LEARNING_RATE = 1e-5
    GRAD_ACCUM_STEPS = 32
    N_ITERS = 10

    def __init__(self, state_path: Path, perturbation_strength=0.02, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._state_path = state_path
        self._perturbation_strength = perturbation_strength
        self.reset_params()

    def reset_params(self):
        self._vqvae = VQVAE().cuda()
        vqvae_state_dict = torch.load(str(self._state_path))
        self._vqvae.load_state_dict(vqvae_state_dict, strict=True)
        self._vqvae.decoder = Decoder().cuda()
        self._vqvae.decoder.apply(randomize_weights)
        self._vqvae.requires_grad_(False)

    @property
    def vqvae(self):
        return self._vqvae

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = get_mask(x)

        z_e = self._vqvae.encoder(x)
        z_q, _ = self._vqvae.vq_layer(z_e)
        perturbation = self._vqvae.decoder(z_q)

        pixel_norms = torch.norm(perturbation, dim=1, keepdim=True)  # Shape: [1, H, W]
        max_pixel_norm = torch.max(pixel_norms)
        perturbation = perturbation * (self._perturbation_strength / max_pixel_norm)
        perturbation *= mask
        output = x + perturbation
        output = torch.clamp(output, 0, 1)  # Clamp to [0, 1]
        return output

    def fit(self, policy: nn.Module, obs: np.ndarray) -> Generator[Tuple, None, None]:
        obs_torch = self.preproc_obs(obs)
        with torch.no_grad():
            gt_action, _, _ = policy(obs_torch)
        optimizer = optim.Adam(self._vqvae.decoder.parameters(), lr=self.LEARNING_RATE)
        accumulated_loss = 0  # To track aggregated loss for logging

        for i in tqdm(range(self.N_ITERS)):
            optimizer.zero_grad()  # Reset gradients at the beginning of accumulation

            for _ in range(self.GRAD_ACCUM_STEPS):
                perturbed_obs = self(obs_torch)
                action, _, _ = policy(perturbed_obs)

                loss = -abs(action[:, 0] - gt_action[:, 0])
                loss.backward()  # Accumulate gradients
                accumulated_loss += loss.item()

            optimizer.step()  # Apply accumulated gradients
            accumulated_loss /= self.GRAD_ACCUM_STEPS  # Average loss for logging

            yield i, accumulated_loss  # Report the averaged loss
            accumulated_loss = 0  # Reset loss accumulator for the next iteration


class VQVAELatentPerturbation(Perturbation):
    LEARNING_RATE = 1e-4
    GRAD_ACCUM_STEPS = 32
    N_ITERS = 10

    def __init__(self, state_path: Path, perturbation_strength=0.02, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._state_path = state_path
        self._perturbation_variance = perturbation_strength
        self.reset_params()

    def reset_params(self):
        self._vqvae = VQVAE().cuda()
        vqvae_state_dict = torch.load(str(self._state_path))
        self._vqvae.load_state_dict(vqvae_state_dict, strict=True)

    @property
    def vqvae(self):
        return self._vqvae

    def forward(self, x: torch.Tensor, z_perturbation: Optional[nn.Parameter] = None) -> torch.Tensor:
        mask = get_mask(x)
        z_e = self._vqvae.encoder(x)

        if z_perturbation is None:
            z_perturbation = torch.randn_like(z_e).cuda() * self._perturbation_variance
        perturbed_z_e = z_e + z_perturbation
        z_q, _ = self._vqvae.vq_layer(perturbed_z_e)
        perturbed_output = self._vqvae.decoder(z_q)
        output = (x * ~mask) + (perturbed_output * mask)
        # output = torch.clamp(output, 0, 1)  # Clamp to [0, 1]
        return output

    def fit(self, policy: nn.Module, obs: np.ndarray) -> Generator[Tuple, None, None]:
        obs_torch = self.preproc_obs(obs)
        with torch.no_grad():
            gt_action, _, _ = policy(obs_torch)

        latent_perturbation = nn.Parameter(torch.randn((256, 6, 6)).cuda())
        optimizer = optim.Adam([latent_perturbation], lr=self.LEARNING_RATE)
        accumulated_loss = 0

        for i in tqdm(range(self.N_ITERS)):
            optimizer.zero_grad()

            for _ in range(self.GRAD_ACCUM_STEPS):
                perturbed_obs = self(obs_torch)
                action, _, _ = policy(perturbed_obs)
                loss = -torch.mean((action - gt_action) ** 2)
                loss.backward()
                accumulated_loss += loss.item()

            optimizer.step()
            accumulated_loss /= self.GRAD_ACCUM_STEPS
            yield i, accumulated_loss
            accumulated_loss = 0

        self._vqvae.encoder.requires_grad_(False)
