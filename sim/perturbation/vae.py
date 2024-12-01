from numbers import Number
from pathlib import Path
from typing import Generator, Tuple

import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

from models.vae import ConvVAE
from perturbation.perturbation import Perturbation
import torch.nn.functional as F


class VAEFramePerturbation(Perturbation):
    LEARNING_RATE = 1e-5
    GRAD_ACCUM_STEPS = 64
    N_ITERS = 10
    def __init__(self, state_path: Path, z_size: int = 32, perturbation_strength=0.02, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vae = ConvVAE(z_size=z_size)
        vae_state_dict = torch.load(str(state_path))
        self._vae.load_state_dict(vae_state_dict, strict=True)
        self._perturbation_strength = perturbation_strength
        self._vae.requires_grad_(False)

    @property
    def vae(self):
        return self._vae

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = self.get_mask(x)
        latent = self._vae.reparameterize(*self._vae.encoder(x))
        perturbation = self._vae.decoder(latent)
        delta = torch.nn.functional.normalize(perturbation.view(-1), dim=0).view_as(perturbation)
        delta *= mask
        return x + delta

    def fit(self, policy: nn.Module, obs: np.ndarray) -> Generator[Tuple[Number, Number], None, None]:
        obs_torch = self.preproc_obs(obs)
        with torch.no_grad():
            gt_action, _, _ = policy(obs_torch)
        optimizer = optim.Adam(self._vae.decoder.parameters(), lr=self.LEARNING_RATE)
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



class VAELatentPerturbation(Perturbation):
    LEARNING_RATE = 1e-5
    GRAD_ACCUM_STEPS = 128
    N_ITERS = 100
    def __init__(self, state_path: Path, z_size: int = 32, perturbation_strength=0.02, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vae = ConvVAE(z_size=z_size)
        vae_state_dict = torch.load(str(state_path))
        self._vae.load_state_dict(vae_state_dict, strict=True)
        self._perturbation_strength = perturbation_strength
        self._vae.requires_grad_(False)
        self.latent_delta = nn.Parameter(torch.zeros(z_size))

    @property
    def vae(self):
        return self._vae

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar = self._vae.encoder(x)
        latent = self._vae.reparameterize(mu, logvar)
        normalized_delta = torch.nn.functional.normalize(self.latent_delta, dim=0)
        latent += normalized_delta * self._perturbation_strength
        return self._vae.decoder(latent)

    def fit(self, policy: nn.Module, obs: np.ndarray) -> Generator[Tuple[Number, Number], None, None]:
        obs_torch = self.preproc_obs(obs)
        with torch.no_grad():
            gt_action, _, _ = policy(obs_torch)
        optimizer = optim.Adam([self.latent_delta], lr=self.LEARNING_RATE)
        accumulated_loss = 0  # To track aggregated loss for logging

        for i in tqdm(range(self.N_ITERS)):
            optimizer.zero_grad()  # Reset gradients at the beginning of accumulation

            for _ in range(self.GRAD_ACCUM_STEPS):
                perturbed_obs = self(obs_torch)
                action, _, _ = policy(perturbed_obs)

                loss = -F.cosine_similarity(action, gt_action, dim=-1).mean()
                loss.backward()  # Accumulate gradients
                accumulated_loss += loss.item()

            optimizer.step()  # Apply accumulated gradients
            accumulated_loss /= self.GRAD_ACCUM_STEPS  # Average loss for logging

            yield i, accumulated_loss  # Report the averaged loss
            accumulated_loss = 0  # Reset loss accumulator for the next iteration