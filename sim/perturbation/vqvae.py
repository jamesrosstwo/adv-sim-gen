from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Generator, Tuple

from models.vqvae import VQVAE
from perturbation.perturbation import Perturbation
from util.frame import get_mask

class VQVAEFramePerturbation(Perturbation):
    LEARNING_RATE = 1e-5
    GRAD_ACCUM_STEPS = 32
    N_ITERS = 10

    def __init__(self, state_path: Path, perturbation_strength=0.02, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vqvae = VQVAE()
        vqvae_state_dict = torch.load(str(state_path))
        self._vqvae.load_state_dict(vqvae_state_dict, strict=True)
        self._perturbation_strength = perturbation_strength
        self._vqvae.requires_grad_(False)
        self.perturbation = nn.Parameter(torch.zeros(1, 3, 96, 96))

    @property
    def vqvae(self):
        return self._vqvae

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        perturbation = self.perturbation
        perturbation = self._normalize_perturbation(perturbation)
        mask = get_mask(x)
        perturbation *= mask
        output = x + perturbation
        output = torch.clamp(output, 0, 1)
        return output

    def _normalize_perturbation(self, perturbation):
        pixel_norms = torch.norm(perturbation.view(perturbation.size(0), -1), dim=1, keepdim=True)
        max_norm = torch.max(pixel_norms)
        if max_norm > self._perturbation_strength:
            perturbation = perturbation * (self._perturbation_strength / max_norm)
        return perturbation

    def fit(self, policy: nn.Module, obs: np.ndarray) -> Generator[Tuple, None, None]:
        obs_torch = self.preproc_obs(obs)
        obs_torch.requires_grad = False
        with torch.no_grad():
            gt_action, _, _ = policy(obs_torch)
        optimizer = optim.Adam([self.perturbation], lr=self.LEARNING_RATE)
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

class VQVAELatentPerturbation(Perturbation):
    LEARNING_RATE = 1e-3
    GRAD_ACCUM_STEPS = 32
    N_ITERS = 100

    def __init__(self, state_path: Path, perturbation_strength=0.02, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vqvae = VQVAE()
        vqvae_state_dict = torch.load(str(state_path))
        self._vqvae.load_state_dict(vqvae_state_dict, strict=True)
        self._perturbation_strength = perturbation_strength
        self._vqvae.requires_grad_(False)
        self.latent_perturbation = None

    @property
    def vqvae(self):
        return self._vqvae

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._vqvae.encoder.requires_grad_(True)
        z_e = self._vqvae.encoder(x)
        if self.latent_perturbation is None:
            self.latent_perturbation = nn.Parameter(torch.zeros_like(z_e))
        z_e_perturbed = z_e + self._normalize_perturbation(self.latent_perturbation)
        z_q, _ = self._vqvae.vq_layer(z_e_perturbed)
        x_recon = self._vqvae.decoder(z_q)
        self._vqvae.encoder.requires_grad_(False)
        return x_recon

    def _normalize_perturbation(self, perturbation):
        perturbation_flat = perturbation.view(perturbation.size(0), -1)
        perturbation_norm = torch.norm(perturbation_flat, dim=1, keepdim=True)
        max_norm = torch.max(perturbation_norm)
        if max_norm > self._perturbation_strength:
            perturbation = perturbation * (self._perturbation_strength / max_norm)
        return perturbation

    def fit(self, policy: nn.Module, obs: np.ndarray) -> Generator[Tuple, None, None]:
        obs_torch = self.preproc_obs(obs)
        obs_torch.requires_grad = False
        with torch.no_grad():
            gt_action, _, _ = policy(obs_torch)
        optimizer = optim.Adam([self.latent_perturbation], lr=self.LEARNING_RATE)
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
