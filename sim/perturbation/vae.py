from pathlib import Path

import torch

from models.vae import ConvVAE
from perturbation.perturbation import Perturbation


class VAEFramePerturbation(Perturbation):
    def __init__(self, state_path: Path, z_size: int = 32, perturbation_strenth=0.5):
        self._vae = ConvVAE(z_size=z_size)
        vae_state_dict = torch.load(str(state_path))
        self._vae.load_state_dict(vae_state_dict, strict=True)
        self._vae.encoder.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = self.get_mask(x)
        latent = self._vae.encoder(x)
        perturbation = self._vae.decoder(latent)
        perturbed = (x * ~mask) + (perturbation * mask)
        return perturbed


class VAELatentPerturation(Perturbation):
    def __init__(self, state_path: Path, z_size: int = 32, perturbation_strenth=0.5):
        self._vae = ConvVAE(z_size=z_size)
        vae_state_dict = torch.load(str(state_path))
        self._vae.load_state_dict(vae_state_dict, strict=True)
        self._vae.encoder.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = self.get_mask(x)
        latent = self._vae.encoder(x)
        perturbation = self._vae.decoder(latent)
        perturbed = (x * ~mask) + (perturbation * mask)
        return perturbed
