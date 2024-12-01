from pathlib import Path

import torch
from torch import nn

from models.vae import ConvVAE
from perturbation.perturbation import Perturbation


class ConvPerturbation(Perturbation):
    def __init__(self, state_path: Path, perturbation_strength=0.5):
        pass

    @property
    def vae(self):
        return self._vae

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = self.get_mask(x)
        latent = self._vae.encoder(x)
        normalized_delta = torch.nn.functional.normalize(self.latent_delta, dim=0)
        latent += normalized_delta * self._perturbation_strength
        perturbation = self._vae.decoder(latent)
        perturbed = (x * ~mask) + (perturbation * mask)
        return perturbed
