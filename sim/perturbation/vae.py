import torch

from perturbation.perturbation import Perturbation


class VAEFramePerturbation(Perturbation):
    def __init__(self):
        self._

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class VAELatentPerturation(Perturbation):
    def __init__(self):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
