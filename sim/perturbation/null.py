import torch

from perturbation.perturbation import Perturbation


class NullPerturbation(Perturbation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
