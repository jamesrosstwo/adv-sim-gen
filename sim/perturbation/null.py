import torch

from perturbation.perturbation import ObservationPerturbation


class NullPerturbation(ObservationPerturbation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
