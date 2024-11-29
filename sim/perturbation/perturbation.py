from abc import ABC, abstractmethod

import torch


class ObservationPerturbation(ABC):
    @abstractmethod
    def perturb(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Abstract method to perturb the observation.

        Args:
            observation (Any): The original observation from the environment.

        Returns:
            Any: The perturbed observation.
        """
        pass