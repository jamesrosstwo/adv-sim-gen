from abc import ABC, abstractmethod

import torch
from torch import nn


class Perturbation(nn.Module, ABC):
    def get_mask(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Obtains a binary mask of editable pixels given an observation
        :param obs: batch observation. Shape: [batch_size, 3, 96, 96]
        :return: Binary mask of shape [batch_size, 96, 96]
        """
        return obs


    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass