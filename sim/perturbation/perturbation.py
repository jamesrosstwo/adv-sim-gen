from abc import ABC, abstractmethod
from numbers import Number
from typing import Generator, Tuple

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm


class Perturbation(nn.Module, ABC):
    LEARNING_RATE = 3e-4
    N_ITERS = 10000

    @staticmethod
    def preproc_obs(obs):
        batch_obs = torch.tensor(obs).float().cuda()
        batch_obs /= 255.0
        batch_obs = batch_obs.permute(0, 3, 1, 2)  # Rearrange to [batch_size, channels, w, h]
        return batch_obs


    @staticmethod
    def postproc_obs(obs):
        batch_obs = obs * 255.0
        batch_obs = batch_obs.permute(0, 2, 3, 1)
        return batch_obs.detach().cpu().numpy().astype(np.uint8)


    @abstractmethod
    def fit(self, policy, obs) -> Generator[Tuple[Number, Number], None, None]:
        raise NotImplementedError()


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