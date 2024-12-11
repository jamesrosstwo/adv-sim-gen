from abc import ABC, abstractmethod
from numbers import Number
from typing import Generator, Tuple

import numpy as np
import torch
from torch import nn


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


    @abstractmethod
    def reset_params(self):
        pass


    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


    def generate_perturbations(self, obs, n: int):
        o = self.preproc_obs(obs).cuda()
        for i in range(n):
            self.reset_params()
            yield i, self(o)
