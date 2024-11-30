from abc import ABC, abstractmethod
from torch import nn


class DrivingPolicy(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    def forward(self, obs):
        raise NotImplementedError()