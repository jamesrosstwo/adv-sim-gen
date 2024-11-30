from abc import ABC

import torch
from pathlib import Path
from imitation.data import serialize

from dataset.traj import TrajectoryDataset
from definitions import ROOT_PATH
from sim.experiment.experiment import BaseExperiment


class RolloutExperiment(BaseExperiment, ABC):
    def __init__(self, rollouts_path: str, *args, **kwargs):
        rollouts_path: Path = ROOT_PATH / rollouts_path
        assert rollouts_path.exists()
        super().__init__(*args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dataset: TrajectoryDataset = TrajectoryDataset(serialize.load(rollouts_path))