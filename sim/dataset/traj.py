import torch
from torch.utils.data import Dataset, DataLoader

class TrajectoryDataset(Dataset):
    def __init__(self, trajectories):
        """
        Initializes the dataset by flattening observations and actions from all trajectories.

        Args:
            trajectories (Sequence[Trajectory]): A sequence of Trajectory objects.
        """
        self.obs = []
        self.acts = []
        for traj in trajectories:
            # Extract observations and actions, aligning them properly.
            # We exclude the last observation to match the number of actions.
            obs = traj.obs[:-1]  # Shape: (trajectory_len, ...)
            acts = traj.acts     # Shape: (trajectory_len, ...)
            self.obs.append(obs)
            self.acts.append(acts)
        # Concatenate all observations and actions from different trajectories.
        self.obs = np.concatenate(self.obs, axis=0)
        self.acts = np.concatenate(self.acts, axis=0)

    def __len__(self):
        return len(self.acts)

    def __getitem__(self, idx):
        # Return the observation-action pair at the specified index.
        obs = self.obs[idx]
        act = self.acts[idx]
        return obs, act