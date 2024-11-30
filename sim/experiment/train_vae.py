import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from imitation.data import serialize
from omegaconf import DictConfig
import hydra

from dataset.traj import TrajectoryDataset
from definitions import ROOT_PATH
from sim.experiment.experiment import BaseExperiment
from models.vae import ConvVAE


class TrainVAEExperiment(BaseExperiment):
    def __init__(self, rollouts_path: Path, z_size: int = 32, batch_size: int = 100,
                 learning_rate: float = 0.0001, kl_tolerance: float = 0.5, num_epochs: int = 10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.z_size = z_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.kl_tolerance = kl_tolerance
        self.num_epochs = num_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize VAE model
        self._vae = ConvVAE(z_size=z_size).to(self.device)

        # Load dataset
        self._dataset: TrajectoryDataset = TrajectoryDataset(serialize.load(rollouts_path))

        # Optimizer and loss components
        self.optimizer = optim.Adam(self._vae.parameters(), lr=learning_rate)
        self.reconstruction_loss_fn = nn.MSELoss(reduction='sum')  # Assume reconstruction loss is MSE

    def run(self):
        self._vae.train()
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            total_recon_loss = 0.0
            total_kl_loss = 0.0
            dataloader = DataLoader(self._dataset, self.batch_size, shuffle=True)

            # Example of iterating over the data loader
            for batch_obs, batch_acts in dataloader:
                # Convert numpy arrays to tensors if necessary
                batch_obs = torch.tensor(batch_obs)
                batch_acts = torch.tensor(batch_acts)
                # Your training code here

            # Logging
            print(
                f"Epoch {epoch + 1}/{self.num_epochs}, Total Loss: {total_loss:.4f}, Recon Loss: {total_recon_loss:.4f}, KL Loss: {total_kl_loss:.4f}")

        # Save model
        torch.save(self._vae.state_dict(), str(self._out_path / "vae_model.pth"))
        print("Model saved as vae_model.pth")


@hydra.main(config_name="train_vae", config_path=str(ROOT_PATH / "config"))
def main(cfg: DictConfig):
    exp = TrainVAEExperiment(**cfg)
    exp.run()


if __name__ == "__main__":
    main()
