import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import hydra
from tqdm import tqdm

from definitions import ROOT_PATH
from sim.experiment.rollout import RolloutExperiment
from models.vae import ConvVAE
import torch.nn.functional as F


class TrainVAEExperiment(RolloutExperiment):
    def __init__(
            self,
            z_size: int = 32,
            batch_size: int = 100,
            learning_rate: float = 0.00007,
            kl_tolerance: float = 0.5,
            num_epochs: int = 200,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.z_size = z_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.kl_tolerance = kl_tolerance
        self.num_epochs = num_epochs

        self._vae = ConvVAE(z_size=z_size).to(self.device)
        self.optimizer = optim.Adam(self._vae.parameters(), lr=learning_rate)
        self.reconstruction_loss_fn = nn.MSELoss(reduction='sum')

    def run(self):
        self._vae.train()
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            total_recon_loss = 0.0
            total_kl_loss = 0.0
            dataloader = DataLoader(self._dataset, self.batch_size, shuffle=True)

            for batch_obs, batch_acts in tqdm(dataloader):
                batch_obs = torch.tensor(batch_obs).float().to(self.device)
                batch_obs /= 255.0
                self.optimizer.zero_grad()
                recon_x, mu, logvar = self._vae(batch_obs)
                recon_loss = F.mse_loss(recon_x, batch_obs)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + self.kl_tolerance * kl_loss
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()

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
