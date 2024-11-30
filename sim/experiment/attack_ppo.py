import torch
from hydra.utils import instantiate
from imitation.policies.serialize import load_policy
from omegaconf import DictConfig
from stable_baselines3 import PPO
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm

from definitions import ROOT_PATH
from models.ppo import PPOPolicy
from sim.experiment.rollout import RolloutExperiment
from perturbation.perturbation import Perturbation


# Idea 0: Train a VAE purely on reconstruction, and freeze the encoder. Reuse this encoder for all subsequent ideas.

# Idea 1: train a VAE that directly attacks the policy. in: image, VAE predicts a masked perturbation added to image.
#   Loss is difference between GT action and the estimated action by passed model

# Idea 2: Train a VAE that estimates the next frame. Takes in action and observation, loss is pixel loss between pred and actual frame?
# Or maybe loss is the difference between encodings?



class AttackExperiment(RolloutExperiment):
    def __init__(
            self,
            ppo_path: str,
            perturbation: DictConfig,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._perturbation: Perturbation = instantiate(perturbation)
        self._policy = PPOPolicy(PPO.load(ppo_path))

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
                recon_loss = F.mse_loss(recon_x, batch_obs, reduction='sum')
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
