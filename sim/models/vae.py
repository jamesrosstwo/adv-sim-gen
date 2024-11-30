# Adapted from https://raw.githubusercontent.com/hardmaru/WorldModelsExperiments/refs/heads/master/carracing/vae/vae.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvVAE(nn.Module):
    def __init__(self, z_size):
        super(ConvVAE, self).__init__()
        self.z_size = z_size

        # Encoder
        self.enc_conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2)  # 64x64x3 -> 32x32x32
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)  # 32x32x32 -> 16x16x64
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2)  # 16x16x64 -> 8x8x128
        self.enc_conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2)  # 8x8x128 -> 4x4x256
        self.fc_mu = nn.Linear(2 * 2 * 256, z_size)
        self.fc_logvar = nn.Linear(2 * 2 * 256, z_size)

        # Decoder
        self.dec_fc = nn.Linear(z_size, 4 * 256)  # latent vector to 4x256
        self.dec_deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2)  # 1x1x4x256 -> 2x2x128
        self.dec_deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2)  # 2x2x128 -> 4x4x64
        self.dec_deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2)  # 4x4x64 -> 8x8x32
        self.dec_deconv4 = nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2)  # 8x8x32 -> 16x16x3

    def encode(self, x):
        h = F.relu(self.enc_conv1(x))
        h = F.relu(self.enc_conv2(h))
        h = F.relu(self.enc_conv3(h))
        h = F.relu(self.enc_conv4(h))
        h = h.view(h.size(0), -1)  # Flatten
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.dec_fc(z))
        h = h.view(h.size(0), 256, 1, 1)  # Reshape for deconvolution
        h = F.relu(self.dec_deconv1(h))
        h = F.relu(self.dec_deconv2(h))
        h = F.relu(self.dec_deconv3(h))
        h = torch.sigmoid(self.dec_deconv4(h))  # Sigmoid activation for output
        return h

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
