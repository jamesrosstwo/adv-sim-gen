import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, z_size):
        super(Encoder, self).__init__()
        self.z_size = z_size

        # Encoder layers with padding=1
        self.enc_conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)   # 96x96x3 -> 48x48x32
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # 48x48x32 -> 24x24x64
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # 24x24x64 -> 12x12x128
        self.enc_conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1) # 12x12x128 -> 6x6x256
        self.fc_mu = nn.Linear(6 * 6 * 256, z_size)
        self.fc_logvar = nn.Linear(6 * 6 * 256, z_size)

    def forward(self, x):
        h = F.relu(self.enc_conv1(x))
        h = F.relu(self.enc_conv2(h))
        h = F.relu(self.enc_conv3(h))
        h = F.relu(self.enc_conv4(h))
        h = h.view(h.size(0), -1)  # Flatten
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, z_size):
        super(Decoder, self).__init__()
        self.z_size = z_size
        self.dec_fc = nn.Linear(z_size, 6 * 6 * 256)
        self.dec_deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 6x6x256 -> 12x12x128
        self.dec_deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   # 12x12x128 -> 24x24x64
        self.dec_deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)    # 24x24x64 -> 48x48x32
        self.dec_deconv4 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)     # 48x48x32 -> 96x96x3

    def forward(self, z):
        h = F.relu(self.dec_fc(z))
        h = h.view(h.size(0), 256, 6, 6)  # Reshape for deconvolution
        h = F.relu(self.dec_deconv1(h))
        h = F.relu(self.dec_deconv2(h))
        h = F.relu(self.dec_deconv3(h))
        h = torch.sigmoid(self.dec_deconv4(h))  # Sigmoid activation for output
        return h

class ConvVAE(nn.Module):
    def __init__(self, z_size):
        super(ConvVAE, self).__init__()
        self.z_size = z_size
        self.encoder = Encoder(z_size)
        self.decoder = Decoder(z_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

if __name__ == "__main__":
    test_input = torch.zeros((100, 3, 96, 96))
    model = ConvVAE(z_size=32)
    output, mu, logvar = model(test_input)
    print("Output shape:", output.shape)
