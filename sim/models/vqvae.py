import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Encoder layers with padding=1
        self.enc_conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)   # 96x96x3 -> 48x48x32
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # 48x48x32 -> 24x24x64
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # 24x24x64 -> 12x12x128
        self.enc_conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1) # 12x12x128 -> 6x6x256

    def forward(self, x):
        h = F.relu(self.enc_conv1(x))
        h = F.relu(self.enc_conv2(h))
        h = F.relu(self.enc_conv3(h))
        h = self.enc_conv4(h)  # Note: No activation function here
        return h  # Output shape: (batch_size, 256, 6, 6)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.dec_deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 6x6x256 -> 12x12x128
        self.dec_deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   # 12x12x128 -> 24x24x64
        self.dec_deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)    # 24x24x64 -> 48x48x32
        self.dec_deconv4 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)     # 48x48x32 -> 96x96x3

    def forward(self, z):
        h = F.relu(self.dec_deconv1(z))
        h = F.relu(self.dec_deconv2(h))
        h = F.relu(self.dec_deconv3(h))
        x_hat = torch.tanh(self.dec_deconv4(h))  # Output range: (-1, 1)
        return x_hat

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # Initialize the embeddings
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, inputs):
        # inputs: (batch_size, embedding_dim, height, width)
        # Permute to (batch_size, height, width, embedding_dim)
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape  # (batch_size, height, width, embedding_dim)

        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)  # (batch_size * height * width, embedding_dim)

        # Compute distances to embeddings
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        # Encoding indices
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        device = inputs.device

        # One-hot encodings
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantized inputs
        quantized = torch.matmul(encodings, self.embedding.weight)
        quantized = quantized.view(input_shape)

        # Compute losses
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()

        # Permute back to (batch_size, embedding_dim, height, width)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return quantized, loss

class VQVAE(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=256, commitment_cost=0.25):
        super(VQVAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

    def forward(self, x):
        z_e = self.encoder(x)  # Encoder output
        z_q, vq_loss = self.vq_layer(z_e)  # Quantized latent vectors and loss
        x_hat = self.decoder(z_q)  # Decoder output
        return x_hat, vq_loss

if __name__ == "__main__":
    test_input = torch.zeros((100, 3, 96, 96))
    model = VQVAE(num_embeddings=32, embedding_dim=16, commitment_cost=0.25)
    output, vq_loss = model(test_input)
    print("Output shape:", output.shape)
    print("VQ Loss:", vq_loss.item())
