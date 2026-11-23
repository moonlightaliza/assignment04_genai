import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalAutoEncoder(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        # encoder: 1x28x28 -> 32x14x14 -> 64x7x7 -> flatten
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Flatten())
        self.fc_mu = nn.Linear(64 * 7 * 7, d)
        self.fc_logvar = nn.Linear(64 * 7 * 7, d)
        # decoder
        self.fc_dec = nn.Linear(d, 64 * 7 * 7)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1))     # logits, NO sigmoid

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        sigma = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        z = mu + sigma * eps
        return z 

    def decode(self, z):
        h = self.fc_dec(z).view(-1, 64, 7, 7)
        return self.dec(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(logits, x, mu, logvar):
    batch_size = x.size(0)
    reconstruct = F.binary_cross_entropy_with_logits(logits, x, reduction = "sum")/batch_size
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    return reconstruct, kl_div