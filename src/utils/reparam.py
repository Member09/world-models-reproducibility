import torch

def reparameterize(mu, logvar):
    """
    The Reparameterization Trick: z = mu + std * epsilon
    Ensures the sampling process is differentiable.
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std