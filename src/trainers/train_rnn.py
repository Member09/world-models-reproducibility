import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from tqdm import tqdm
import os

from src.models.mdn_rnn import MDNRNN
from src.utils.dataset import BouncingBallDataset
from torch.utils.data import DataLoader

def mdn_loss_fn(pi, mu, logsigma, target):
    """
    Negative Log-Likelihood of a Gaussian Mixture Model.
    pi: [Batch, Seq, K]
    mu: [Batch, Seq, K, Z]
    logsigma: [Batch, Seq, K, Z]
    target: [Batch, Seq, Z]
    """
    sigma = torch.exp(logsigma)
    
    # Create a Normal distribution for each Gaussian in the mixture
    # target is expanded to match the K gaussians
    dist = Normal(mu, sigma)
    
    # Calculate log probability of the target under each Gaussian
    # We sum over the Z (latent) dimensions
    log_probs = dist.log_prob(target.unsqueeze(2).expand_as(mu))
    log_probs = torch.sum(log_probs, dim=-1) # Sum across latent dims
    
    # Combine with the mixing coefficients (pi)
    # Using log-sum-exp trick for numerical stability
    weighted_log_probs = log_probs + torch.log(pi + 1e-10)
    loss = -torch.logsumexp(weighted_log_probs, dim=-1)
    
    return torch.mean(loss)

def train_rnn(epochs=30, batch_size=64, seq_len=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Setup Data
    dataset = BouncingBallDataset(data_dir='data/latents', mode='sequence', seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Setup Model & Optimizer
    model = MDNRNN(z_dim=32, hidden_dim=256, n_gaussians=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-3)
    
    # 3. Track the Best Loss
    best_loss = float('inf')
    save_path = 'models/rnn'
    os.makedirs(save_path, exist_ok=True)

    print(f"Training MDN-RNN on {device}...")

    for epoch in range(epochs):
        model.train()
        total_epoch_loss = 0
        
        loop = tqdm(loader, desc=f"Epoch [{epoch+1}/{epochs}]")
        for obs, targets in loop:
            obs, targets = obs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            pi, mu, logsigma, _ = model(obs)
            loss = mdn_loss_fn(pi, mu, logsigma, targets)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # Check for Best Model
        avg_loss = total_epoch_loss / len(loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(save_path, 'rnn_best.pth'))
            print(f"--> New Best RNN saved (NLL: {best_loss:.4f})")

        # Periodic Checkpoint for safety
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'rnn_epoch_{epoch+1}.pth'))

if __name__ == "__main__":
    train_rnn(epochs=150, seq_len=50)