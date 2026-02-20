import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import os

from src.models.vae import VAE
from src.utils.dataset import BouncingBallDataset
from torch.utils.data import DataLoader

def loss_function(recon_x, x, mu, logvar, beta=1.0):
    # 1. Reconstruction Loss (Binary Cross Entropy or MSE)
    # Using MSE as it's more standard for continuous pixel values
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') # reduction = 'sum', means we sum errors over all pixels and batch

    # 2. KL Divergence: How much our 'Bubble' deviates from a Unit Gaussian
    # Formula: 0.5 * sum(exp(logvar) + mu^2 - 1 - logvar)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + (beta * kl_loss), recon_loss, kl_loss

def train_vae(epochs=50, batch_size=64, learning_rate=1e-3, latent_dim=32):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    
    # Initialize Model, Dataset, and Loader
    model = VAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    dataset = BouncingBallDataset(data_dir='data/raw', mode='frame')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"Starting VAE Training on {device}...")

    best_loss = float('inf')  # Initialize with infinity
    save_path = 'models/vae'
    os.makedirs(save_path, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss, total_recon, total_kl = 0, 0, 0
        
        # Beta Scheduler: Slowly increase KL weight from 0 to 1
        # This prevents the model from ignoring the latent space early on
        beta = min(1.0, epoch / 20.0) 

        loop = tqdm(loader, desc=f"Epoch [{epoch+1}/{epochs}]")
        for batch in loop:
            batch = batch.to(device)

            # Forward Pass
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch)
            
            # Loss Calculation
            loss, recon, kl = loss_function(recon_batch, batch, mu, logvar, beta=beta)
            
            # Backward Pass
            loss.backward()
            optimizer.step()

            # Metrics
            train_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()
            loop.set_postfix(loss=loss.item() / len(batch), beta=beta)

        # Calculate average loss for this epoch
        avg_loss = train_loss / len(loader.dataset)
        
        # Check if this is the best version we've seen so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(save_path, 'vae_best.pth'))
            print(f"--> New Best Model saved with loss: {best_loss:.4f}")

        # Still keep your interval saves for safety/checkpoints
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'vae_epoch_{epoch+1}.pth'))


    print("Training Complete. Weights saved in models/vae/")

if __name__ == "__main__":
    train_vae()