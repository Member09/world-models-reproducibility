import torch
import numpy as np
import os
from tqdm import tqdm
from src.models.vae import VAE

def extract():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Load the frozen Vision System
    latent_dim = 32
    vae = VAE(latent_dim=latent_dim).to(device)
    # Ensure you have a trained model checkpoint!
    vae.load_state_dict(torch.load('models/vae/vae_best.pth', map_location=device))
    vae.eval()

    raw_data_dir = 'data/raw'
    latent_data_dir = 'data/latents'
    os.makedirs(latent_data_dir, exist_ok=True)

    filenames = [f for f in os.listdir(raw_data_dir) if f.endswith('.npy')]

    print(f"Extracting latents from {len(filenames)} rollouts...")

    with torch.no_grad():
        for fname in tqdm(filenames):
            # Load raw rollout [Seq, C, H, W]
            raw_rollout = np.load(os.path.join(raw_data_dir, fname))
            raw_rollout = torch.from_numpy(raw_rollout).float().to(device) / 255.0

            # Encode to mu (we use mu for the RNN to avoid unnecessary noise)
            mu, _ = vae.encode(raw_rollout)
            
            # Save the compressed sequence [Seq, latent_dim]
            np.save(os.path.join(latent_data_dir, fname), mu.cpu().numpy())

if __name__ == "__main__":
    extract()