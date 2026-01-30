import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from src.models.vae import VAE
from src.models.mdn_rnn import MDNRNN

def sample_from_gmm(pi, mu, logsigma, mode='mean'):
    """
    pi: [1, 1, K]
    mu: [1, 1, K, Z]
    logsigma: [1, 1, K, Z]
    """
    if mode == 'mean':
        # Weighted average of all means - suppresses 'ghosting' by 
        # combining all probabilistic guesses into one center of mass.
        weighted_mu = (mu * pi.unsqueeze(-1)).sum(dim=2) # Result: [1, 1, Z]
        return weighted_mu.squeeze(0).squeeze(0)
    
    elif mode == 'greedy':
        k = torch.argmax(pi[0, -1]).item()
        return mu[0, -1, k]
    
    else: # Stochastic sampling (original)
        k = torch.multinomial(pi[0, -1], 1).item()
        sigma_k = torch.exp(logsigma[0, -1, k])
        return mu[0, -1, k] + torch.randn_like(sigma_k) * sigma_k

def run_dream_comparison(seq_len=60, context_len=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Load Models
    vae = VAE(latent_dim=32).to(device)
    vae.load_state_dict(torch.load('models/vae/vae_best.pth', map_location=device))
    vae.eval()

    rnn = MDNRNN(z_dim=32, hidden_dim=256, n_gaussians=5).to(device)
    rnn.load_state_dict(torch.load('models/rnn/rnn_best.pth', map_location=device))
    rnn.eval()

    # 2. Get Ground Truth for comparison
    # Load rollout_0 and normalize
    raw_rollout = np.load('data/raw/rollout_0.npy') / 255.0
    reality_frames = raw_rollout[:seq_len + context_len]
    
    dream_frames = []
    
    with torch.no_grad():
        # Warm-up (Context)
        context_tensor = torch.from_numpy(reality_frames[:context_len]).float().to(device)
        z_context, _ = vae.encode(context_tensor) 
        z_context = z_context.unsqueeze(0)        # Shape: [1, 15, 32]
        
        # RNN sees the whole history
        pi, mu, logsigma, hidden = rnn(z_context)
        
        # B. Start Dreaming
        # We only want the prediction for the VERY LAST step of the sequence
        # pi[:, -1:] ensures we get shape [1, 1, K] instead of [1, 15, K]
        current_z = sample_from_gmm(pi[:, -1:], mu[:, -1:], logsigma[:, -1:], mode='mean').view(1, 1, -1)

        for i in range(seq_len):
            # --- THE VAE HANDSHAKE ---
            # VAE Decoder expects [Batch, 32]. 
            # We must strip the sequence dimension: .view(1, -1)
            z_for_vae = current_z.view(1, -1) 
            decoded = vae.decode(z_for_vae).cpu().squeeze().numpy()
            dream_frames.append(decoded)
            
            # --- THE RNN HANDSHAKE ---
            # RNN expects [Batch, Seq=1, 32]
            pi, mu, logsigma, hidden = rnn(current_z, hidden)
            
            # Update for next iteration
            next_z = sample_from_gmm(pi, mu, logsigma, mode='mean')
            current_z = next_z.view(1, 1, -1)

    # 3. Side-by-Side Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.axis('off'); ax2.axis('off')
    
    # Use np.squeeze() to force (1, 32, 32) -> (32, 32)
    initial_real = np.squeeze(reality_frames[context_len])
    initial_dream = np.squeeze(dream_frames[0])

    img_real = ax1.imshow(initial_real, cmap='gray')
    ax1.set_title("Reality (Ground Truth)")
    
    img_dream = ax2.imshow(initial_dream, cmap='gray')
    ax2.set_title("Dream (RNN Prediction)")
    
    def update(i):
        # We use np.squeeze to aggressively remove all unit dimensions
        real_img = np.squeeze(reality_frames[context_len + i]) 
        dream_img = np.squeeze(dream_frames[i])
        
        img_real.set_data(real_img)
        img_dream.set_data(dream_img)
        return [img_real, img_dream]

    ani = FuncAnimation(fig, update, frames=len(dream_frames), interval=50, blit=True)
    
    os.makedirs('results/gifs', exist_ok=True)
    ani.save('results/gifs/dream_vs_reality.gif', writer=PillowWriter(fps=15))
    print("Done! Check results/gifs/dream_vs_reality.gif")

if __name__ == "__main__":
    run_dream_comparison()