import torch

def lobotomy_test(model, input_data, mode='none'):
    """
    Modes: 
    'blind'    - Zero out latent vector z (Sever VAE -> RNN)
    'dementia' - Zero out RNN hidden state h (Sever Memory)
    'bottleneck' - Keep only 1 or 2 latent dimensions
    """
    # 1. Encode via VAE
    mu, logvar = model.vae.encode(input_data)
    z = model.vae.reparameterize(mu, logvar)
    
    # --- THE LOBOTOMY OVERRIDES ---
    
    if mode == 'blind':
        # Feed the RNN nothing but zeros
        z = torch.zeros_like(z)
        
    elif mode == 'bottleneck':
        # Mask all but the first 2 dimensions
        mask = torch.zeros_like(z)
        mask[:, :2] = 1.0
        z = z * mask
        
    # 2. Pass through RNN
    # Assuming hidden state is managed internally or passed
    h = model.rnn_init_hidden()
    
    if mode == 'dementia':
        # Force hidden state to zero at every step in a loop
        # (Simplified logic for a single step here)
        h = torch.zeros_like(h)

    output, next_h = model.rnn(z, h)
    
    # 3. Decode for visualization
    reconstruction = model.vae.decode(z)
    
    return reconstruction, output

# --- QUICK RUN ---
# modes = ['none', 'blind', 'dementia', 'bottleneck']
# for m in modes:
#    recon, pred = lobotomy_test(my_model, my_batch, mode=m)
#    save_plot(recon, title=f"Result of {m}")