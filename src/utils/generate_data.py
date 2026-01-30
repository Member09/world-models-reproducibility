import numpy as np
import os
import argparse

def generate_bouncing_ball_data(num_videos=1000, seq_len=100, size=32, r=2, save_dir='data/raw'):
    """
    Generates and SAVES synthetic bouncing ball sequences to disk.
    Each rollout is saved as an individual .npy file for lazy-loading.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"Generating {num_videos} videos... saving to {save_dir}")

    for i in range(num_videos):
        video = np.zeros((seq_len, 1, size, size), dtype=np.uint8) # Save as uint8 to save space (instead of float32)
        
        # Initial physics state
        x, y = np.random.randint(r, size-r), np.random.randint(r, size-r) # random start position with margins
        vx, vy = np.random.choice([-1, 1]), np.random.choice([-1, 1]) # random initial velocity(-2 to +2 pixels/frame)

        for t in range(seq_len):
            Y, X = np.ogrid[:size, :size] # Create grid of coordinates
            dist = np.sqrt((X-x)**2 + (Y-y)**2) # calculate distance from center (x,y)
            frame = (dist <= r).astype(np.uint8) * 255 # Scale to 0-255 for uint8
            video[t, 0] = frame

            # Physics Update
            x += vx
            y += vy

            # Collision Logic
            if x <= r or x >= size - r: vx = -vx
            if y <= r or y >= size - r: vy = -vy

        # Save individual rollout
        np.save(os.path.join(save_dir, f'rollout_{i}.npy'), video)

if __name__ == "__main__":
    # to run: python src/utils/generate_data.py --num 1000
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=1000, help='Number of videos to generate')
    args = parser.parse_args()
    
    generate_bouncing_ball_data(num_videos=args.num)