# World Models Reproducibility 🌍🧠

> An independent research implementation of [World Models (Ha & Schmidhuber, 2018)](https://arxiv.org/abs/1803.10122).

This repository documents my journey to build and train an AI agent capable of learning a compressed spatial-temporal representation of its environment (a "World Model") and using that simulated reality to evolve driving reflexes.



## The Architecture

The system is broken down into three distinct neural networks, inspired by human cognitive processes:

1. **V (Vision) - Variational Autoencoder (VAE):** Compresses high-dimensional pixel inputs ($64 \times 64$ RGB frames) into a compact 32-dimensional latent vector ($z$). It learns the spatial features of the world using the Reparameterization Trick.
2. **M (Memory) - MDN-RNN:** A Long Short-Term Memory network combined with a Mixture Density Network. It looks at the VAE's output and the agent's actions to predict *multiple possible futures* by parameterizing a Gaussian Mixture Model.
3. **C (Controller) - Linear Neural Network:** A minimalist decision-making layer. It takes the current visual state (32 numbers) and the RNN's hidden state (256 numbers) to map out actions (Steering, Gas, Brake). Because the physics engine is non-differentiable, **C** is trained entirely via Covariance Matrix Adaptation Evolution Strategy (CMA-ES).

---

## The Experiments

I approached this paper by scaling up complexity across two distinct environments:

### Experiment 01: The 2D Spatial-Temporal Test (`01_bouncing_ball/`)
An implementation of the "World Models" architecture (Ha & Schmidhuber, 2018) applied to a 2D physics environment. This system learns to "dream" the future states of a bouncing ball environment by compressing visual data into a latent space and modeling the dynamics over time. 
* **Goal:** Can the VAE compress the ball's coordinates, and can the MDN-RNN accurately predict its future trajectory (including wall bounces)?
* **Result:** Successfully validated the log-sum-exp stabilization and Gaussian mixture predictions without the noise of an RL action-space.

### Experiment 02: Full Closed-Loop Control (`02_car_racing/`)
With the V and M components mathematically proven, I implemented the full pipeline in the `CarRacing-v3` OpenAI Gym environment.
* **Dataset:** Collected 10,000 episodes (10,000,000 frames) of random policy driving.
* **Evolution:** Spawned generations of 64 mutant controllers, using CMA-ES to evolve the weight matrix without backpropagation.
* **Result:** The agent successfully learned to navigate sharp turns and stay on the track using only its compressed internal "dream" of the world.



---

## 📂 Repository Structure

```text
world-models-reproducibility/
├── experiments/
│   ├── 01_bouncing_ball/           # Phase 1: V & M Isolation
│   └── 02_car_racing/              # Phase 2: Full V, M, C Evolution
│       ├── src/                    # Source code for data prep and models
│       ├── play.py                 # Real-time game rendering script
│       └── plot_scores.py          # CMA-ES visualization generator
├── .gitignore                      
├── requirements.txt                
└── README.md