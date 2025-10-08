#%%
# --- Radial Velocity Example ---
# This script demonstrates simple Bayesian parameter estimation for a 
# sinusoidal radial velocity model of the form:
# v(t) = K * sin(2πt / P + φ)

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)  # Set random seed for reproducibility

#%%
# --- True Parameter Values ---

K = 10            # Velocity semi-amplitude [m/s]
P = 21.4          # Period of the signal [days]
phi = 0.25*np.pi  # Phase offset [radians]
sig_ob = 3.0      # Observational uncertainty [m/s]

#%%
# --- Generate Synthetic Data ---

# Simulate observation times: 34 points uniformly spaced across 42 days
t = np.sort(42 * np.random.rand(34))

# Generate noisy observed velocities using the true model + Gaussian noise
v_ob = K*np.sin(((2*np.pi*t)/P) + phi) + np.random.normal(0, sig_ob, size=len(t))

#%%
# --- Plot the Observations Only ---

plt.figure(figsize=(6,4))

# Plot each data point with error bars
plt.errorbar(
    t, v_ob, yerr=sig_ob, fmt='o', color='black',
    ecolor='gray', elinewidth=1, capsize=2,
    label='Observations', markersize=4
)

plt.xlabel("Time [days]", fontsize=12)
plt.ylabel("Radial velocity [m/s]", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# %%
# --- Plot the Observations and the True Model ---

plt.figure(figsize=(6,4))
plt.errorbar(
    t, v_ob, yerr=sig_ob, fmt='o', color='black',
    ecolor='gray', elinewidth=1, capsize=2,
    label='Observations', markersize=4
)

# Generate a dense time grid for plotting the smooth true model
t_dense = np.linspace(0, 42, 1000)
plt.plot(
    t_dense, K*np.sin(2*np.pi*(t_dense/P)+phi),
    'r--', linewidth=2, label='True model'
)

plt.xlabel("Time [days]", fontsize=12)
plt.ylabel("Radial velocity [m/s]", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# %%
# --- Brute-Force Bayesian Parameter Estimation ---
# We'll now define prior, likelihood, and posterior functions,
# and evaluate them over a 3D parameter grid.

# Uniform prior over reasonable ranges for K, P, and φ
def prior(K, P, phi):
    if 5 < K < 15 and 20 < P < 30 and 0 < phi < 2*np.pi:
        return 1 / (10 * 10 * 2 * np.pi)  # Normalized uniform prior
    else:
        return 0

# Gaussian likelihood function assuming constant σ
def likelihood(K, P, phi, t, v_ob, sig_ob):
    v_model = K * np.sin((2 * np.pi * t / P) + phi)
    return np.prod(
        (1/(np.sqrt(2 * np.pi) * sig_ob)) *
        np.exp(-((v_ob - v_model)**2) / (2 * sig_ob**2))
    )

# Posterior is proportional to prior × likelihood
def posterior(K, P, phi, t, v_ob, sig_ob):
    return likelihood(K, P, phi, t, v_ob, sig_ob) * prior(K, P, phi)

#%%
# --- Evaluate Posterior on a Grid ---

norm = 0  # Normalization constant
K_vals = np.linspace(5, 15, 50)
P_vals = np.linspace(20, 30, 50)
phi_vals = np.linspace(0, 2*np.pi, 50)

posterior_vals = np.zeros((len(K_vals), len(P_vals), len(phi_vals)))

# Compute posterior probability at each grid point
for i, K_idx in enumerate(K_vals):
    for j, P_idx in enumerate(P_vals):
        for k, phi_idx in enumerate(phi_vals):
            posterior_vals[i, j, k] = posterior(K_idx, P_idx, phi_idx, t, v_ob, sig_ob)
            norm += posterior_vals[i, j, k]

# Normalize the posterior distribution so total probability = 1
posterior_vals /= norm

#%%
# --- Compute Marginal Posterior Distributions and Statistics ---

# Integrate (sum) over other parameters to get 1D marginals
posterior_K   = np.sum(np.sum(posterior_vals, axis=2), axis=1)
posterior_P   = np.sum(np.sum(posterior_vals, axis=0), axis=1)
posterior_phi = np.sum(np.sum(posterior_vals, axis=0), axis=0)

# Compute mean and standard deviation for each parameter
mean_K = np.sum(K_vals * posterior_K) / np.sum(posterior_K)
std_K  = np.sqrt(np.sum((K_vals - mean_K)**2 * posterior_K) / np.sum(posterior_K))

mean_P = np.sum(P_vals * posterior_P) / np.sum(posterior_P)
std_P  = np.sqrt(np.sum((P_vals - mean_P)**2 * posterior_P) / np.sum(posterior_P))

mean_phi = np.sum(phi_vals * posterior_phi) / np.sum(posterior_phi)
std_phi  = np.sqrt(np.sum((phi_vals - mean_phi)**2 * posterior_phi) / np.sum(posterior_phi))

#%%
# --- Plot Marginal Posteriors ---

# Posterior for K
plt.figure(figsize=(4,4))
plt.plot(K_vals, posterior_K, color='blue', label=f"μ={mean_K:.2f}, σ={std_K:.2f}")
plt.xlabel("K [m/s]")
plt.title("Posterior of K")
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Posterior for P
plt.figure(figsize=(4,4))
plt.plot(P_vals, posterior_P, color='green', label=f"μ={mean_P:.2f}, σ={std_P:.2f}")
plt.xlabel("T [days]")
plt.title("Posterior of T")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Posterior for φ
plt.figure(figsize=(4,4))
plt.plot(phi_vals, posterior_phi, color='red', label=f"μ={mean_phi:.2f}, σ={std_phi:.2f}")
plt.xlabel("ϕ [radians]")
plt.title("Posterior of ϕ")
plt.legend(loc="upper right")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# %%
# --- Compare True Model vs Posterior Predictions ---

plt.figure(figsize=(8, 5))
N_samples = 500  # Number of random samples to visualize uncertainty

# Draw random parameter samples from the 1D marginal posteriors
for i in range(N_samples):
    K_draws   = np.random.choice(K_vals,   p=posterior_K)
    P_draws   = np.random.choice(P_vals,   p=posterior_P)
    phi_draws = np.random.choice(phi_vals, p=posterior_phi)

    # Plot random draws (light blue for posterior spread)
    plt.plot(
        t_dense,
        K_draws * np.sin(2*np.pi*(t_dense/P_draws) + phi_draws),
        color='lightblue', alpha=0.2
    )

# Plot true model (red dashed) and posterior mean model (blue solid)
plt.plot(t_dense, K*np.sin(2*np.pi*(t_dense/P)+phi), 'r--', lw=2, label='True model')
plt.plot(t_dense, mean_K * np.sin(2*np.pi*(t_dense/mean_P) + mean_phi),
         'b-', lw=2, label='Posterior mean prediction')

# Add observed data points
plt.errorbar(
    t, v_ob, yerr=sig_ob, fmt='o', color='black',
    ecolor='gray', elinewidth=1, capsize=2,
    label='Observations', markersize=4
)

plt.xlabel("Time [days]")
plt.ylabel("Radial velocity [m/s]")
plt.title("Posterior Mean and Uncertainty vs True Model")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()
# %%
