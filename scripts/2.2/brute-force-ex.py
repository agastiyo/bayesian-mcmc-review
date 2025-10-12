#%%
# --- Radial Velocity Example ---
# This script demonstrates simple Bayesian parameter estimation for a 
# sinusoidal radial velocity model of the form:
# v(t) = K * sin(2 * pi * t / P + phi)

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)  # The meaning of life for reproducibility

#%%
# --- True Parameter Values ---
K = 10.0            # Velocity semi-amplitude [m/s]
P = 21.4            # Period of the signal [days]
phi = 0.25 * np.pi  # Phase offset [radians]
sig_ob = 3.0        # Observational uncertainty [m/s]

#%%
# --- Generate Synthetic Data ---
# Simulate 34 observations across 42 days
t = np.sort(42 * np.random.rand(34))

# Observed velocities = true model + Gaussian noise
v_ob = K * np.sin((2 * np.pi * t / P) + phi) + np.random.normal(0, sig_ob, size=len(t))

#%%
# --- Plot Observations Only ---
plt.figure(figsize=(6, 4))
plt.errorbar(
    t, v_ob, yerr=sig_ob, fmt='o', color='black',
    ecolor='gray', elinewidth=1, capsize=2,
    label='Observations', markersize=4
)
plt.xlabel("Time [days]", fontsize=12)
plt.ylabel("Radial velocity [m/s]", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

#%%
# --- Plot Observations and True Model ---
t_dense = np.linspace(0, 42, 1000)
plt.figure(figsize=(6, 4))
plt.errorbar(
    t, v_ob, yerr=sig_ob, fmt='o', color='black',
    ecolor='gray', elinewidth=1, capsize=2,
    label='Observations', markersize=4
)
plt.plot(t_dense, K * np.sin(2 * np.pi * (t_dense / P) + phi),
         'r--', linewidth=2, label='True model')
plt.xlabel("Time [days]", fontsize=12)
plt.ylabel("Radial velocity [m/s]", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

#%%
# --- Brute-Force Bayesian Parameter Estimation ---
# Define prior, likelihood, and posterior functions

# Uniform prior over plausible parameter ranges
def log_prior(K, P, phi):
    if 5 < K < 15 and 20 < P < 30 and 0 < phi < 2 * np.pi:
        return -np.log(10 * 10 * 2 * np.pi)
    else:
        return -np.inf

# Gaussian likelihood assuming constant sigma
def log_likelihood(K, P, phi):
    v_model = K * np.sin((2 * np.pi * t / P) + phi)
    return -0.5 * np.sum(
      ((v_ob - v_model)/sig_ob)**2 + np.log(2*np.pi*sig_ob**2)
    )

#%%
# --- Evaluate Posterior on a Grid ---
K_vals = np.linspace(5, 15, 100)
P_vals = np.linspace(20, 30, 100)
phi_vals = np.linspace(0, 2 * np.pi, 100)
posterior_vals = np.zeros((len(K_vals), len(P_vals), len(phi_vals)))

norm = 0
for i in range(len(K_vals)):
    for j in range(len(P_vals)):
        for k in range(len(phi_vals)):
            posterior_vals[i, j, k] = np.exp(log_prior(K_vals[i], P_vals[j], phi_vals[k]) + log_likelihood(K_vals[i], P_vals[j], phi_vals[k]))
            norm += posterior_vals[i, j, k]
posterior_vals /= norm

#%%
# --- Compute Marginal Posteriors and Statistics ---
posterior_K = np.sum(np.sum(posterior_vals, axis=2), axis=1)
posterior_P = np.sum(np.sum(posterior_vals, axis=0), axis=1)
posterior_phi = np.sum(np.sum(posterior_vals, axis=0), axis=0)

mean_K = np.sum(K_vals * posterior_K) / np.sum(posterior_K)
std_K  = np.sqrt(np.sum((K_vals - mean_K) ** 2 * posterior_K) / np.sum(posterior_K))
mean_P = np.sum(P_vals * posterior_P) / np.sum(posterior_P)
std_P  = np.sqrt(np.sum((P_vals - mean_P) ** 2 * posterior_P) / np.sum(posterior_P))
mean_phi = np.sum(phi_vals * posterior_phi) / np.sum(posterior_phi)
std_phi  = np.sqrt(np.sum((phi_vals - mean_phi) ** 2 * posterior_phi) / np.sum(posterior_phi))

#%%
# --- Plot Marginal Posterior Distributions ---
plt.figure(figsize=(4, 4))
plt.plot(K_vals, posterior_K, color='blue')
plt.axvline(K, color='r', ls='--', label='True K')
plt.xlabel("K [m/s]")
plt.title("Posterior of K")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
print(f"mu={mean_K:.2f}, sigma={std_K:.2f}")

plt.figure(figsize=(4, 4))
plt.plot(P_vals, posterior_P, color='green')
plt.axvline(P, color='r', ls='--', label='True T')
plt.xlabel("P [days]")
plt.title("Posterior of T")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
print(f"mu={mean_P:.2f}, sigma={std_P:.2f}")

plt.figure(figsize=(4, 4))
plt.plot(phi_vals, posterior_phi, color='orange')
plt.axvline(phi, color='r', ls='--', label='True phi')
plt.xlabel("phi [radians]")
plt.title("Posterior of phi")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
print(f"mu={mean_phi:.2f}, sigma={std_phi:.2f}")

#%%
# --- Compare True Model vs Posterior Predictions ---
plt.figure(figsize=(8, 5))
N_samples = 500

for i in range(N_samples):
    K_draw = np.random.choice(K_vals, p=posterior_K)
    P_draw = np.random.choice(P_vals, p=posterior_P)
    phi_draw = np.random.choice(phi_vals, p=posterior_phi)
    plt.plot(t_dense, K_draw * np.sin(2 * np.pi * (t_dense / P_draw) + phi_draw),
             color='lightblue', alpha=0.2)

plt.plot(t_dense, K * np.sin(2 * np.pi * (t_dense / P) + phi),
         'r--', lw=2, label='True model')
plt.plot(t_dense, mean_K * np.sin(2 * np.pi * (t_dense / mean_P) + mean_phi),
         'b-', lw=2, label='Posterior mean prediction')

plt.errorbar(t, v_ob, yerr=sig_ob, fmt='o', color='black',
             ecolor='gray', elinewidth=1, capsize=2,
             label='Observations', markersize=4)
plt.xlabel("Time [days]")
plt.ylabel("Radial velocity [m/s]")
plt.title("Posterior Mean and Uncertainty vs True Model")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()
# %%
