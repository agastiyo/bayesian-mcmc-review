#%%

# Radial velocity example:

# v(t) = K sin(2πt/P + phi)

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42) # The meaning of life (for reproducability)

#%%

# True Parameter Values

K = 10 #m/s
P = 21.4 #days
phi = 0.25*np.pi #rad

sig_ob = 3.0 #observational uncertainty

#%%

# Create data

t = np.sort(42 * np.random.rand(34)) # 34 observations over 42 days

v_ob = K*np.sin(((2*np.pi*t)/P) + phi) + np.random.normal(0,sig_ob,size=len(t))

#%%

# Plot the data without the fit model

plt.figure(figsize=(6,4))

plt.errorbar(t, v_ob, yerr=sig_ob, fmt='o', color='black',ecolor='gray', elinewidth=1, capsize=2, label='Observations', markersize=4)

plt.xlabel("Time [days]", fontsize=12)
plt.ylabel("Radial velocity [m/s]", fontsize=12)

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# %%

# Plot the data with the fit model

plt.figure(figsize=(6,4))

plt.errorbar(t, v_ob, yerr=sig_ob, fmt='o', color='black',ecolor='gray', elinewidth=1, capsize=2, label='Observations', markersize=4)

t_dense = np.linspace(0, 42, 1000)
plt.plot(t_dense, K*np.sin(2*np.pi*(t_dense/P)+phi), 'r--',linewidth=2, label='True model')

plt.xlabel("Time [days]", fontsize=12)
plt.ylabel("Radial velocity [m/s]", fontsize=12)

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
# %%

# Now we are actually going to do the parameter estimation with brute force Bayesian inference

# Define the prior, likelihood, and posterior functions

def prior(K, P, phi):
  if 5 < K < 15 and 20 < P < 30 and 0 < phi < 2*np.pi:
    return 1/(10*10*2*np.pi) # Uniform prior
  else:
    return 0

# Constant sigma so we can treat it as a constant
def likelihood(K, P, phi, t, v_ob, sig_ob):
  v_model = K * np.sin((2 * np.pi * t / P) + phi)
  
  return np.prod((1/(np.sqrt(2 * np.pi) * sig_ob)) * np.exp(-((v_ob - v_model)**2)/(2* sig_ob**2)))

def posterior(K, P, phi, t, v_ob, sig_ob):
  return likelihood(K, P, phi, t, v_ob, sig_ob) * prior(K, P, phi)

#%%

# Calculate posterior over a grid of K, P, phi values and normalize
norm = 0
K_vals = np.linspace(5, 15, 50)
P_vals = np.linspace(20, 30, 50)
phi_vals = np.linspace(0, 2*np.pi, 50)
posterior_vals = np.zeros((len(K_vals), len(P_vals), len(phi_vals)))

for i, K_idx in enumerate(K_vals):
  for j, P_idx in enumerate(P_vals):
    for k, phi_idx in enumerate(phi_vals):
      posterior_vals[i,j,k] = posterior(K_idx, P_idx, phi_idx, t, v_ob, sig_ob)
      norm += posterior_vals[i,j,k]
posterior_vals /= norm

#%%

# Plot the posterior distributions
plt.figure(figsize=(8, 6))

# Calculate marginal distributions
posterior_K = np.sum(np.sum(posterior_vals, axis=2), axis=1)
posterior_P = np.sum(np.sum(posterior_vals, axis=0), axis=1)
posterior_phi = np.sum(np.sum(posterior_vals, axis=0), axis=0)

# Calculate means and standard deviations
mean_K = np.sum(K_vals * posterior_K) / np.sum(posterior_K)
std_K = np.sqrt(np.sum((K_vals - mean_K)**2 * posterior_K) / np.sum(posterior_K))

mean_P = np.sum(P_vals * posterior_P) / np.sum(posterior_P)
std_P = np.sqrt(np.sum((P_vals - mean_P)**2 * posterior_P) / np.sum(posterior_P))

mean_phi = np.sum(phi_vals * posterior_phi) / np.sum(posterior_phi)
std_phi = np.sqrt(np.sum((phi_vals - mean_phi)**2 * posterior_phi) / np.sum(posterior_phi))

# Plot K
plt.figure(figsize=(4,4))
plt.plot(K_vals, posterior_K, color='blue', label=f"μ={mean_K:.2f}, σ={std_K:.2f}")
plt.xlabel("K [m/s]")
plt.title("Posterior of K")
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Plot P
plt.figure(figsize=(4,4))
plt.plot(P_vals, posterior_P, color='green', label=f"μ={mean_P:.2f}, σ={std_P:.2f}")
plt.xlabel("T [days]")
plt.title("Posterior of T")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Plot phi
plt.figure(figsize=(4,4))
plt.plot(phi_vals, posterior_phi, color='red', label=f"μ={mean_phi:.2f}, σ={std_phi:.2f}")
plt.xlabel("ϕ [radians]")
plt.title("Posterior of ϕ")
plt.legend(loc="upper right")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
# %%

# Creating the final plot that compares the predicted and real model

plt.figure(figsize=(8, 5))
N_samples = 500

for i in range(N_samples):
  K_draws = np.random.choice(K_vals, p=posterior_K)
  P_draws = np.random.choice(P_vals, p=posterior_P)
  phi_draws = np.random.choice(phi_vals, p=posterior_phi)
  
  plt.plot(t_dense, K_draws * np.sin(2*np.pi*(t_dense/P_draws)+phi_draws), color='lightblue', alpha=0.2)

plt.plot(t_dense, K*np.sin(2*np.pi*(t_dense/P)+phi), 'r--',lw=2, label='True model')
plt.plot(t_dense, mean_K * np.sin(2*np.pi*(t_dense/mean_P) + mean_phi), 'b-', lw=2, label='Posterior mean prediction')
plt.errorbar(t, v_ob, yerr=sig_ob, fmt='o', color='black',ecolor='gray', elinewidth=1, capsize=2, label='Observations', markersize=4)

plt.xlabel("Time [days]")
plt.ylabel("Radial velocity [m/s]")
plt.title("Posterior Mean and Uncertainty vs True Model")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()
# %%
