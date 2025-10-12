#%%
# --- Radial Velocity Example ---
# This script demonstrates simple MCMC parameter estimation for a 
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
sig_ob = 3.0        # Observational uncertainty (standard deviations)

#%%
# --- Generate Synthetic Data ---
# Simulate 34 observations across 42 days
t = np.sort(42 * np.random.rand(34))

# Observed velocities = true model + Gaussian noise
v_ob = K * np.sin((2 * np.pi * t / P) + phi) + np.random.normal(0, sig_ob, size=len(t))

#%%
# --- MCMC Parameter Estimation ---
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
# Define proposal distribution
def newProposal(current):
  step_size = [0.25, 0.25, 0.1] # Standard deviation for new proposals
  return [
    np.random.normal(current[0], step_size[0]),
    np.random.normal(current[1], step_size[1]),
    np.random.normal(current[2], step_size[2])
  ]

#%%
# Metropolis Algorithm
walks = 10
burn_steps = 1000
steps = 12000
chain = np.zeros((walks, burn_steps + steps + 1,3))

for w in range(walks):
  curr_state = [14,25,np.pi] # Initial guess, should be a good guess within the parameter space
  
  curr_prior = log_prior(*curr_state)
  curr_likelihood = log_likelihood(*curr_state)
  chain[w,0] = curr_state
  
  accept_count = 0
  
  for i in range(burn_steps + steps):
    new_proposal = newProposal(curr_state)
    
    new_prior = log_prior(*new_proposal)
    new_likelihood = -np.inf
    # Only calculate likelihood if the new proposal is in the parameter space
    if (new_prior != -np.inf):
      new_likelihood = log_likelihood(*new_proposal)

    # Accept or reject the proposal?
    new_log_post = new_prior + new_likelihood
    curr_log_post = curr_prior + curr_likelihood
    alpha = np.exp(new_log_post - curr_log_post)
    if alpha > 1:
      alpha = 1
    
    if np.random.rand() < alpha:
      curr_state = new_proposal
      curr_prior = new_prior
      curr_likelihood = new_likelihood
      accept_count += 1
    
    chain[w,i+1] = curr_state
  print(accept_count/(burn_steps + steps))

# Remove burn steps
chain = chain[:,burn_steps+1:]

#%%
# Plot the results of the algorithm
K_chain = chain[:,:,0]
P_chain = chain[:,:,1]
phi_chain = chain[:,:,2]
chain_range = np.arange(burn_steps,burn_steps+steps)

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# K trace
for p in range(len(chain)):
  axes[0].plot(chain_range, K_chain[p])
axes[0].axhline(y=K, color='r', ls='--', label='True K')
axes[0].set_ylabel('K [m/s]')
axes[0].set_title('Trace of K')
axes[0].legend()

# P trace
for p in range(len(chain)):
  axes[1].plot(chain_range, P_chain[p])
axes[1].axhline(y=P, color='r', ls='--', label='True T')
axes[1].set_ylabel('Period [days]')
axes[1].set_title('Trace of T')
axes[1].legend()

# phi trace
for p in range(len(chain)):
  axes[2].plot(chain_range, phi_chain[p])
axes[2].axhline(y=phi, color='r', ls='--', label='True phi')
axes[2].set_ylabel('Phase [rad]')
axes[2].set_xlabel('Step')
axes[2].set_title('Trace of phi')
axes[2].legend()

plt.tight_layout()
plt.show()
# %%
# Plot histograms
plt.figure(figsize=(4, 4))
plt.hist(K_chain.flatten(), bins=50, color='tab:blue', alpha=0.7)
plt.axvline(K, color='r', ls='--', label='True K')
plt.ylabel('Count')
plt.xlabel('K [m/s]')
plt.title('Posterior of K')
plt.legend()
plt.show()

plt.figure(figsize=(4, 4))
plt.hist(P_chain.flatten(), bins=50, color='tab:blue', alpha=0.7)
plt.axvline(P, color='r', ls='--', label='True T')
plt.ylabel('Count')
plt.xlabel('Period [days]')
plt.title('Posterior of T')
plt.legend()
plt.show()

plt.figure(figsize=(4, 4))
plt.hist(phi_chain.flatten(), bins=50, color='tab:blue', alpha=0.7)
plt.axvline(phi, color='r', ls='--', label='True phi')
plt.ylabel('Count')
plt.xlabel('Phase [rad]')
plt.title('Posterior of phi')
plt.legend()
plt.show()
# %%
# Calculate expectation values and standard deviations
K_mean = np.mean(K_chain)
K_std = np.std(K_chain)

P_mean = np.mean(P_chain)
P_std = np.std(P_chain)

phi_mean = np.mean(phi_chain)
phi_std = np.std(phi_chain)

print(f"K = {K_mean:.2f} +/- {K_std:.2f}")
print(f"P = {P_mean:.2f} +/- {P_std:.2f}")
print(f"phi = {phi_mean:.2f} +/- {phi_std:.2f}")
# %%
