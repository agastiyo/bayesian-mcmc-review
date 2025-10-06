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
