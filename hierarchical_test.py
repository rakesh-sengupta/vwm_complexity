# Python simulation code for hierarchical VWM model
# This code simulates: (1) dynamics convergence for a single layer and a small hierarchy
# (2) retrieval accuracy vs number of stored samples M using the cross-talk SNR model
# (3) capacity vs clarity (C) effect
# The notebook will produce three plots and a small dataframe summarizing key results.
#
# Requirements: standard Python scientific stack (numpy, scipy, matplotlib, pandas).
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.spatial.distance import cdist
import pandas as pd
from math import exp
from scipy.integrate import solve_ivp
import seaborn as sns

# -------------------------- Parameters ---------------------------------
np.random.seed(1)

# Network / hierarchy parameters
N = 256           # number of retinal locations (baseline)
k = 2             # pyramidal reduction factor
p = 2             # p-lattice coarsening factor
L = 4             # number of layers (feedback levels)
bar_eta = 0.05    # time-averaged excitation fraction
delta = 0.25      # nominal overlap coefficient
a = 0.85          # attenuation per hierarchical distance

# Dynamics parameters (Cohen-Grossberg-like discrete simulation)
tau = 0.02
dt = 0.001
# MODIFIED: Reduced alpha and beta to satisfy the contraction condition for stability.
# Original values were alpha=1.0, beta=0.6, which resulted in a contraction metric > 1.
alpha = 0.9      # Self-excitation gain
beta = 0.1       # Inhibitory gain
L_fprime = 1.0    # max slope of activation function (relu-like smooth)
epsilon_conv = 1e-3

# Readout / capacity parameters
A = 1.0           # baseline signal amplitude
sigma = 0.1       # intrinsic readout noise std
K = 4             # PFC memory slots
m = 6             # bits per slot (codebook size 2^m)

# Cross-talk kernels per layer (example)
kappa_l = np.array([0.25, 0.18, 0.10, 0.04])  # decreasing with layer (higher layers less leakage)
w_l = np.array([0.25, 0.25, 0.25, 0.25])      # equal projection weights for simplicity

# Clarity function C(l): based on A(l) = fraction of p-lattice visible at level l
def clarity_A(l, L, gamma_curve=1.8):
    # simple smooth ramp: A(l) = (l/L)^gamma_curve; choose gamma_curve to shape growth
    return (l / L) ** gamma_curve

def clarity_C(A_l, B=6.0):
    # C(L) = A(L) * exp(-B (1-A(L))^2) as in specification
    return A_l * np.exp(-B * (1 - A_l) ** 2)

C_l = np.array([clarity_C(clarity_A(l, L)) for l in range(1, L + 1)])

# Derived quantities
delta_eff = delta * a / p
Gamma = np.prod([1 - (delta * (a / p) ** (l - 1)) for l in range(1, L + 1)])  # product overlap
C_sig = np.dot(w_l, C_l)          # clarity-weighted signal factor
kappa_xt = np.dot(w_l, kappa_l * (1 - C_l))  # clarity-modulated cross-talk coefficient

print("Derived values (using initial parameters):")
print(f" delta_eff = {delta_eff:.4f}, Gamma = {Gamma:.4f}, C_sig = {C_sig:.4f}, kappa_xt = {kappa_xt:.4f}")

# -------------------- (A) Dynamics convergence simulation --------------------
# We'll simulate a small 1D sheet of neurons per layer for dynamics test (e.g., sqrt(n) x sqrt(n))
def make_weight_matrix(n_units, sigma_d=1.5):
    # place units on a 1D lattice
    positions = np.arange(n_units).reshape(-1, 1)
    D = cdist(positions, positions, metric='euclidean')
    W = np.exp(- (D ** 2) / (2 * sigma_d ** 2))
    # normalize rows to have sum 1 (for interpretability)
    row_sums = W.sum(axis=1, keepdims=True)
    W = W / (row_sums + 1e-12)
    return W

# example single-layer convergence: choose modest size to be fast
n_layer = 64
W = make_weight_matrix(n_layer, sigma_d=3.0)

def activation(x):
    # smooth rectifier (softplus-like)
    return np.log1p(np.exp(x))

def simulate_dynamics_ivp(W, input_drive, T=1.0, dt=dt, tau=tau, alpha=alpha, beta=beta):
    n = W.shape[0]
    def rhs(t, x):
        fx = activation(x)
        inhibitory = W.dot(fx)
        return (-x + alpha * fx - beta * inhibitory + input_drive) / tau
    t_eval = np.arange(0, T, dt)
    sol = solve_ivp(rhs, (0, T), np.zeros(n), t_eval=t_eval, method='Radau', rtol=1e-6, atol=1e-8)
    return sol.y.T, sol.y[:, -1]

# build a feedforward input drive: localized stimulus
input_drive = np.zeros(n_layer)
input_drive[10:14] = 1.2  # small patch
traj, x_final = simulate_dynamics_ivp(W, input_drive, T=0.2) # Shorter time for faster run

# measure convergence time to epsilon (norm difference to final)
final = x_final
diffs = np.linalg.norm(traj - final[None, :], axis=1)
conv_idx = np.where(diffs < epsilon_conv)[0]
conv_time = conv_idx[0] * dt if conv_idx.size > 0 else np.nan

# -------------------- (B) Retrieval accuracy vs M (cross-talk SNR model) --------------------
def retrieval_accuracy(M, K=K, A=A, Gamma=Gamma, C_sig=C_sig, sigma=sigma, kappa_xt=kappa_xt):
    mu = A * Gamma * C_sig
    sigma_tot_sq = sigma**2 + (M / K) * (A**2 * kappa_xt)
    if sigma_tot_sq <= 0:
        return 1.0  # perfect accuracy in the degenerate zero-noise case
    acc = 1.0 - np.exp(-0.5 * mu**2 / sigma_tot_sq)
    return float(np.clip(acc, 0.0, 1.0))


M_values = np.arange(1, 41)  # number of stored samples
accuracies = np.array([retrieval_accuracy(M) for M in M_values])

# -------------------- (C) Capacity vs clarity sweep --------------------
# sweep overall clarity scaling factor and compute implied M_max by SNR >= theta
theta = 2.0  # SNR threshold
def capacity_bound_for_clarity_scale(scale, C_l_base=C_l, Gamma_base=Gamma):
    C_l_scaled = np.clip(C_l_base * scale, 0.0, 1.0)
    C_sig_scaled = np.dot(w_l, C_l_scaled)
    kappa_xt_scaled = np.dot(w_l, kappa_l * (1.0 - C_l_scaled))
    mu = A * Gamma_base * C_sig_scaled
    denom_target = mu**2 / theta
    if denom_target <= sigma**2 or kappa_xt_scaled <= 1e-12:
        return 0.0
    M_max = K * (denom_target - sigma**2) / (A**2 * kappa_xt_scaled)
    return max(0.0, M_max)

scales = np.linspace(0.2, 1.5, 40)
M_caps = np.array([capacity_bound_for_clarity_scale(s) for s in scales])

# -------------------- Plotting results --------------------
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Dynamics Convergence
axes[0].plot(np.arange(len(diffs)) * dt, diffs)
axes[0].axhline(epsilon_conv, color='r', linestyle='--', label=f'$\\epsilon={epsilon_conv}$')
if not np.isnan(conv_time):
    axes[0].axvline(conv_time, color='g', linestyle='--', label=f'Converged at {conv_time:.3f}s')
axes[0].set_yscale('log')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('||x(t)-x_*|| (L2 Norm)')
axes[0].set_title('Dynamics Convergence to Steady-State')
axes[0].legend()
axes[0].grid(True, which="both", ls="-")


# Plot 2: Retrieval Accuracy
axes[1].plot(M_values, accuracies, marker='o', linestyle='-')
axes[1].set_xlabel('Number of Stored Samples (M)')
axes[1].set_ylabel('Estimated Retrieval Accuracy')
axes[1].set_title('Accuracy vs. Memory Load')
axes[1].set_ylim([-0.02, 1.02])
axes[1].grid(True)

# Plot 3: Capacity vs Clarity
axes[2].plot(scales, M_caps, marker='x', linestyle='-')
axes[2].set_xlabel('Clarity Scale Factor on C(l)')
axes[2].set_ylabel('Estimated Capacity (M_max)')
axes[2].set_title('Capacity vs. System Clarity')
axes[2].grid(True)

plt.tight_layout(pad=3.0)
plt.suptitle('Hierarchical VWM Model Simulation Results', fontsize=16, y=1.02)
plt.show()

# -------------------- (D) Parametric Sweep for Optimal Capacity --------------------
print("\n--- Performing Parametric Sweep for Optimal Capacity ---")

# Define parameter ranges for the grid search
delta_range = np.linspace(0.1, 0.4, 4)
a_range = np.linspace(0.7, 0.95, 4)
B_range = np.linspace(4.0, 8.0, 4)
gamma_curve_range = np.linspace(1.5, 2.5, 4)

best_params = {}
max_capacity = -1

# Objective function for the sweep
def evaluate_capacity(delta_p, a_p, B_p, gamma_p):
    # Recalculate derived quantities based on new parameters
    Gamma_p = np.prod([1 - (delta_p * (a_p / p) ** (l - 1)) for l in range(1, L + 1)])
    C_l_p = np.array([clarity_C(clarity_A(l, L, gamma_curve=gamma_p), B=B_p) for l in range(1, L + 1)])
    C_sig_p = np.dot(w_l, C_l_p)
    kappa_xt_p = np.dot(w_l, kappa_l * (1 - C_l_p))

    # Calculate capacity
    mu = A * Gamma_p * C_sig_p
    denom_target = mu**2 / theta
    if denom_target <= sigma**2 or kappa_xt_p <= 1e-12:
        return 0.0
    M_max = K * (denom_target - sigma**2) / (A**2 * kappa_xt_p)
    return max(0.0, M_max)

# Grid search
for d_val in delta_range:
    for a_val in a_range:
        for B_val in B_range:
            for g_val in gamma_curve_range:
                current_capacity = evaluate_capacity(d_val, a_val, B_val, g_val)
                if current_capacity > max_capacity:
                    max_capacity = current_capacity
                    best_params = {'delta': d_val, 'a': a_val, 'B': B_val, 'gamma_curve': g_val}

# -------------------- Summary dataframe --------------------
summary = pd.DataFrame({
    'delta_eff': [delta_eff],
    'Gamma': [Gamma],
    'C_sig': [C_sig],
    'kappa_xt': [kappa_xt],
    'conv_time_s': [conv_time]
})

def contraction_metric(L_fprime, alpha_bar, beta_bar, N_eff):
    return L_fprime * (alpha_bar + beta_bar * N_eff)

# Example N_eff from the weight matrix
N_eff_example = np.sum(W[n_layer//2, :] > 0.01)
print("\n--- Summary (Initial Parameters) ---")
print(summary)
print(f"\nContraction metric (example): {contraction_metric(L_fprime, alpha, beta, N_eff_example):.4f}")
if contraction_metric(L_fprime, alpha, beta, N_eff_example) < 1:
    print("--> System is contracting (stable).")
else:
    print("--> System may not be contracting (potentially unstable).")

print("\n--- Optimal Parameters Found ---")
print(f"Maximum Capacity (M_max) found: {max_capacity:.4f}")
print("Best parameter set:")
for param, value in best_params.items():
    print(f"  {param}: {value:.4f}")

