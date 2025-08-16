import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

# Fixed parameters
class BaseParams:
    def __init__(self):
        self.N = 1000  # Retinal locations
        self.K = 50    # Memory slots
        self.A = 1.0   # Signal amplitude
        self.σ = 0.2   # Noise std dev
        self.θ = 5.0   # SNR threshold
        self.L = 4     # Number of layers

# Parameter search configuration
param_ranges = {
    'clarity_range': np.linspace(0.1, 0.9, 5),        # Average clarity (C_avg)
    'overlap_range': np.linspace(0.01, 0.3, 5),        # Average overlap (γ_avg)
    'feedback_range': np.linspace(0.01, 0.4, 5),       # Effective feedback gain (δ_eff)
    'crosstalk_range': np.linspace(0.1, 2.0, 5),       # Average cross-talk (κ_avg)
    'layer_range': [2, 3, 4, 5, 6]                    # Number of layers (L)
}

# Initialize results storage
results = {key: [] for key in ['M', 'C_avg', 'γ_avg', 'δ_eff', 'κ_avg', 'L']}

# =====================
# CAPACITY CALCULATION
# =====================
def compute_capacity(params):
    """Compute memory capacity with parameterized clarity and overlap"""
    # Generate layer parameters based on averages
    γ_l = np.linspace(params.γ_avg, params.γ_avg/2, params.L)  # Decreasing overlap
    C_l = np.linspace(params.C_avg/2, params.C_avg, params.L)  # Increasing clarity
    κ_l = np.full(params.L, params.κ_avg)  # Uniform cross-talk
    w_l = np.array([0.5**l for l in range(params.L, 0, -1)])  # Weights favoring higher layers
    w_l /= w_l.sum()  # Normalize

    # Calculate capacity factors
    Γ = np.prod([1 - γ for γ in γ_l])  # Aggregate distinguishability
    C_sig = np.sum(w_l * C_l)  # Clarity-weighted signal
    κ_xt = np.sum(w_l * κ_l * (1 - C_l))  # Cross-talk coefficient

    # SNR inequality
    numerator = (params.A**2 * Γ**2 * C_sig**2 / params.θ) - params.σ**2
    if numerator <= 0:
        return 0  # Below SNR threshold

    M = params.K * numerator / (params.A**2 * κ_xt)
    return min(M, params.K)  # Can't exceed available slots

# =====================
# PARAMETER SWEEP
# =====================
print("Starting parameter sweep...")
total_combinations = np.prod([len(v) for v in param_ranges.values()])
pbar = tqdm(total=total_combinations)

for C_avg in param_ranges['clarity_range']:
    for γ_avg in param_ranges['overlap_range']:
        for δ_eff in param_ranges['feedback_range']:
            for κ_avg in param_ranges['crosstalk_range']:
                for L in param_ranges['layer_range']:
                    # Create parameter set
                    params = BaseParams()
                    params.C_avg = C_avg
                    params.γ_avg = γ_avg
                    params.δ_eff = δ_eff
                    params.κ_avg = κ_avg
                    params.L = L

                    # Compute capacity
                    M = compute_capacity(params)

                    # Store results
                    results['M'].append(M)
                    results['C_avg'].append(C_avg)
                    results['γ_avg'].append(γ_avg)
                    results['δ_eff'].append(δ_eff)
                    results['κ_avg'].append(κ_avg)
                    results['L'].append(L)

                    pbar.update(1)

pbar.close()
print("Parameter sweep completed!")

# =====================
# VISUALIZATION
# =====================
# Convert to arrays for easier plotting
M = np.array(results['M'])
C_avg = np.array(results['C_avg'])
γ_avg = np.array(results['γ_avg'])
δ_eff = np.array(results['δ_eff'])
κ_avg = np.array(results['κ_avg'])
L = np.array(results['L'])

plt.figure(figsize=(16, 12))

# 1. Clarity vs. Capacity
plt.subplot(2, 2, 1)
for clarity in param_ranges['clarity_range']:
    mask = (C_avg == clarity)
    plt.scatter(γ_avg[mask], M[mask], label=f'C={clarity:.1f}')

plt.xlabel('Average Overlap (γ_avg)')
plt.ylabel('Resolvable Memories (M)')
plt.title('Impact of Clarity and Overlap on Memory Capacity')
plt.legend(title='Clarity')
plt.grid(True)

# 2. Feedback Gain vs. Cross-talk
plt.subplot(2, 2, 2)
sc = plt.scatter(δ_eff, κ_avg, c=M, cmap='viridis',
                 vmin=0, vmax=params.K, s=50)
plt.colorbar(sc, label='Resolvable Memories (M)')
plt.xlabel('Effective Feedback Gain (δ_eff)')
plt.ylabel('Average Cross-talk (κ_avg)')
plt.title('Feedback vs. Cross-talk Tradeoff')
plt.grid(True)

# 3. Hierarchical Depth Effect
plt.subplot(2, 2, 3)
for layers in param_ranges['layer_range']:
    mask = (L == layers)
    plt.scatter(C_avg[mask], M[mask], label=f'L={layers}')

plt.xlabel('Average Clarity (C_avg)')
plt.ylabel('Resolvable Memories (M)')
plt.title('Impact of Hierarchical Depth')
plt.legend(title='Layers')
plt.grid(True)

# 4. 3D: Clarity, Overlap, Capacity
ax = plt.subplot(2, 2, 4, projection='3d')
sc = ax.scatter(C_avg, γ_avg, M, c=M, cmap='viridis',
                vmin=0, vmax=params.K, s=20)
ax.set_xlabel('Clarity (C_avg)')
ax.set_ylabel('Overlap (γ_avg)')
ax.set_zlabel('Resolvable Memories (M)')
ax.set_title('3D Capacity Landscape')
plt.colorbar(sc, label='Resolvable Memories (M)')

plt.tight_layout()
plt.savefig('capacity_parameter_sweep.png', dpi=300)
plt.show()

# =====================
# STATISTICAL ANALYSIS
# =====================
# Filter valid results (M > 0)
valid_mask = M > 0
valid_M = M[valid_mask]
valid_C = C_avg[valid_mask]
valid_γ = γ_avg[valid_mask]
valid_δ = δ_eff[valid_mask]
valid_κ = κ_avg[valid_mask]
valid_L = L[valid_mask]

print("\n" + "="*50)
print("CAPACITY ANALYSIS RESULTS")
print("="*50)
print(f"Total simulations: {len(M)}")
print(f"Valid configurations (M > 0): {len(valid_M)} ({len(valid_M)/len(M):.1%})")
print(f"Average capacity (valid): {np.mean(valid_M):.1f} ± {np.std(valid_M):.1f}")
print(f"Maximum capacity: {np.max(valid_M):.1f}")
print(f"Minimum capacity (valid): {np.min(valid_M):.1f}")

# Regression coefficients
print("\nParameter impact on capacity (standardized coefficients):")
print(f"- Clarity: {np.corrcoef(valid_C, valid_M)[0,1]:.3f}")
print(f"- Overlap: {np.corrcoef(valid_γ, valid_M)[0,1]:.3f}")
print(f"- Feedback: {np.corrcoef(valid_δ, valid_M)[0,1]:.3f}")
print(f"- Cross-talk: {np.corrcoef(valid_κ, valid_M)[0,1]:.3f}")
print(f"- Layers: {np.corrcoef(valid_L, valid_M)[0,1]:.3f}")

# Optimal configuration analysis
opt_idx = np.argmax(valid_M)
print("\nOptimal configuration:")
print(f"- Capacity: {valid_M[opt_idx]:.1f}/{params.K} slots")
print(f"- Layers: {valid_L[opt_idx]}")
print(f"- Clarity: {valid_C[opt_idx]:.2f}")
print(f"- Overlap: {valid_γ[opt_idx]:.2f}")
print(f"- Feedback: {valid_δ[opt_idx]:.2f}")
print(f"- Cross-talk: {valid_κ[opt_idx]:.2f}")
