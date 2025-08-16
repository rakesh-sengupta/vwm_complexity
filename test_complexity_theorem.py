import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

np.random.seed(0)
# =====================
# MODEL PARAMETERS (from revised assumptions)
# =====================
class VWMParameters:
    def __init__(self):
        # Neural dynamics parameters
        self.τ = 1.0  # Time constant
        self.L_f = 0.8  # Lipschitz constant for f
        self.L_f_prime = 0.9  # Max slope of f
        self.α_bar = 0.5  # Max excitation gain
        self.β_bar = 0.1  # Max inhibitory gain
        self.N_eff = 3.0  # Effective neighborhood size

        # Hierarchy parameters
        self.L = 4  # Number of layers
        self.k = 2.5  # Pyramidal reduction factor
        self.p = 1.8  # Lattice coarsening factor
        self.r = self.k * self.p  # Effective reduction factor

        # Structural parameters
        self.δ = 0.3  # Overlap coefficient
        self.a = 0.7  # Distance attenuation
        self.η_bar = 0.2  # Time-averaged excitation bound
        self.δ_eff = self.δ * self.a / self.p  # Effective feedback gain

        # Capacity parameters
        self.γ = [0.1, 0.08, 0.05, 0.03]  # Layer overlaps
        self.C = [0.2, 0.4, 0.7, 0.9]  # Clarity values
        self.κ = [0.5, 0.4, 0.3, 0.2]  # Cross-talk kernels
        self.w = [0.1, 0.2, 0.3, 0.4]  # Projection weights
        self.A = 1.0  # Signal amplitude
        self.σ = 0.1  # Noise std dev
        self.θ = 5.0  # SNR threshold

        # Task parameters
        self.η = [1.2, 1.1, 1.0, 0.9]  # Neighborhood factors
        self.ψ = [0.6, 0.5, 0.4, 0.3]  # Effective interference
        self.κ0 = 0.1  # Reference scaling constant

        # Validation flags
        self.structural_stable = 2 * self.k * self.δ_eff < 1
        self.dynamic_stable = self.L_f_prime * (self.α_bar + self.β_bar * self.N_eff) < 1

params = VWMParameters()

# =====================
# THEOREM 1: STRUCTURAL COMPLEXITY
# =====================
def compute_neuron_counts(N, params):
    """Compute neuron counts per layer using the recurrence relation"""
    n = np.zeros(params.L)
    n[0] = N * (1 + params.η_bar)  # Layer 1

    for l in range(1, params.L):
        n[l] = N / (params.k ** l)
        for m in range(l):
            n[l] += params.δ * (params.a / params.p) ** (l - m) * n[m]
    return n

# Test structural complexity
N_values = np.array([100, 200, 400, 800, 1600])
total_neurons = []

for N in N_values:
    n_layers = compute_neuron_counts(N, params)
    total_neurons.append(np.sum(n_layers))

# =====================
# THEOREM 2: DYNAMIC CONVERGENCE
# =====================
def f(x):
    """Activation function (sigmoid)"""
    return 1 / (1 + np.exp(-x))

def f_prime(x):
    """Derivative of activation function"""
    fx = f(x)
    return fx * (1 - fx)

def neural_dynamics(t, x, W, I, alpha, beta):
    """Cohen-Grossberg dynamics for a neural layer"""
    fx = f(x)
    dxdt = (-x + alpha * fx - beta * W @ fx + I) / params.τ
    return dxdt

def simulate_layer_dynamics(params, condition_met):
    """Simulate layer dynamics under different stability conditions"""
    n_neurons = 20
    t_span = (0, 10)
    t_eval = np.linspace(*t_span, 100)

    # Create distance matrix and kernel
    positions = np.linspace(0, 1, n_neurons)
    D = np.abs(positions[:, None] - positions[None, :])
    W = np.exp(-5 * D)  # Distance attenuation kernel

    # Initial state and input
    x0 = np.random.randn(n_neurons)
    I = 0.5 * np.sin(2 * np.pi * positions)  # Spatial input pattern

    # Set gains based on stability condition
    if condition_met:
        alpha = 0.4 * params.α_bar
        beta = 0.3 * params.β_bar
    else:
        alpha = 1.2 * params.α_bar
        beta = 1.1 * params.β_bar

    # Solve ODE
    sol = solve_ivp(neural_dynamics, t_span, x0, args=(W, I, alpha, beta),
                t_eval=t_eval, method='Radau', rtol=1e-6, atol=1e-8)

    return sol.t, sol.y

# =====================
# THEOREM 3: MEMORY CAPACITY
# =====================
def compute_capacity(params, N, K, m_bits=6):
    """Compute memory capacity using SNR inequality (returns float >=0).
    Does not silently cap at K; optional upper cap is K * 2**m_bits if desired.
    """
    # aggregate distinguishability
    Gamma = np.prod([1.0 - g for g in params.γ])
    # clarity-weighted signal
    C_sig = sum(w * C for w, C in zip(params.w, params.C))
    # clarity-modulated cross-talk
    kappa_xt = sum(w * kappa * (1.0 - C) for w, kappa, C in zip(params.w, params.κ, params.C))

    # avoid division by zero
    if kappa_xt <= 0:
        # no cross-talk -> capacity limited only by noise and desired SNR
        denom = params.A**2  # but M appears multiplied by kappa_xt; if kappa_xt == 0, M->infty in model
        # practically, saturate to K * 2^m_bits or return a very large number
        return float('inf')  # or return K * (2**m_bits)

    numerator = (params.A**2 * Gamma**2 * C_sig**2 / params.θ) - params.σ**2
    if numerator <= 0:
        return 0.0

    M = K * numerator / (params.A**2 * kappa_xt)
    # optional: cap at feasible maximum K * 2^m_bits
    max_possible = K * (2 ** m_bits)
    M = max(0.0, M)
    M = min(M, max_possible)
    return M


# =====================
# THEOREM 4: TASK-DRIVEN RETRIEVAL
# =====================
def retrieval_cost(T, S, params):
    """Compute retrieval cost for a task"""
    cost = 0
    for l in range(params.L):
        S_l = S / (params.r ** l)  # Features at layer l
        log_term = np.log(1 + params.ψ[l] / params.κ0)
        cost += T * S_l * params.η[l] * log_term
    return cost

# =====================
# SIMULATION AND VISUALIZATION
# =====================
plt.figure(figsize=(14, 10))

# 1. Structural Complexity (Theorem 1)
plt.subplot(2, 2, 1)
plt.plot(N_values, total_neurons, 'o-', label='Total Neurons')
plt.plot(N_values, 15 * N_values, 'r--', label='O(N) Reference')
plt.xlabel('Retinal Input Size (N)')
plt.ylabel('Total Neurons')
plt.title(f'(a)Structural Complexity (δ_eff={params.δ_eff:.3f}, Stable={params.structural_stable})')
plt.legend()
plt.grid(True)

# 2. Dynamic Convergence (Theorem 2)
plt.subplot(2, 2, 2)
# Stable condition
t, y = simulate_layer_dynamics(params, condition_met=params.dynamic_stable)
for i in range(5):  # Plot first 5 neurons
    plt.plot(t, y[i], 'b-', alpha=0.6, label='Stable' if i==0 else None)

# Unstable condition
t, y = simulate_layer_dynamics(params, condition_met=not params.dynamic_stable)
for i in range(5):  # Plot first 5 neurons
    plt.plot(t, y[i], 'r--', alpha=0.6, label='Unstable' if i==0 else None)

plt.xlabel('Time')
plt.ylabel('Activity')
plt.title(f'(b)Dynamic Convergence (Stable={params.dynamic_stable})')
plt.legend()
plt.grid(True)

# 3. Memory Capacity (Theorem 3)
plt.subplot(2, 2, 3)
K_values = np.arange(10, 101, 10)
M_values = [compute_capacity(params, N=1000, K=K) for K in K_values]

plt.plot(K_values, M_values, 's-')
plt.plot(K_values, K_values, 'k--', label='Theoretical Maximum')
plt.xlabel('Available Slots (K)')
plt.ylabel('Resolvable Memories (M)')
plt.title('(c)Memory Capacity Scaling')
plt.grid(True)
plt.legend()

# 4. Task-Driven Retrieval (Theorem 4)
plt.subplot(2, 2, 4)
S_values = np.array([10, 20, 50, 100])
T = 5
costs = [retrieval_cost(T, S, params) for S in S_values]

plt.plot(S_values, costs, 'o-', label='Actual Cost')
plt.plot(S_values, 0.5 * T * S_values, 'r--', label='O(T·S) Reference')
plt.xlabel('Features per Event (S)')
plt.ylabel('Retrieval Cost')
plt.title(f'(d)Task Retrieval Complexity (T={T})')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('vwm_model_validation.png', dpi=300)
plt.show()

# =====================
# PRINT KEY RESULTS
# =====================
print("="*50)
print("MODEL VALIDATION RESULTS")
print("="*50)
print(f"Structural Stability (2kδ_eff < 1): {2*params.k*params.δ_eff:.3f} < 1? {params.structural_stable}")
print(f"Dynamic Stability Condition: {params.L_f_prime*(params.α_bar + params.β_bar*params.N_eff):.3f} < 1? {params.dynamic_stable}")

# Capacity for a specific configuration
K = 50
M = compute_capacity(params, N=1000, K=K)
print(f"\nMemory Capacity (K={K} slots): M ≤ {M:.1f} resolvable memories")

# Retrieval cost
T, S = 5, 50
cost = retrieval_cost(T, S, params)
print(f"Retrieval Cost (T={T} events, S={S} features): C = {cost:.1f} operations")

# Verify O(T·S) scaling
cost_ratio = cost / (T * S)
print(f"Retrieval Cost Scaling: C/(T·S) = {cost_ratio:.3f} (should approach constant)")
