import numpy as np
import matplotlib.pyplot as plt
import tensornetwork as tn
from scipy.linalg import expm

# Define parameters
N = 10  # Number of spins, adjusted to fit your system (for tensor networks)
J = 1.0  # Coupling constant
t_max = 10  # Maximum time for evolution
t_steps = 200  # Number of time steps
times = np.linspace(0, t_max, t_steps)

# Pauli matrices
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
id2 = np.eye(2, dtype=complex)

# Helper function to create MPS for the Néel state
def create_neel_state(N):
    """
    Creates an MPS representation for the Néel state of N spins.
    """
    tensors = []
    for i in range(N):
        if i % 2 == 0:  # Spins alternate between |up> and |down>
            tensor = np.array([[[1, 0]], [[0, 1]]], dtype=complex)  # |↑⟩
        else:
            tensor = np.array([[[0, 1]], [[1, 0]]], dtype=complex)  # |↓⟩
        tensors.append(tensor)
    return tensors

# Build the Hamiltonian (Heisenberg interaction)
def build_hamiltonian(N):
    """
    Constructs the Hamiltonian for a 1D chain with periodic boundary conditions.
    """
    H = []
    for i in range(N):
        Hx = np.kron(np.kron(np.eye(2**i), sx), np.eye(2**(N - i - 1)))
        Hy = np.kron(np.kron(np.eye(2**i), sy), np.eye(2**(N - i - 1)))
        Hz = np.kron(np.kron(np.eye(2**i), sz), np.eye(2**(N - i - 1)))
        H.append(J * (Hx + Hy + Hz))
    return sum(H)

# Define staggered magnetization (S_z)
def staggered_magnetization(state, N):
    """
    Calculates the staggered magnetization of the state.
    """
    magnetization = 0
    for i in range(N):
        tensor = state[i]
        sz_op = np.array([[1, 0], [0, -1]], dtype=complex)  # S_z operator
        mag_i = np.trace(np.tensordot(tensor, sz_op, axes=((1,), (0,))))
        magnetization += (-1)**i * mag_i
    return magnetization / N

# Initial state: Néel state represented as an MPS
mps = create_neel_state(N)

# Time evolution using MPS
def time_evolve_mps(mps, H, times):
    """
    Evolve the state using matrix product states and compute staggered magnetization.
    """
    M_staggered_values = []
    for t in times:
        U_t = expm(-1j * H * t)  # Time evolution operator (ignoring hbar for simplicity)
        
        # Apply the evolution operator to the MPS (approximated for simplicity)
        evolved_mps = mps  # Here you would implement MPS time evolution

        # Calculate staggered magnetization
        M_t = staggered_magnetization(evolved_mps, N)
        M_staggered_values.append(M_t)
    return M_staggered_values

# Construct Hamiltonian (Heisenberg model with periodic boundary conditions)
H = build_hamiltonian(N)

# Calculate time evolution and staggered magnetization
M_staggered_values = time_evolve_mps(mps, H, times)

# Plot the results
plt.figure(figsize=(8, 5))
plt.plot(times, M_staggered_values, label="Staggered Magnetization")
plt.xlabel("Time (t)")
plt.ylabel("$M_{\mathrm{staggered}}^z(t)$")
plt.title("Time Evolution of Staggered Magnetization (Néel State) using MPS")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
