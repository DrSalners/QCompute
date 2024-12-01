import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

# Parameters
n = 3  # Number of spins
steps = 100  # Number of time steps
times = np.linspace(0, 2.5, steps)  # Time points

# Pauli matrices
I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

# Initial Neel state |010>
neel_state = np.zeros(2**n, dtype=complex)
neel_state[2] = 1  # |010> corresponds to index 2 in computational basis

# Helper function for tensor product
def tensor_product(ops):
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result

# Construct Heisenberg Hamiltonian for N=3 with periodic boundary conditions
H = np.zeros((2**n, 2**n), dtype=complex)
for i in range(n):
    X_pair = [I if j != i and j != (i + 1) % n else -0.8*X for j in range(n)]
    Y_pair = [I if j != i and j != (i + 1) % n else -0.2*Y for j in range(n)]
    Z_pair = [I if j != i and j != (i + 1) % n else 0*Z for j in range(n)]
    H += tensor_product(X_pair)
    H += tensor_product(Y_pair)
    H += tensor_product(Z_pair)

# Construct staggered magnetization operator
M_s = np.zeros((2**n, 2**n))
for i in range(n):
    factor = (-1)**i  # Alternating sign
    z_i = [I if j != i else Z for j in range(n)]
    M_s += factor * tensor_product(z_i)
M_s /= n

# Time evolution and staggered magnetization calculation
magnetizations = []
for t in times:
    U_t = expm(-1j * H * t)  # Time evolution operator
    psi_t = U_t @ neel_state  # Evolve the state
    ms_t = np.real(psi_t.conj().T @ M_s @ psi_t)  # Staggered magnetization
    magnetizations.append(ms_t)

# Plot the staggered magnetization over time
plt.figure(figsize=(8, 5))
plt.plot(times, magnetizations, label='Staggered Magnetization (N=3)')
plt.xlabel('Time')
plt.ylabel('Magnetization $M_s(t)$')
plt.title('Staggered Magnetization Evolution for 3-Qubit Heisenberg Chain')
plt.grid(True)
plt.legend()
plt.show()
