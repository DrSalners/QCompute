import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# Define constants
for N in [3]:
    # N = 5  # Number of spins
    J = 1.0  # Coupling strength
    Jx=-0.8
    Jy=-0.2
    Jz=0
    hbar = 1.0  # Planck's constant (set to 1 for simplicity)
    t_max = 3.14  # Maximum time for evolution
    t_steps = 200  # Number of time steps
    times = np.linspace(0, t_max, t_steps)

    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    id2 = np.eye(2, dtype=complex)

    # Helper function: Build N-spin operators
    def build_operator(op, site, N):
        """Constructs an operator acting on a specific site in an N-spin chain."""
        op_list = [id2] * N
        op_list[site] = op
        result = op_list[0]
        for i in range(1, N):
            result = np.kron(result, op_list[i])
        return result

    # Build the Heisenberg Hamiltonian
    H = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(N):
        SxSx = build_operator(sx, i, N) @ build_operator(sx, (i + 1) % N, N)
        SySy = build_operator(sy, i, N) @ build_operator(sy, (i + 1) % N, N)
        SzSz = build_operator(sz, i, N) @ build_operator(sz, (i + 1) % N, N)
        H += (Jx*SxSx + Jy*SySy + Jz*SzSz)

    # Define staggered magnetization operator
    M_staggered = sum((-1)**i * build_operator(sz, i, N) for i in range(N)) / N

    # Initial state: Néel state (alternating spins up and down)
    psi_neel = np.zeros(2**N, dtype=complex)
    neel_index = int(''.join(['1' if i % 2 == 0 else '0' for i in range(N)]), 2)
    psi_neel[neel_index] = 1
    print(psi_neel)

    # Time evolution
    M_staggered_values = []
    for t in times:
        U_t = expm(-1j * H * t / hbar)  # Time evolution operator
        psi_t = U_t @ psi_neel  # Time-evolved state
        M_t = np.real(np.conj(psi_t) @ (M_staggered @ psi_t))  # Staggered magnetization
        M_staggered_values.append(M_t)

    # Plot the time evolution of staggered magnetization
    plt.plot(times, M_staggered_values, label="Staggered Magnetization")
plt.xlabel("Time (t)")
plt.ylabel("$M_{\mathrm{staggered}}^z(t)$")
plt.title("Time Evolution of Staggered Magnetization (Néel State)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
