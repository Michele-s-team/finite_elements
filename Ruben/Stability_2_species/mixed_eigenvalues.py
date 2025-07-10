# run via python3 mixed_eigenvalues.py
"""What this script does:
- Builds the Jacobian matrix of a reaction-diffusion system for different values of rB.
- Computes eigenvalues and eigenvectors at each rB.
- Tracks three eigenmodes continuously by following eigenvectors with the highest overlap from initial eigenmodes at rB = 0.5:
    1. The mode with the largest eigenvalue.
    2. Two modes with eigenvalues closest to zero (zero modes).
- Plots the evolution of these three tracked eigenvalues alongside the global maximum eigenvalue.
"""

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

# Model parameters
DA, DB = 1.0, 1.5
cA0, cB0 = 1.0, 5.0
kappa0 = 1.0
RA = 1.0
nPts = 60
L = 100.0
h = L / (nPts - 1)

# Discrete grid (not used directly in current analysis)
x = np.linspace(0, L, nPts)

# Helper functions for nonlinear terms and Laplacian
def B2(rA, rB):
    return 16 * np.pi / 3 * (rA + rB)**3

def B3(rA, rB, rC):
    return (16 * np.pi**2 / 9) * (
        rB**3 * rC**3
      + 3 * rA * rB**2 * rC**2 * (rB + rC)
      + rA**3 * (rB + rC)**3
      + 3 * rA**3 * rB * rC * (rB**2 + 3*rB*rC + rC**2)
    )

def neumann_D2(n, h):
    main = -2 * np.ones(n)
    off = np.ones(n - 1)
    D2 = sp.diags([off, main, off], [-1, 0, 1], format="lil")
    D2[0, :2] = [-1, 1]
    D2[-1, -2:] = [1, -1]
    return (D2 / h**2).tocsr()

D2 = neumann_D2(nPts, h)
D4 = D2.dot(D2)

# Build full Jacobian for given rB value
def build_J(rB):
    B2AA, B2BB, B2AB = B2(RA, RA), B2(rB, rB), B2(RA, rB)
    B3AAA, B3AAB, B3BBA, B3BBB = B3(RA, RA, RA), B3(RA, RA, rB), B3(rB, rB, RA), B3(rB, rB, rB)
    fAA = 1 + B2AA + B3AAA + B3AAB
    fBB = 1 + B2BB + B3BBB + B3BBA
    fAB = B2AB + B3AAB + B3BBA

    JAA = DA * fAA * D2 - DA * cA0 * kappa0 * D4
    JBB = DB * fBB * D2 - DB * cB0 * kappa0 * D4
    JAB = DA * fAB * D2 - DA * cA0 * kappa0 * D4
    JBA = DB * fAB * D2 - DB * cB0 * kappa0 * D4

    return sp.bmat([[JAA, JAB], [JBA, JBB]], format="csr")

# Parameter sweep for rB values
rB_list = np.exp(np.linspace(np.log(0.1), np.log(1.25), 1000))

# Storage for eigenvalues
max_eigenvalues = np.zeros_like(rB_list)
tracked_main = np.zeros_like(rB_list)
tracked_zero1 = np.zeros_like(rB_list)
tracked_zero2 = np.zeros_like(rB_list)

# Initial eigen-decomposition at rB = 0.5
J0 = build_J(0.5).toarray()
vals0, vecs0 = np.linalg.eig(J0)
vals0 = np.real(vals0)

# Identify indices: largest mode and two zero modes
idx_main = np.argmax(vals0)
# Two indices with smallest absolute value (closest to zero)
absvals = np.abs(vals0)
sorted_zero_indices = np.argsort(absvals)
idx_zero1, idx_zero2 = sorted_zero_indices[0], sorted_zero_indices[1]

# Normalize and store initial eigenvectors
vec_main = vecs0[:, idx_main] / np.linalg.norm(vecs0[:, idx_main])
vec_zero1 = vecs0[:, idx_zero1] / np.linalg.norm(vecs0[:, idx_zero1])
vec_zero2 = vecs0[:, idx_zero2] / np.linalg.norm(vecs0[:, idx_zero2])

# Store initial eigenvalues
tracked_main[0] = vals0[idx_main]
tracked_zero1[0] = vals0[idx_zero1]
tracked_zero2[0] = vals0[idx_zero2]
max_eigenvalues[0] = np.max(vals0)

# Sweep over rB and track eigenmodes
for i, rB in enumerate(rB_list[1:], start=1):
    J = build_J(rB).toarray()
    vals, vecs = np.linalg.eig(J)
    vals = np.real(vals)
    # Normalize eigenvectors for overlap computation
    vecs /= np.linalg.norm(vecs, axis=0)

    # Track main mode by overlap
    ov_main = np.abs(vecs.T @ vec_main)
    idx_main = np.argmax(ov_main)
    vec_main = vecs[:, idx_main]
    tracked_main[i] = vals[idx_main]

    # Track first zero mode
    ov_z1 = np.abs(vecs.T @ vec_zero1)
    idx_zero1 = np.argmax(ov_z1)
    vec_zero1 = vecs[:, idx_zero1]
    tracked_zero1[i] = vals[idx_zero1]

    # Track second zero mode
    ov_z2 = np.abs(vecs.T @ vec_zero2)
    idx_zero2 = np.argmax(ov_z2)
    vec_zero2 = vecs[:, idx_zero2]
    tracked_zero2[i] = vals[idx_zero2]

    # Store maximum eigenvalue
    max_eigenvalues[i] = np.max(vals)

# Plotting results
plt.figure(figsize=(8, 5))
plt.plot(rB_list[1:], max_eigenvalues[1:], lw=2, label='Global max eigenvalue')
plt.plot(rB_list[1:], tracked_main[1:], '-.', lw=2, label='Tracked main mode')
plt.plot(rB_list[1:], tracked_zero1[1:], 'o', lw=1.5, markersize = 3,label='Tracked zero mode 1')
plt.plot(rB_list[1:], tracked_zero2[1:], 'o', lw=1.5, markersize = 3,label='Tracked zero mode 2')

plt.axhline(0, color='k', lw=1)
plt.xlabel(r'$r_B$')
plt.ylabel('Re(eigenvalue)')
plt.title('Turing Instability: Tracking Multiple Eigenmodes')
plt.legend()
plt.tight_layout()
plt.savefig('solutions/mixed_modes.jpg', dpi=300)
plt.show()
