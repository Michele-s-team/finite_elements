"""
Description:
    This script assembles the tridiagonal matrix corresponding to the second‐derivative operator
    for the diffusion equation with Dirichlet boundary conditions, computes its eigenvalues, and
    evaluates the stability limit for an explicit time‐stepping scheme.

How to run:
    python diffusion_stability.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.linalg import eigvals

# Diffusion coefficient
diff_coef = 1.0

# Total length of the spatial domain
length = 1.0

# Number of interior grid points
n_pts = 50

# Grid spacing
h = length / (n_pts + 1)

# -----------------------------------------------------------------------------
# 1) Build the tridiagonal matrix A for the second derivative (d²/dx²) operator
#
#    A[i,i]   = -2 / h²
#    A[i,i±1] =  1 / h²
#
#    This corresponds to central finite differences with Dirichlet boundary
#    conditions (u=0 at both ends).
# -----------------------------------------------------------------------------
main_diag = -2.0 * np.ones(n_pts) / h**2
off_diag  =  1.0 * np.ones(n_pts - 1) / h**2

A = diags(
    diagonals=[off_diag, main_diag, off_diag],
    offsets=[-1, 0, 1],
    shape=(n_pts, n_pts),
    format='csr'
)

# -----------------------------------------------------------------------------
# 2) Define the diffusion operator L = D * A
# -----------------------------------------------------------------------------
L = diff_coef * A

# -----------------------------------------------------------------------------
# 3) Compute eigenvalues of L
#
#    For moderate sizes we convert to a dense array; for large n_pts you could
#    use scipy.sparse.linalg.eigs to compute only the largest eigenvalues.
# -----------------------------------------------------------------------------
L_dense   = L.toarray()
eigvals_L = eigvals(L_dense)

# All eigenvalues are real (symmetric matrix), so we take the real part and sort
eig_num = np.sort(eigvals_L.real)[::-1]

k = np.arange(1, n_pts+1)
eig_anal = -4.0/h**2 * np.sin(k * np.pi / (2*(n_pts+1)))**2
# 5) Plot numerical vs analytical
plt.figure()
plt.plot(k, eig_num, 'o', label='Numerical')
plt.plot(k, eig_anal, '-', label='Analytical')
plt.xlabel('Mode number k')
plt.ylabel('Eigenvalue $\lambda_k$')
#plt.title('Numerical vs Analytical Eigenvalues for D=1')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Simple_Diffusion_D=1.0.jpg")
plt.show()