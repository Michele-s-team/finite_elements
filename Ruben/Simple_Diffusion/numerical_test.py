import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.linalg import eigh

# Parameters
D       = 1.0           # diffusion coefficient
L       = 1.0           # domain length
N       = 50            # number of interior grid points
h       = L/(N+1)       # grid spacing
t_final = 0.003    # total evolution time

# choose explicit‐Euler dt below stability limit Δt ≤ 2/|λ_max| ≈ 2/(4/h^2)
dt      = 0.01 * (2/(4/h**2))
n_steps = int(np.ceil(t_final/dt))

# Build the 1D Laplacian matrix A
main_diag = -2.0 * np.ones(N) / h**2
off_diag  =  1.0 * np.ones(N-1) / h**2
A = diags([off_diag, main_diag, off_diag], offsets=[-1,0,1], format='csr')
Lop = D * A

# Diagonalize A
lams, V = eigh(A.toarray())    # lams[k] ≤ 0, V[:,k] eigenmode

# Evolve each eigenmode with explicit Euler and measure amplitude decay
lam_euler = np.zeros(N)
for k in range(N):
    u = V[:,k].copy()
    u0 = u.copy()
    for _ in range(n_steps):
        u += dt * (Lop.dot(u))
    amp = np.linalg.norm(u) / np.linalg.norm(u0)
    lam_euler[k] = np.log(amp)/(D*t_final)
    

# Plot recovered vs true eigenvalues
k = np.arange(1, N+1)
plt.figure()
plt.plot(k, lam_euler[::-1], 'o', label=r'$\hat\lambda_k$ from Euler')
plt.plot(k, lams[::-1],          '-', label=r'$\lambda_k$ from Pertubation Theory')
plt.xlabel('mode number $k$')
plt.ylabel(r'eigenvalue $\lambda_k$')
#plt.title(r'Recovered $\lambda_k = \frac{\ln(\|u(t)\|/\|u(0)\|)}{D\,t}$')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("numerical_comparison.jpg")
plt.show()

