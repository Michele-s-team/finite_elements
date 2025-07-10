"""
This script compares linear stability eigenvalues with nonlinear growth rates from explicit Euler evolution.

How to run:
    pip install numpy scipy matplotlib
    python linear_vs_nonlinear_evolution.py

What it does:
1. Defines finite-difference operators (D1, D2, D4) with Neumann BCs.
2. Builds the Jacobian J(rB) for each rB.
3. Finds the principal nonzero eigenvalue λ_lin and its eigenvector.
4. Uses that eigenvector as a small perturbation and integrates the full nonlinear PDE by explicit Euler to t_final.
5. Computes the nonlinear growth rate: log(||u(t_final)||/||u(0)||)/t_final.
6. Plots λ_lin(rB) against the nonlinear growth rate for comparison.
"""

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from numpy.linalg import norm

# Model parameters
DA, DB       = 1.0, 1.0
cA0, cB0     = 1.0, 1.0
kappa0       = 1.0
RA, nPts, L  = 1.0, 60, 100.0
h             = L/(nPts-1)
x             = np.linspace(0, L, nPts)

# Nonlinear coupling functions
def B2(a, b): return 16*np.pi/3*(a+b)**3

def B3(a, b, c):
    return (16*np.pi**2/9)*(
        b**3*c**3
      + 3*a*b**2*c**2*(b+c)
      + a**3*(b+c)**3
      + 3*a**3*b*c*(b**2+3*b*c+c**2)
    )

# Finite-difference with Neumann BC

def neumann_D1(n, h):
    D1 = sp.lil_matrix((n,n))
    for i in range(1, n-1):
        D1[i,i+1] = +0.5/h
        D1[i,i-1] = -0.5/h
    D1[0,0], D1[0,1]     = -1.0/h, 1.0/h
    D1[-1,-2], D1[-1,-1] = -1.0/h, 1.0/h
    return D1.tocsr()


def neumann_D2(n, h):
    main = -2*np.ones(n)
    off  =  np.ones(n-1)
    D2 = sp.diags([off, main, off], [-1,0,1], format='lil')
    # 2nd-order Neumann
    D2[0,:2]     = [-1, 1]
    D2[-1,-2:]  = [1, -1]
    return (D2/h**2).tocsr()

D1 = neumann_D1(nPts, h)
D2 = neumann_D2(nPts, h)
D4 = D2.dot(D2)

# Build Jacobian J(rB)
def build_J(rB):
    B2AA = B2(RA,RA); B2BB = B2(rB,rB); B2AB = B2(RA,rB)
    B3AAA= B3(RA,RA,RA); B3AAB= B3(RA,RA,rB)
    B3BBA= B3(rB,rB,RA); B3BBB= B3(rB,rB,rB)
    fAA = 1 + cA0*B2AA + cA0**2*B3AAA + cA0*cB0*B3AAB
    fBB = 1 + cB0*B2BB + cB0**2*B3BBB + cA0*cB0*B3BBA
    fAB = cA0*B2AB + cA0**2*B3AAB + cA0*cB0*B3BBA
    fBA = cB0*B2AB + cB0**2*B3BBA + cA0*cB0*B3AAB

    JAA = DA * fAA * D2 - DA * kappa0 * cA0 * D4
    JBB = DB * fBB * D2 - DB * kappa0 * cB0 * D4
    JAB = DA * fAB * D2 - DA * kappa0 * cA0 * D4
    JBA = DB * fBA * D2 - DB * kappa0 * cB0 * D4

    return sp.bmat([[JAA, JAB], [JBA, JBB]], format='csr')

def find_zero_crossings(x, y):
    roots = []
    for k in range(len(x)-1):
        y0, y1 = y[k], y[k+1]
        # exaktes zero?
        if y0 == 0:
            roots.append(x[k])
        # Vorzeichenwechsel?
        elif y0*y1 < 0:
            # linear interpolieren
            root = x[k] - y0*(x[k+1]-x[k])/(y1 - y0)
            roots.append(root)
    return roots
# Parameter range
rB_list = np.linspace(0.925, 1.075, 50)

J0     = build_J(rB_list[0]).toarray()
vals0, vecs0 = np.linalg.eig(J0)
re0    = np.real(vals0)
# Indices with real part close to 0
zero_idxs = np.where(np.isclose(re0, 0.0, atol=1e-8))[0]
# Eigenvectors of the zero modes
Z = vecs0[:, zero_idxs]
# Orthonormalize via QR – Q_zero columns form a basis for the nullspace
Q_zero, _ = np.linalg.qr(Z)

# Storage for the results
lambda_lin = np.zeros_like(rB_list)
growth_nl  = np.zeros_like(rB_list)

# Simulation parameters
dt      = 1e-3
t_final = 50.0
eps     = 1e-6

for i, rB in enumerate(rB_list):
    # 1) Linear eigen analysis
    J = build_J(rB).toarray()
    vals, vecs = np.linalg.eig(J)
    reals = np.real(vals)
    # exclude zero modes
    projs = np.abs(Q_zero.T @ vecs)
    # Maximum overlap of each eigenvector with any zero mode
    max_proj = np.max(projs, axis=0)
    # Mask: True for non-zero modes (overlap small enough)
    mask_nonzero = max_proj < 1e-3

    # 1b) Filter out zero modes, set their eigenvalues to -∞
    reals_nz = reals.copy()
    reals_nz[~mask_nonzero] = -np.inf

    # 1c) Index of the largest real-part eigenvalue among non-zero modes
    idx = np.argmax(reals_nz)
    lambda_lin[i] = reals_nz[idx]
    mode = vecs[:, idx]
    mode /= norm(mode)
    mode /= norm(mode)

    # 2) Nonlinear Euler evolution
    # initialize fields
    vA = mode[:nPts] * eps
    vB = mode[nPts:] * eps
    cA = cA0 + vA.copy()
    cB = cB0 + vB.copy()
    norm0 = norm(np.hstack([vA, vB]))
    #norm0 = norm(mode)
    n_steps = int(t_final/dt)
    capped = False
    for _ in range(n_steps):
        # spatial derivatives
        dA_dx  = D1.dot(cA);  dB_dx  = D1.dot(cB)
        d3A_dx = D1.dot(D1.dot(D1.dot(cA)))
        d3B_dx = D1.dot(D1.dot(D1.dot(cB)))
        # chemical potential gradients
        termA = B2(RA,RA)*dA_dx + B2(RA,rB)*dB_dx - kappa0*d3A_dx - kappa0*d3B_dx
        termB = B2(RA,rB)*dA_dx + B2(rB,rB)*dB_dx - kappa0*d3A_dx - kappa0*d3B_dx
        # third-order flux
        flux3A = (B3(RA,RA,RA)*cA**2*dA_dx
                + B3(RA,RA,rB)*(cA**2*dB_dx + cA*cB*dA_dx)
                + B3(rB,rB,RA)*(cA*cB*dB_dx))
        flux3B = (B3(rB,rB,rB)*cB**2*dB_dx
                + B3(rB,rB,RA)*(cB**2*dA_dx + cB*cA*dB_dx)
                + B3(RA,RA,rB)*(cB*cA*dA_dx))
        # fluxes and update
        JA = -DA*(dA_dx + cA*termA + flux3A)
        JB = -DB*(dB_dx + cB*termB + flux3B)
        JA[0]=JA[-1]=0; JB[0]=JB[-1]=0
        cA += dt * (-D1.dot(JA))
        cB += dt * (-D1.dot(JB))
        perturb_norm = norm(np.hstack([cA-cA0, cB-cB0]))
        if not np.all(np.isfinite(perturb_norm)) or perturb_norm > 1e10:
            perturb_norm = 1e10
            capped = True
            break
    # compute nonlinear growth rate
    normf = perturb_norm
    growth_nl[i] = np.log(normf/norm0)/t_final
roots_lin = find_zero_crossings(rB_list, lambda_lin)
roots_nl  = find_zero_crossings(rB_list, growth_nl)
# Plot comparison
plt.figure(figsize=(7,5))
plt.plot(rB_list, lambda_lin, '-o', label='Linear $\lambda_{pert}$')
plt.plot(rB_list, growth_nl, '-s', label='$\lambda_{euler}$')
plt.axhline(0, color='k', lw=1)
for i, x0 in enumerate(roots_lin):
    plt.axvline(x0, color='C0', linestyle='--',
                label='$λ_{pert}=0$' if i==0 else None)
for i, x0 in enumerate(roots_nl):
    plt.axvline(x0, color='C1', linestyle=':',
                label='$\lambda_{euler}$' if i==0 else None)
plt.xlabel('rB')
plt.ylabel('Growth rate')
plt.title('Linear vs Nonlinear Growth Rates')
plt.legend()
plt.tight_layout()
plt.savefig('solutions/linear_nonlinear_compare.png', dpi=300)
plt.show()
