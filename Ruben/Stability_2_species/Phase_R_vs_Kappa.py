import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

# — fixed parameters —
kBT, cA0, cB0 = 1.0, 1.0, 1.0
DA, DB, RA    = 1.0, 1.0, 1.0
nPts          = 50
L             = 100.0
h             = L/(nPts-1)

# — build the 1D Neumann Laplacian and bi‐Lap operator —
def neumann_D2(n, h):
    main = -2*np.ones(n)
    off  =  1*np.ones(n-1)
    D2   = sp.diags([off,main,off],[-1,0,1],format="lil")
    D2[0,0], D2[0,1]   = -1, +1
    D2[-1,-2], D2[-1,-1] = +1, -1
    return (D2/h**2).tocsr()

D2 = neumann_D2(nPts, h)
D4 = D2.dot(D2)

# — kernels —
def B2(rA, rB):
    return 4*np.pi/3*(rA+rB)**3

def B3(rA, rB, rC):
    return (16*np.pi**2/9) * (
        rB**3*rC**3
      + 3*rA*rB**2*rC**2*(rB+rC)
      + rA**3*(rB+rC)**3
      + 3*rA**3*rB*rC*(rB**2+3*rB*rC+rC**2)
    )

def build_J(rB, kappa0):
    B2AA, B2BB, B2AB = B2(RA,RA), B2(rB,rB), B2(RA,rB)
    B3AAA, B3AAB, B3BBA, B3BBB = (
        B3(RA,RA,RA), B3(RA,RA,rB),
        B3(rB,rB,RA), B3(rB,rB,rB)
    )
    fAA = 1 + B2AA + B3AAA + B3AAB
    fBB = 1 + B2BB + B3BBB + B3BBA
    fAB =     B2AB + B3AAB + B3BBA

    JAA =  DA*fAA * D2 - DA*cA0*kappa0 * D4
    JBB =  DB*fBB * D2 - DB*cB0*kappa0 * D4
    JAB =  DA*fAB * D2 - DA*cA0*kappa0 * D4
    JBA =  DB*fAB * D2 - DB*cB0*kappa0 * D4

    return sp.bmat([[JAA, JAB],
                    [JBA, JBB]], format="csr")


# — parameter grids —
rB_vals    = np.linspace(0.5, 1.5, 50)
kappa_vals = np.linspace(0.0, 5.0, 50)

# — compute max real eigenvalue for each (rB, kappa0) —
maxeig = np.zeros((kappa_vals.size, rB_vals.size))
for i, kappa0 in enumerate(kappa_vals):
    for j, rB in enumerate(rB_vals):
        J = build_J(rB, kappa0)
        w = np.linalg.eigvals(J.toarray())
        maxeig[i,j] = np.max(np.real(w))

# — threshold tiny noise to zero —
tol = 1e-8
mask = (np.abs(maxeig) < tol) & (maxeig > 0)
maxeig[mask] = 0.0

# — plot —
X, Y = np.meshgrid(rB_vals, kappa_vals)
plt.figure(figsize=(6,5))
pcm = plt.pcolormesh(X, Y, maxeig,
                     cmap='RdBu_r', shading='auto')
plt.colorbar(pcm, label='max Re [eig(J)]')

# crisp boundary where eigenvalue crosses tol
plt.contour(X, Y, maxeig, levels=[tol],
            colors='k', linewidths=2)

plt.xlabel(r'$r_B/r_A$')
plt.ylabel(r'$\kappa_0$')
plt.title('Stability Phase Diagram: r_B vs. κ₀')
plt.tight_layout()
plt.savefig('phase_R_vs_kappa.png', dpi=300)

