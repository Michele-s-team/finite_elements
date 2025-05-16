import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# ------------------------------
# 1) Model‐ und Numerikparameter
# ------------------------------
RA     = 0.1
kBT    = 1.0
cA0    = 1.0
cB0    = 1.0
DA     = 1.0
DB     = 1.0
kappa0 = 0.01

# Domain & Auflösung (λ_max ≈ 0.05)
L     = 5.0
nPts  = 1001
h     = L/(nPts-1)

# Szenarien für rB
rB_list = [0.1, 1.5, 2.5]

# -------------------------------------
# 2) Diskretisierung: D2 (Neumann‐Ränder)
# -------------------------------------
D2 = np.zeros((nPts, nPts))
for i in range(1, nPts-1):
    D2[i, i-1] =  1.0/h**2
    D2[i, i]   = -2.0/h**2
    D2[i, i+1] =  1.0/h**2
# Neumann BC
D2[0,   0], D2[0,   1] = -1.0/h**2,  1.0/h**2
D2[-1, -2], D2[-1, -1]=  1.0/h**2, -1.0/h**2

D4 = D2 @ D2

# ------------------------------
# 3) Virial & CH‐Koeffizienten
# ------------------------------
def B2(i, j):     return (16*np.pi/3)*(i+j)**3
def sigma(i, j):  return np.pi*(i+j)**2
def kappa(i, j):  return kappa0*(i+j)**2

# ------------------------------
# 4) Simulation via Matrix‐Exponential
# ------------------------------
x = np.linspace(0, L, nPts)
np.random.seed(0)
for rB in rB_list:
    # — Jacobi‐Blöcke
    B2AA, B2BB, B2AB = B2(RA,RA), B2(rB,rB), B2(RA,rB)
    sAA, sBB, sAB    = sigma(RA,RA), sigma(rB,rB), sigma(RA,rB)
    kAA, kBB, kAB    = kappa(RA,RA), kappa(rB,rB), kappa(RA,rB)

    fAA = kBT*(1/cA0 + B2AA/sAA)
    fBB = kBT*(1/cB0 + B2BB/sBB)
    fAB = kBT*(    B2AB/sAB)

    JAA = DA*cA0*( fAA*D2 - (kAA/sAA)*D4 )
    JBB = DB*cB0*( fBB*D2 - (kBB/sBB)*D4 )
    JAB = DA*cA0*( fAB*D2 - (kAB/sAB)*D4 )
    JBA = DB*cB0*( fAB*D2 - (kAB/sAB)*D4 )
    Jsys = np.block([[JAA, JAB],
                     [JBA, JBB]])

    # — spektrale Wachstumsrate und t_final wählen
    eigs = np.linalg.eigvals(Jsys)
    omega_max = np.max(eigs.real)
    if omega_max>0:
        t_final = np.log(10)/omega_max  # ~10-fach Wachstum
    else:
        t_final = 1.0

    # — Anfangszustand
    c_init = np.concatenate([
        cA0 + 1e-4*np.random.randn(nPts),
        cB0 + 1e-4*np.random.randn(nPts)
    ])

    # — exakte lineare Lösung
    c_end = expm(Jsys * t_final) @ c_init

    # — Plot: Anfang
    plt.figure(figsize=(6,3))
    plt.plot(x, c_init[:nPts], label='cA init')
    plt.plot(x, c_init[nPts:], label='cB init', alpha=0.8)
    plt.title(f'Initial (rB={rB})')
    plt.xlabel('x'); plt.ylabel('c'); plt.grid(True)
    plt.legend(); plt.tight_layout()
    plt.show()

    # — Plot: nach Wachstum
    plt.figure(figsize=(6,3))
    plt.plot(x, c_end[:nPts], label=f'cA @ t={t_final:.2f}')
    plt.plot(x, c_end[nPts:], label=f'cB @ t={t_final:.2f}', alpha=0.8)
    plt.title(f'After growth (rB={rB})')
    plt.xlabel('x'); plt.ylabel('c'); plt.grid(True)
    plt.legend(); plt.tight_layout()
    plt.show()

    print(f"rB={rB}: omega_max={omega_max:.1f}, t_final={t_final:.2f}\n")
