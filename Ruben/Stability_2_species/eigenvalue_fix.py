import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse import diags
import matplotlib.pyplot as plt


kBT = 1.0
cA0 = 1.0
cB0 = 2.0
DA = 1.0
DB = 1.0
kappa0 = 1.0
RA = 1.0 
nPts = 60
n = nPts
L = 100.0
x = np.linspace(0, L, nPts)
h = L/(nPts-1)

def B2(rA, rB):
  return 16*np.pi/3*(rA+rB)**3
def sigma(rA, rB):
  return 1
def B3(rA, rB, rC):

    return (16*np.pi**2/9) * (
        rB**3*rC**3
      + 3*rA*rB**2*rC**2*(rB+rC)
      + rA**3*(rB+rC)**3
      + 3*rA**3*rB*rC*(rB**2+3*rB*rC+rC**2)
    )


def stencil_2(i):
   return {-1:1.0/h**2, 0:-2.0/h**2, 1:1.0/h**2}
def stencil_4(i):
   return {-2:1.0/h**4, -1:-4.0/h**4, 0:6.0/h**4, 1:-4.0/h**4, 2:1.0/h**4}
def neumann_D2(n,h):
    main = -2*np.ones(n)
    off  =  np.ones(n-1)
    D2 = sp.diags([off, main, off], offsets=[-1, 0, +1],
                  shape=(n, n), format="lil")
    D2[0, 0] =  2
    D2[0, 1] = -5
    D2[0, 2] =  4
    D2[0, 3] = -1
    D2[-1, -1] =  2
    D2[-1, -2] = -5
    D2[-1, -3] =  4
    D2[-1, -4] = -1

    return (D2 / h**2).tocsr()
def neumann_D1(n,h):
    off = 0.5/h * np.ones(n-1)
    D1 = sp.diags([-off, off], offsets=[-1, +1],
                  shape=(n, n), format="lil")
    D1[0, 0]    = -1.0/h
    D1[0, 1]    = +1.0/h
    D1[-1, -2]  = -1.0/h
    D1[-1, -1]  = +1.0/h
    return D1.tocsr()
D1 = neumann_D1(nPts,h)
D2 = neumann_D2(nPts,h)
D3 = D1.dot(D2)
D4 = D1.dot(D3) 

def idx_A(i):    return i
def idx_B(i):  return n + i
def build_J(rB):
    B2AA, B2BB, B2AB, B2BA = B2(RA,RA), B2(rB,rB), B2(RA,rB), B2(rB, RA)
    B3AAA, B3AAB, B3BBA, B3BBB = B3(RA,RA,RA), B3(RA,RA,rB), B3(rB,rB,RA), B3(rB,rB,rB)
    fAA = 1.0 +  cA0*B2AA + cA0**2*B3AAA + B3AAB*cA0*cB0
    fBB = 1.0 + cB0*B2BB + cB0**2*B3BBB + cA0*cB0*B3BBA 
    fAB = cA0*B2AB + cA0**2*B3AAB + cA0*cB0*B3BBA
    fBA = B2BA*cB0 + cB0**2*B3BBA + cA0*cB0*B3AAB

    JAA = DA * fAA * D2
    JBB = DB * fBB * D2
    JAB = DA * fAB * D2
    JBA = DB * fBA * D2

    JAA = JAA - DA*cA0*kappa0 *D4
    JBB = JBB - DB*cB0*kappa0 *D4
    JAB = JAB - DA*cA0*kappa0 *D4
    JBA = JBA - DB*cB0*kappa0 *D4

    return sp.bmat([[JAA, JAB],
                    [JBA, JBB]], format="csr")


def build_J_with_BC(rB, kappa= kappa0):
    B2AA, B2BB, B2AB = B2(RA,RA), B2(rB,rB), B2(RA,rB)
    B3AAA, B3AAB, B3BBA, B3BBB = B3(RA,RA,RA), B3(RA,RA,rB), B3(rB,rB,RA), B3(rB,rB,rB)
    MAA =  DA*( D1 + cA0*(B2AA*D1 - kappa0*D3) )
    MAB =  DA*(        cA0*(B2AB*D1 - kappa0*D3) )
    MBA =  DB*(        cB0*(B2AB*D1 - kappa0*D3) )
    MBB =  DB*( D1 + cB0*(B2BB*D1 - kappa0*D3) )
    Msys = sp.bmat([[MAA, MAB],
                [MBA, MBB]], format="csr")
    M = Msys.tolil()
    iA0, iAL = idx_A(0),    idx_A(nPts-1)
    iB0, iBL = idx_B(0),    idx_B(nPts-1)
    M[iA0, :] = 0
    M[iAL, :] = 0
    M[iB0, :] = 0
    M[iBL, :] = 0
    Msys = M.tocsr()
    D1_big = sp.block_diag([D1, D1], format="csr")
    Jsys = D1_big.dot(Msys)
    return Jsys.tocsr()
   

rB_list = np.linspace(0.8, 1.2, 100)
rB0 = rB_list[0]
J0 = build_J(rB0).toarray()
vals0, _ = np.linalg.eig(J0.T)
idx_cont = np.argmax(np.real(vals0))
eigvals = np.zeros_like(rB_list)
eigvals_im = np.zeros_like(rB_list)
eigvals_min = np.zeros_like(rB_list)
eigvals_continous = np.zeros_like(rB_list)
eigvals_2 = np.zeros_like(rB_list)
eigvectors = []
for i, rB in enumerate(rB_list):
    Jsys    = build_J(rB)
    J_dense = Jsys.toarray()
    vals, _ = np.linalg.eig(J_dense.T)

    # 1) “unfiltered” leading real‐part
    eigvals[i] = np.max(np.real(vals))

    # 2) drop *all* near-zero modes in one go
    mask       = ~np.isclose(np.real(vals), 0, atol=1e-8)
    reals_full = np.real(vals).copy()
    reals_full[~mask] = -np.inf

    # 3) now this is the largest non-zero real part
    eigvals_2[i] = np.max(reals_full)
    eigvals_continous[i] = np.real(vals[idx_cont])
eigvals_comparison = np.zeros_like(rB_list)
for i, rB in enumerate(rB_list):
   J = build_J_with_BC(rB, kappa0)
   #val = np.max(np.real(spla.eigs(J, k=1, which='LR', return_eigenvectors=False)))
   eigvals_comparison[i] = np.max(np.real(np.linalg.eigvals(J.toarray().T)))
signs = np.sign(eigvals_min)
crossings = np.where((signs[:-1] <= 0) & (signs[1:] > 0))[0]
#print("kritisches rB* =",rB_list[crossings][0], rB_list[crossings][-1])
'''
J0    = build_J_with_BC(1.0).toarray()
vals0, vecs0 = np.linalg.eig(J0.T)
zero_idxs = np.where(np.isclose(vals0, 0, atol=1e-8))[0]
print("Found", len(zero_idxs), "at rB=1.0")
for zi in zero_idxs:
    vec = vecs0[:, zi]
    vec /= np.linalg.norm(vec)
    print("Eigenvectors:", vec)
nonzero_mask = np.ones_like(vals0, dtype=bool)
nonzero_mask[zero_idxs] = False
reals = np.real(vals0).copy()
reals[~nonzero_mask] = -np.inf
idx_max = np.argmax(reals)
vmax = vecs0[:, idx_max]
vmax /= np.linalg.norm(vmax)
print("Largest non-zero eigenvalue:", vals0[idx_max])
print("Corresponding eigenvector", vmax)

'''
    
plt.figure()
#plt.plot(rB_list, eigvals, label = 'With zeromodes')
#plt.plot(rB_list, eigvals_im)
#plt.plot(rB_list, eigvals_min, label = 'removed first 0 mode')
plt.plot(rB_list, eigvals_comparison, label='With BC?')
#plt.plot(rB_list, eigvals_2, label = 'removed 2 zeromodes')
#plt.plot(rB_list, eigvals_continous,'--', label='Continous eigenvector')
plt.axhline(0, color='k', lw=1)
plt.xlabel("r_B")
plt.legend()
plt.ylabel("max real eig")
plt.title("Instability Curve")
plt.savefig("eigvals_B3=0_2nd_smallest_bigger_range.png")
plt.savefig("testcB=2.png")

intersection_mask = np.isclose(eigvals_2, 0.0, atol = 1e-3)
intersection_indices = np.where(intersection_mask)[0]
'''
if intersection_indices.size >= 2:
   first_idx = intersection_indices[0]
   last_idx = intersection_indices[-1]
   print(f"Erstes null bei eigvals_2:  rB = {rB_list[first_idx]:.6f}  (Index {first_idx})")
   print(f"Letztes null bei eigvals_2:  rB = {rB_list[last_idx]:.6f}  (Index {last_idx})")

for k in (1.0, 1000.0):
   kappa0 = k
   J = build_J(0.95).toarray()
   JAA = J[:nPts, :nPts]
   print(f"κ={k:8.1f} → ‖JAA‖_max = {np.max(np.abs(JAA)):.6g}")
'''
rB = 1.0
Jbc = build_J_with_BC(rB).toarray()    # full (2n×2n) array
vals, vecs = np.linalg.eig(Jbc.T)      # eigen‐decomposition of the transpose

# find indices of the two zero modes (up to numerical tolerance)
tol = 1e-8
is_zero_real = np.abs(vals.real) < tol
is_zero_imag = np.abs(vals.imag) < tol
zero_idxs = np.where(is_zero_real & is_zero_imag)[0]
print("Zero‐mode indices:", zero_idxs)

for zi in zero_idxs:
    mode = vecs[:, zi]
    mode /= np.linalg.norm(mode)       # normalize
    print(f"\nEigenvector for zero‐mode #{zi}:")
    print(mode)