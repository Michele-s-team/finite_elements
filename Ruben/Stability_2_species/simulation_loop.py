import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

kBT, cA0, cB0 = 1.0, 1.0, 2.0
DA, DB        = 1.0, 1.0
kappaAA = kappaAB = kappaBB = 1.0    
RA, nPts, L   = 1.0, 60, 100.0
x             = np.linspace(0, L, nPts)
h             = L/(nPts-1)

def B2(a,b): return 16*np.pi/3*(a+b)**3
def B3(a,b,c):
    return (16*np.pi**2/9)*(
       b**3*c**3
     + 3*a*b**2*c**2*(b+c)
     + a**3*(b+c)**3
     + 3*a**3*b*c*(b**2+3*b*c+c**2)
    )

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

D1 = neumann_D1(nPts, h)
D2 = neumann_D2(nPts, h)
D3 = D1.dot(D2)
rng = np.random.default_rng(12345)
K_num = 100000
Kmax = 1000000
eps = 1e-6
Ks_A = rng.integers(1, Kmax+1, size = K_num)
Ks_B = rng.integers(1, Kmax+1, size = K_num)
amps_A = rng.standard_normal(K_num)
amps_B = rng.standard_normal(K_num)
pertA = sum(amps_A[i] * np.cos(2*np.pi*Ks_A[i]*x/L) for i in range(K_num))
pertB = sum(amps_B[i] * np.cos(2*np.pi*Ks_B[i]*x/L) for i in range(K_num))
pertA /= np.linalg.norm(pertA)
pertB /= np.linalg.norm(pertB)
pertA = eps*pertA
pertB = eps*pertB
data = np.load('largest_eigenmodes.npz')
eigvecs = data['eigvecs']



def run_one(rB,index = 0, dt=5e-4, t_final=50):
    B2AA = B2(RA,RA)
    B2AB = B2(RA,rB)
    B2BB = B2(rB,rB)
    B3AAA = B3(RA,RA,RA)
    B3AAB = B3(RA,RA,rB)
    B3BBA = B3(rB,rB,RA)
    B3BBB = B3(rB,rB,rB)
    pert = np.zeros_like(x)
    v = eigvecs[index]
    vA = v[:nPts].real
    vB = v[nPts:].real
    vA *= 1e-9 / np.linalg.norm(vA)
    vB *= 1e-9 / np.linalg.norm(vB)

    cA = cA0 + vA
    cB = cB0 + vB

    n_steps = int(t_final/dt)
    ampsA = np.zeros(n_steps+1)
    ampsB = np.zeros(n_steps+1)

    for _ in range(n_steps):
        dcA_dx  = D1.dot(cA)
        dcB_dx  = D1.dot(cB)
        d3cA_dx = D1.dot(D1.dot(D1.dot(cA)))
        d3cB_dx = D1.dot(D1.dot(D1.dot(cB)))

        termA   = (B2AA*dcA_dx + B2AB*dcB_dx
                 - kappaAA*d3cA_dx - kappaAB*d3cB_dx)
        termB   = (B2AB*dcA_dx + B2BB*dcB_dx
                 - kappaAB*d3cA_dx - kappaBB*d3cB_dx)

        flux3A  = (B3AAA*(cA**2)*dcA_dx
                 + B3AAB*((cA**2)*dcB_dx + cA*cB*dcA_dx)
                 + B3BBA*(cA*cB)*dcB_dx)
        flux3B  = (B3BBB*(cB**2)*dcB_dx
                 + B3BBA*((cB**2)*dcA_dx + cB*cA*dcB_dx)
                 + B3AAB*(cB*cA)*dcA_dx)

        JA = -DA*(dcA_dx + cA*termA + flux3A)
        JB = -DB*(dcB_dx + cB*termB + flux3B)
        JA[0] = 0;  JA[-1] = 0
        JB[0] = 0;  JB[-1] = 0

        dcA_dt = -D1.dot(JA)
        dcB_dt = -D1.dot(JB)

        cA += dt*dcA_dt
        cB += dt*dcB_dt
        if not np.all(np.isfinite(cA)) or not np.all(np.isfinite(cB)):
            return 1e4, 1e4

    return np.linalg.norm(cA - cA0), np.linalg.norm(cB - cB0)

#rB_list    = np.linspace(0.8, 1.2, 20)
rB_list  = data['rB']
finalA     = np.zeros_like(rB_list)
finalB     = np.zeros_like(rB_list)

for i, rB in enumerate(rB_list):
    print(f"Simulating rB = {rB:.3f}…")
    a, b = run_one(rB)
    finalA[i], finalB[i] = a, b
plt.figure(figsize=(6,4))
plt.plot(rB_list, finalA, '-o', label='A final ||Δc||')
plt.plot(rB_list, finalB, '-s', label='B final ||Δc||')
plt.axvline(1.0, color='k', ls='--', label='rB=1.0')
plt.xlabel('rB')
plt.ylabel('final perturbation amplitude')
plt.yscale('log')
plt.legend()
plt.title('Nonlinear Stability Sweep')
plt.tight_layout()
plt.savefig('simulation_cB=2.png')