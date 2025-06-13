import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt


kBT, cA0, cB0 = 1.0, 1.0, 1.0
DA, DB        = 1.0, 1.0
kappaAA = kappaAB = kappaBB = 1.0    
RA, nPts, L   = 1.0, 60, 100.0
x             = np.linspace(0, L, nPts)
h             = L/(nPts-1)


rB = 0.9
def B2(a,b): return 16*np.pi/3*(a+b)**3
def B3(a,b,c):
    return (16*np.pi**2/9)*(
       b**3*c**3
     + 3*a*b**2*c**2*(b+c)
     + a**3*(b+c)**3
     + 3*a**3*b*c*(b**2+3*b*c+c**2)
    )
B2AA, B2AB, B2BB = B2(RA,RA), B2(RA,rB), B2(rB,rB)
B3AAA = B3(RA,RA,RA)
B3AAB = B3(RA,RA, rB)
B3BBA = B3(rB, rB,RA)
B3BBB = B3(rB, rB,rB)

def neumann_D2(n,h):
    main = -2*np.ones(n)
    off  =  np.ones(n-1)
    D2   = sp.diags([off,main,off],[-1,0,1],format="lil")
    D2[0,0],   D2[0,1]   = -1, +1
    D2[-1,-2],D2[-1,-1] = +1, -1
    return (D2/h**2).tocsr()

def neumann_D1(n,h):
    D1 = sp.lil_matrix((n,n))
    for i in range(1,n-1):
        D1[i,i+1] =  +0.5/h
        D1[i,i-1] =  -0.5/h
    D1[0,0] = -1/h; D1[0,1] = +1/h
    D1[-1,-2] = -1/h; D1[-1,-1] = +1/h
    return D1.tocsr()

D1 = neumann_D1(nPts, h)
D2 = neumann_D2(nPts, h)
D3 = D1.dot(D2)

def compute_fluxes(cA, cB):
    dcA_dx  = D1.dot(cA)
    dcB_dx  = D1.dot(cB)
    d3cA_dx = D3.dot(cA)
    d3cB_dx = D3.dot(cB)
    termA = (B2AA*dcA_dx + B2AB*dcB_dx
           - kappaAA*d3cA_dx - kappaAB*d3cB_dx)
    termB = (B2AB*dcA_dx + B2BB*dcB_dx
           - kappaAB*d3cA_dx - kappaBB*d3cB_dx)
    flux3A = (B3AAA*(cA**2)*dcA_dx
            + B3AAB*((cA**2)*dcB_dx + cA*cB*dcA_dx)
            + B3BBA*(cA*cB)*dcB_dx)
    flux3B = (B3BBB*(cB**2)*dcB_dx
            + B3BBA*((cB**2)*dcA_dx + cB*cA*dcB_dx)
            + B3AAB*(cB*cA)*dcA_dx)
    JA = -DA*( dcA_dx + cA*termA + flux3A )
    JB = -DB*( dcB_dx + cB*termB + flux3B )

    return JA, JB

dt, t_final = 0.0001, 20.0
n_steps     = int(t_final/dt)

rng = np.random.default_rng(0)
cA = cA0 + 1e-6*(rng.random(nPts)-0.5)
cB = cB0 + 1e-6*(rng.random(nPts)-0.5)

ampsA = np.zeros(n_steps+1)
ampsB = np.zeros(n_steps+1)
ampsA[0] = np.linalg.norm(cA-cA0)
ampsB[0] = np.linalg.norm(cB-cB0)

for step in range(n_steps):
    JA, JB = compute_fluxes(cA, cB)
    dcA_dt = -D1.dot(JA)
    dcB_dt = -D1.dot(JB)
    cA += dt*dcA_dt
    cB += dt*dcB_dt
    ampsA[step+1] = np.linalg.norm(cA-cA0)
    ampsB[step+1] = np.linalg.norm(cB-cB0)

time = np.linspace(0, t_final, n_steps+1)
plt.semilogy(time, ampsA, label='A perturbation')
plt.semilogy(time, ampsB, label='B perturbation')
plt.xlabel('time'), plt.ylabel('||Δc||₂')
plt.legend()
plt.title(f'Nonlinear growth at rB={rB:.3f}')
plt.savefig("simulation_stability_vs_rB.jpg")

