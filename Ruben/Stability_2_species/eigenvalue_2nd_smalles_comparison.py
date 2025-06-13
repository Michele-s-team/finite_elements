import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse import diags
import matplotlib.pyplot as plt


kBT = 1.0
cA0 = 1.0
cB0 = 5.0
DA = 1.0
DB = 1.5
kappa0 = 1.0  
#kappaAA = 0
#kappaAB = 0
#kappaBB = 0 
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
    off  =  1*np.ones(n-1)
    D2   = sp.diags([off,main,off],[-1,0,1],format="lil")
    D2[0,0],   D2[0,1]   = -1,  +1
    D2[-1,-2],D2[-1,-1] = +1,  -1
    return (D2/h**2).tocsr()
D2 = neumann_D2(nPts,h)
D4 = D2.dot(D2) 

def idx_A(i):    return i
def idx_B(i):  return n + i
def build_J(rB):
    B2AA, B2BB, B2AB = B2(RA,RA), B2(rB,rB), B2(RA,rB)
    B3AAA, B3AAB, B3BBA, B3BBB = B3(RA,RA,RA), B3(RA,RA,rB), B3(rB,rB,RA), B3(rB,rB,rB)
    fAA = 1 +  B2AA + B3AAA + B3AAB
    fBB = 1 + B2BB + B3BBB + B3BBA 
    fAB = B2AB + B3AAB + B3BBA

    JAA = DA * fAA * D2
    JBB = DB * fBB * D2
    JAB = DA * fAB * D2
    JBA = DB * fAB * D2

    if kappa0!=0:
        JAA = JAA - DA*cA0*kappa0*D4
        JBB = JBB - DB*cB0*kappa0*D4
        JAB = JAB - DA*cA0*kappa0*D4
        JBA = JBA - DB*cB0*kappa0*D4

    return sp.bmat([[JAA, JAB],
                    [JBA, JBB]], format="csr")

'''
def build_M(rB):
    fAA = kBT*(1/cA0 + B2(RA,RA))
    fBB = kBT*(1/cB0 + B2(rB,rB))
    fAB = kBT*(       B2(RA,rB))
    B2AA = B2(RA,RA)
    B2AB = B2(RA,rB)
    B2BA = B2(rB,RA)
    B2BB = B2(rB,rB)
    M = sp.lil_matrix((2*n, 2*n))
    B3AAA = B3(RA,RA,RA)
    B3AAB = B3(RA,RA,rB)
    B3BBA = B3(rB,rB,RA)
    B3BBB = B3(rB,rB,rB)
    for i in range(n):
        C1 = DA*(1 + B2AA + B3AAA + B3AAB)
        C2 = DA*(B2AB + B3BBA+ B3AAB)
        C3 = DB*(1 + B2BB + B3BBA + B3BBB)
        C4 = DB*(B2BA + B3BBA + B3AAB)
        if i == 0:
            M[idx_A(i), idx_A(i+2)] += C1/h**2
            M[idx_A(i), idx_A(i+1)] += -2*C1/h**2
            M[idx_A(i), idx_A(i)]   += C1/h**2
            M[idx_A(i), idx_B(i+2)] += C2/h**2
            M[idx_A(i), idx_B(i+1)] += -2*C2/h**2
            M[idx_A(i), idx_B(i)]   += C2/h**2
            M[idx_B(i), idx_B(i+2)] += C3/h**2
            M[idx_B(i), idx_B(i+1)] += -2*C3/h**2
            M[idx_B(i), idx_B(i)]   += C3/h**2
            M[idx_B(i), idx_A(i+2)] += C4/h**2
            M[idx_B(i), idx_A(i+1)] += -2*C4/h**2
            M[idx_B(i), idx_A(i)]   += C4/h**2

        elif i == n-1:
            M[idx_A(i), idx_A(i-2)] += C1/h**2
            M[idx_A(i), idx_A(i-1)] += -2*C1/h**2
            M[idx_A(i), idx_A(i)]   += C1/h**2
            M[idx_A(i), idx_B(i-2)] += C2/h**2
            M[idx_A(i), idx_B(i-1)] += -2*C2/h**2
            M[idx_A(i), idx_B(i)]   += C2/h**2
            M[idx_B(i), idx_B(i-2)] += C3/h**2
            M[idx_B(i), idx_B(i-1)] += -2*C3/h**2
            M[idx_B(i), idx_B(i)]   += C3/h**2
            M[idx_B(i), idx_A(i-2)] += C4/h**2
            M[idx_B(i), idx_A(i-1)] += -2*C4/h**2
            M[idx_B(i), idx_A(i)]   += C4/h**2

        else:
            M[idx_A(i), idx_A(i-1)] += DA*(1 + B2AA + B3AAA + B3AAB)/h**2
            M[idx_A(i), idx_A(i)]    += -DA*(2 + 2*B2AA + 2*B3AAA + 2*B3AAB)/h**2
            M[idx_A(i), idx_A(i+1)] += DA*(1+ B2AA + B3AAA + B3AAB)/h**2
            M[idx_A(i), idx_B(i-1)] += DA*(B2AB + B3BBA+ B3AAB)/h**2
            M[idx_A(i), idx_B(i)]    += -DA*(2*B2AB + 2*B3BBA + 2*B3AAB)/h**2
            M[idx_A(i), idx_B(i+1)] += DA*(B2AB + B3BBA + B3AAB)/h**2
            M[idx_B(i), idx_B(i-1)] += DB*(1 + B2BB + B3BBA + B3BBB)/h**2
            M[idx_B(i), idx_B(i)]    += -DB*(2 + 2*B2BB + 2*B3BBA + 2*B3BBB)/h**2
            M[idx_B(i), idx_B(i+1)] += DB*(1 + B2BB + B3BBA + B3BBB)/h**2
            M[idx_B(i), idx_A(i-1)] += DB*(B2BA + B3BBA + B3AAB)/h**2
            M[idx_B(i), idx_A(i)]    += -DB*(2*B2BA + 2*B3BBA + 2*B3AAB)/h**2
            M[idx_B(i), idx_A(i+1)] += DB*(B2BA + B3BBA + B3AAB)/h**2
    
    for i in range(n):
        C1 = DA*kappaAA
        C2 = DA*kappaAB
        C3 = DB*kappaBB
        C4 = DB*kappaAB
        if i == 0 or i == 1:
            M[idx_A(i), idx_A(i+4)] += -C1/h**4
            M[idx_A(i), idx_A(i+3)] += 4*C1/h**4
            M[idx_A(i), idx_A(i+2)] += -6*C1/h**4
            M[idx_A(i), idx_A(i+1)] += 4*C1/h**4
            M[idx_A(i), idx_A(i)]   += -C1/h**4
            M[idx_A(i), idx_B(i+4)] += -C2/h**4
            M[idx_A(i), idx_B(i+3)] += 4*C2/h**4
            M[idx_A(i), idx_B(i+2)] += -6*C2/h**4
            M[idx_A(i), idx_B(i+1)] += 4*C2/h**4
            M[idx_A(i), idx_B(i)]   += -C2/h**4
            M[idx_B(i), idx_B(i+4)] += -C3/h**4
            M[idx_B(i), idx_B(i+3)] += 4*C3/h**4
            M[idx_B(i), idx_B(i+2)] += -6*C3/h**4
            M[idx_B(i), idx_B(i+1)] += 4*C3/h**4
            M[idx_B(i), idx_B(i)]   += -C3/h**4
            M[idx_B(i), idx_A(i+4)] += -C4/h**4
            M[idx_B(i), idx_A(i+3)] += 4*C4/h**4
            M[idx_B(i), idx_A(i+2)] += -6*C4/h**4
            M[idx_B(i), idx_A(i+1)] += 4*C4/h**4
            M[idx_B(i), idx_A(i)]   += -C4/h**4
        elif i == n-2 or i == n-1:
            M[idx_A(i), idx_A(i-4)] = -C1/h**4
            M[idx_A(i), idx_A(i-3)] += 4*C1/h**4
            M[idx_A(i), idx_A(i-2)] += -6*C1/h**4
            M[idx_A(i), idx_A(i-1)] += 4*C1/h**4
            M[idx_A(i), idx_A(i)]   += -C1/h**4
            M[idx_A(i), idx_B(i-4)] += -C2/h**4
            M[idx_A(i), idx_B(i-3)] += 4*C2/h**4
            M[idx_A(i), idx_B(i-2)] += -6*C2/h**4
            M[idx_A(i), idx_B(i-1)] += 4*C2/h**4
            M[idx_A(i), idx_B(i)]   += -C2/h**4
            M[idx_B(i), idx_B(i-4)] += -C3/h**4
            M[idx_B(i), idx_B(i-3)] += 4*C3/h**4
            M[idx_B(i), idx_B(i-2)] += -6*C3/h**4
            M[idx_B(i), idx_B(i-1)] += 4*C3/h**4
            M[idx_B(i), idx_B(i)]   += -C3/h**4
            M[idx_B(i), idx_A(i-4)] += -C4/h**4
            M[idx_B(i), idx_A(i-3)] += 4*C4/h**4
            M[idx_B(i), idx_A(i-2)] += -6*C4/h**4
            M[idx_B(i), idx_A(i-1)] += 4*C4/h**4
            M[idx_B(i), idx_A(i)]   += -C4/h**4
        else:

            M[idx_A(i), idx_A(i-2)] += -DA * kappaAA / h**4
            M[idx_A(i), idx_A(i-1)] +=  4*DA * kappaAA / h**4
            M[idx_A(i), idx_A(i)  ] += -6*DA * kappaAA / h**4
            M[idx_A(i), idx_A(i+1)] +=  4*DA * kappaAA / h**4
            M[idx_A(i), idx_A(i+2)] += -DA * kappaAA / h**4

            M[idx_A(i), idx_B(i-2)] += -DA * kappaAB / h**4
            M[idx_A(i), idx_B(i-1)] +=  4*DA * kappaAB / h**4
            M[idx_A(i), idx_B(i)  ] += -6*DA * kappaAB / h**4
            M[idx_A(i), idx_B(i+1)] +=  4*DA * kappaAB / h**4
            M[idx_A(i), idx_B(i+2)] += -DA * kappaAB / h**4

            M[idx_B(i), idx_B(i-2)] += -DB * kappaBB / h**4
            M[idx_B(i), idx_B(i-1)] +=  4*DB * kappaBB / h**4
            M[idx_B(i), idx_B(i)  ] += -6*DB * kappaBB / h**4
            M[idx_B(i), idx_B(i+1)] +=  4*DB * kappaBB / h**4
            M[idx_B(i), idx_B(i+2)] += -DB * kappaBB / h**4

            M[idx_B(i), idx_A(i-2)] += -DB * kappaAB / h**4
            M[idx_B(i), idx_A(i-1)] +=  4*DB * kappaAB / h**4
            M[idx_B(i), idx_A(i)  ] += -6*DB * kappaAB / h**4
            M[idx_B(i), idx_A(i+1)] +=  4*DB * kappaAB / h**4
            M[idx_B(i), idx_A(i+2)] += -DB * kappaAB / h**4

    return M.tocsr()
    '''

rB_list = np.exp(np.linspace(np.log(0.94), np.log(1.0), 1000))
k_list = [5,4,3,2,1]

# Speicher für max-Eigenwert und die jeweils k-t-größten
eigvals_max = np.zeros_like(rB_list)
eigvals_k   = {k: np.zeros_like(rB_list) for k in k_list}

for i, rB in enumerate(rB_list):
    Jsys = build_J(rB).toarray()
    vals = np.sort(np.real(np.linalg.eigvals(Jsys)))
    eigvals_max[i] = vals[-1]
    for k in k_list:
        eigvals_k[k][i] = vals[-k]

# Plot
plt.figure(figsize=(8,5))
plt.plot(rB_list, eigvals_max, lw=2, label='max Re λ')
for k in k_list:
    plt.plot(rB_list, eigvals_k[k], '--', label=f'{k}-tgrößter Re λ')

plt.axhline(0, color='k', lw=1)
plt.xlabel(r'$r_B$')
plt.ylabel('Re Eigenwert')
plt.title('Turing-Instabilität für verschiedene Spektrumsindizes')
plt.legend()
plt.tight_layout()
plt.savefig('eigvals_comparison_zoom_zoom_new_values.png', dpi=300)


