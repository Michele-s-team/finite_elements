import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from fractions import Fraction

def parse_mathematica_number(s):
    # if it’s already numeric, leave it
    if not isinstance(s, str):
        return float(s)

    # handle exact rationals like "3/2"
    if "/" in s and s.count("/") == 1:
        return float(Fraction(s))

    # drop the back-tick precision suffix
    if "`" in s:
        s = s.split("`", 1)[0]

    return float(s)

df = pd.read_csv("solution_1D.csv", sep=",", decimal=".")
df = df.applymap(parse_mathematica_number)

r = df["r"].to_numpy()
z = df["z(r)"].to_numpy()
z_prime = df["z'(r)"].to_numpy()
plt.plot(r, z_prime, '.', label='z(r)')
plt.show()

z_double = df["z''(r)"].to_numpy()
z_triple = np.zeros(len(z_double))
z_quad = np.zeros(len(z_double))

v = df["v(r)"].to_numpy()
v_prime = df["v'(r)"].to_numpy()
sigma = df["sigma(r)"].to_numpy()
H = df["H(r)"].to_numpy()
H_prime = df["H'(r)"].to_numpy()
nabla_nabla_H = df["∇∇H(r)"].to_numpy()
kappa = 1.0
rho = 1.0
eta = 1.0
#plt.plot(r, fVISCt,'.')
#plt.show()
print(z_prime)
npts = 3
rMin, rMax = 1.0, 2.0
rGrid = np.linspace(rMin, rMax, npts)
dr = rGrid[1] - rGrid[0]

def idx_zeta(i): return i
def idx_u   (i): return npts + i
def idx_psi (i): return 2*npts + i

M = np.zeros((3*npts, 3*npts))

#  the term for d_t zeta and -2uvb_rr in dt psi
for i in range(npts):
    M[idx_psi(i), idx_u(i)] += -2*v[i]*z_double[i]/np.sqrt(1+z_prime[i]**2)
    M[idx_zeta(i), idx_psi(i)] += np.sqrt(1+z_prime[i]**2)


# here is the term for d_t psi: -v d_r psi and 1. order term of v^2 delta(b_rr)
for i in range(1, npts-1):
    M[idx_psi(i), idx_psi(i-1)] += v[i]/(2*dr)
    M[idx_psi(i), idx_psi(i+1)] += -v[i/(2*dr)]
    M[idx_psi(i), idx_zeta(i+1)] += -z_double[i]*z_prime[i]*v[i]**2/(2*dr*(1+z_prime[i]**2)**(3/2))
    M[idx_psi(i), idx_zeta(i-1)] += +z_double[i]*z_prime[i]*v[i]**2/(2*dr*(1+z_prime[i]**2)**(3/2))
# for d_t psi: 2nd order term of v^2 delta(b_rr)
for i in range(2, npts-2):
    M[idx_psi(i), idx_zeta(i+1)] += v[i]**2/np.sqrt(1+z_prime[i]**2)/(dr**2)
    M[idx_psi(i), idx_zeta(i-1)] += v[i]**2/np.sqrt(1+z_prime[i]**2)/(dr**2)
    M[idx_psi(i), idx_zeta(i)] += -2*v[i]**2/np.sqrt(1+z_prime[i]**2)/(dr**2)
K = np.zeros(npts)
A = np.zeros(npts)
for i in range(0, npts):
    K[i] = z_prime[i]*z_double[i]/(r[i]*(1+z_prime[i]**2)**(2))
    A[i] = 0.5*(-12*kappa/rho*H[i]**2 - 4*kappa/rho*K[i] + 2/rho*sigma[i])

# 2. order term in d_t psi of delta H
for i in range(2, npts-2):
    M[idx_psi(i), idx_zeta(i+1)] += A[i]/(dr**2*np.sqrt(1+z_prime[i]**2)**(3/2))
    M[idx_psi(i), idx_zeta(i-1)] += A[i]/(dr**2*np.sqrt(1+z_prime[i]**2)**(3/2))
    M[idx_psi(i), idx_zeta(i)] += -2*A[i]/(dr**2*np.sqrt(1+z_prime[i]**2)**(3/2))
# 2nd order term in d_t psi of delta H
for i in range(1, npts-1):
    M(idx_psi(i), idx_zeta(i+1)) += A[i]*(-z_double[i]*3*z_prime[i]/(1+z_prime[i]**2)**(5/2)+ 1/(r[i]*(1+z_prime[i]**2)**(3/2)))
    M[idx_psi(i), idx_zeta(i-1)] += -A[i]*(-z_double[i]*3*z_prime[i]/(1+z_prime[i]**2)**(5/2)+ 1/(r[i]*(1+z_prime[i]**2)**(3/2)))
#implementation here of psi_t the term delta(g^rr nabla_r v^r b_rr)
B = eta/rho
for i in range(2, npts-2):
    M[idx_psi(i), idx_zeta(i+1)] += B/(dr**2*np.sqrt(1+z_prime[i]**2)**(3/2))*(v_prime[i]+ 2*v[i]*z_prime[i]/(1+z_prime[i]**2))
    M[idx_psi(i), idx_zeta(i-1)] += B/(dr**2*np.sqrt(1+z_prime[i]**2)**(3/2))*(v_prime[i]+ 2*v[i]*z_prime[i]/(1+z_prime[i]**2))
    M[idx_psi(i), idx_zeta(i)] += -2*B/(dr**2*np.sqrt(1+z_prime[i]**2)**(3/2))*(v_prime[i]+ 2*v[i]*z_prime[i]/(1+z_prime[i]**2))
# 2nd order term in d_t psi of nabla_r v^r b_rr
for i in range(1, npts-1):
    M[idx_psi(i), idx_u(i+1)] += B*(z_double[i]/(1+z_prime[i]**2)**(3/2))/(2*dr)
    M[idx_psi(i), idx_u(i-1)] += -B*(z_double[i]/(1+z_prime[i]**2)**(3/2))/(2*dr)
    M[idx_psi(i), idx_zeta(i+1)] += B*(-v_prime[i]*z_double[i]*3*z_prime[i]/(1+z_prime[i]**2)**(5/2)+v[i]*z_double[i]*(1-2*z_prime[i]**2)/(1+z_prime[i]**2)**(5/2))/(2*dr)
    M[idx_psi(i), idx_zeta(i-1)] += -B*(-v_prime[i]*z_double[i]*3*z_prime[i]/(1+z_prime[i]**2)**(5/2)+v[i]*z_double[i]*(1-2*z_prime[i]**2)/(1+z_prime[i]**2)**(5/2))/(2*dr)

for i in range(0, npts):
    M[idx_psi(i), idx_u(i)] += B*z_double[i]/(1+z_prime[i]**2)**(3/2)*z_prime[i]*z_double[i]/(1+z_prime[i]**2)
    M[idx_psi(i), idx_psi(i)] += -2*B*(H[i]**2-K[i])

# 2nd order term of 4H*delta K

for i in range(2, npts-2):
    M[idx_psi(i), idx_zeta(i+1)] +=kappa/rho/(dr**2)*(-4*H[i]*z_prime[i]/(r[i]*(1+z_prime[i]**2)**2))
    M[idx_psi(i), idx_zeta(i-1)] +=kappa/rho/(dr**2)*(-4*H[i]*z_prime[i]/(r[i]*(1+z_prime[i]**2)**2))
    M[idx_psi(i), idx_zeta(i)] +=-kappa/rho/(dr**2)*(-4*H[i]*z_prime[i]/(r[i]*(1+z_prime[i]**2)**2))
#1st order term of 4H*delta K
for i in range(1, npts-1):
    M[idx_psi(i), idx_zeta(i+1)] += kappa/rho/(2*dr)*(-4*H[i])*(1-3*z_prime[i]**2)*z_double[i]/(r[i]*(1+z_prime[i]**2)**3)
    M[idx_psi(i), idx_zeta(i-1)] += -kappa/rho/(2*dr)*(-4*H[i])*(1-3*z_prime[i]**2)*z_double[i]/(r[i]*(1+z_prime[i]**2)**3)

# implementation of delta(laplace H)

#part D
for i in range(1, npts-1):
    pref = 2*z_prime[i]*kappa/rho/(2*dr)/2*(z_quad[i]/)
    M[idx_psi(i), idx_zeta(i+1)] += 


#part C

#part B

#part A

#2nd PDE

