import sympy as sp
from sympy import symbols, Function, sqrt, simplify, diff, Matrix, det
import mpmath as mp
mp.mp.dps = 10
import matplotlib.pyplot as plt
import numpy as np

# Symbols and functions
r       = sp.symbols('r', positive=True)
z       = sp.Function('z')(r)
sigma   = sp.Function('sigma')(r)
kappa, eta, rho, C1 = sp.symbols('kappa eta rho C1', positive=True)


zp  = sp.diff(z, r)
zpp = sp.diff(z, r, 2)
z3  = sp.diff(z, r, 3)
z4  = sp.diff(z, r, 4)
v     = C1/(r*sp.sqrt(1+zp**2))
vprime = sp.diff(v, r)
vpp    = sp.diff(vprime, r)
g     = sp.diag(1+zp**2, r**2)
b = sp.diag(zpp/sqrt(1+zp**2),r*zp/sqrt(1+zp**2))
invg  = g.inv()
Gamma_rr_r       = zp*zpp/(1+zp**2)
Gamma_theta_r_th = 1/r  
Delta_r_v     = vprime + v*Gamma_rr_r
Delta_theta_v = v*Gamma_theta_r_th

eq1 = (
    rho*v*Delta_r_v
    - invg[0,0]*sp.diff(sigma, r)
    - 2*eta*invg[0,0]*(
         sp.diff(Delta_r_v, r)
         + (Delta_r_v - Delta_theta_v)*Gamma_theta_r_th
      )
)
b11 = zpp/sqrt(1+zp**2)
#what is b22 correctly?
b22 = r*zp/sqrt(1+zp**2)
H = 1/2*sp.trace(invg*b)
K = b.det()/g.det()
sqrtg = sp.sqrt(g.det())
Phi_expr = (r**2*sp.diff(H,r))/sqrtg
NablanablaH_expr = 1/sqrtg*sp.diff(Phi_expr,r)
NablaNablaH_sym = sp.simplify(1/sqrtg*sp.diff(Phi_expr,r))
NablaNablaH = (1/sqrtg)*sp.diff(r**2*sp.diff(H,r)/sqrtg,r)

eq2 = (
    rho*v**2*b11
    - (
        kappa*(-2*NablaNablaH - 4*H*(H**2 - K))
        + 2*sigma*H
        + 2*eta*( invg[0,0]*Delta_r_v*b11 + invg[1,1]*Delta_theta_v*b22 )
      )
)

eq1 = eq1.subs({sp.diff(v, r): vprime, sp.diff(v, r,2): vpp})
eq2 = eq2.subs({sp.diff(v, r): vprime, sp.diff(v, r,2): vpp})
nums = {kappa:1, eta:1, rho:1, C1:1}
eq1 = eq1.subs(nums)
eq2 = eq2.subs(nums)

sigma_prime_expr = sp.solve(eq1, sp.diff(sigma, r))[0]
z4_expr          = sp.solve(eq2, z4)[0]

from sympy import lambdify
z0, z1, z2, z3, s0 = sp.symbols('z0 z1 z2 z3 s0')

subs_map = {
    z:       z0,
    zp:      z1,
    zpp:     z2,
    sp.diff(z, r, 3): z3,
    sigma:   s0
}

f_sigma_prime = lambdify(
    (r, z0, z1, z2, z3, s0),
    sigma_prime_expr.subs(subs_map),
    'mpmath'
)
f_z4 = lambdify(
    (r, z0, z1, z2, z3, s0),
    z4_expr.subs(subs_map),
    'mpmath'
)

def ode_system(rv, Y):
    z0_val, z1_val, z2_val, z3_val, s_val = Y
    return [
      z1_val,
      z2_val,
      z3_val,
      f_z4(rv, z0_val, z1_val, z2_val, z3_val, s_val),
      f_sigma_prime(rv, z0_val, z1_val, z2_val, z3_val, s_val)
    ]

Rmin, Rmax = 1.0, 2.0
y0 = [0.0,      # z(Rmin)
      0.1,      # z'(Rmin)
      0.0,      # z''(Rmin)
      0.0,      # z'''(Rmin)
      float((201-40*sp.sqrt(101))/202)  # sigma(Rmin)
     ]
z4_sym = sp.symbols('z4')
subs_map_nabla = {
  z:            z0,
  zp:           z1,
  zpp:          z2,
  sp.diff(z,r,3): z3,
  sp.diff(z,r,4): z4_sym,
  sigma:        s0
}
f_NablaNablaH = sp.lambdify(
  (r, z0, z1, z2, z3, z4_sym, s0),
  NablaNablaH.subs(subs_map_nabla),
  'mpmath'
)
solver = mp.odefun(ode_system, Rmin, y0, tol = 1e-20)

r_vals = np.linspace(Rmin, Rmax, 10001)
sol = [solver(float(rv)) for rv in r_vals]
z_vals   = np.array([ float(s[0]) for s in sol])
zp_vals  = np.array([ float(s[1]) for s in sol])
zpp_vals = np.array([ float(s[2]) for s in sol])
z3_vals  = np.array([ float(s[3]) for s in sol])
C1_val = 1.0
v_vals = C1_val / (r_vals * np.sqrt(1 + zp_vals**2))
#H_expr = (zpp/(1+zp**2)**(3/2) + zp/r/(sp.sqrt(1+zp**2)))/2
H_fun = sp.lambdify(
    (r, zp, zpp),
    ( zpp/(1+zp**2)**(sp.Rational(3,2))
    + zp/( r*sp.sqrt(1+zp**2)) )/2,
    'numpy'
)
H_vals = H_fun(r_vals, zp_vals, zpp_vals)
sigma_vals = np.array([float(s[4]) for s in sol])

#here get Hp and Hpp correctly
H_expr   = ( zpp/(1+zp**2)**(sp.Rational(3,2))
           + zp/( r*sp.sqrt(1+zp**2)) )/2

# 2) differentiate it once and twice
Hprime_expr = sp.simplify(sp.diff(H_expr, r))
Hpp_expr    = sp.simplify(sp.diff(Hprime_expr, r))
z1,z2,z3,z4sym = sp.symbols('z1 z2 z3 z4')
subs_Hp = { zp: z1, zpp: z2, sp.diff(z,r,3): z3, z:0 }
subs_Hpp = { zp: z1, zpp: z2, sp.diff(z,r,3): z3,
             sp.diff(z,r,4): z4sym, z:0 }

Hprime_fun = sp.lambdify((r,z1,z2,z3),
                         Hprime_expr.subs(subs_Hp),
                         'numpy')
Hpp_fun    = sp.lambdify((r,z1,z2,z3,z4sym),
                         Hpp_expr.subs(subs_Hpp),
                         'numpy')
Hp_vals_normal = np.gradient(H_vals, r_vals)
Hpp_vals_normal = np.gradient(Hp_vals, r_vals)


# 4) evaluate them on your solver arrays
Hprime_vals = Hprime_fun(r_vals, zp_vals, zpp_vals, z3_vals)
Hpp_vals    = Hpp_fun(   r_vals, zp_vals, zpp_vals, z3_vals, z4_vals)

# 5) (optional) plot to check
plt.figure(); plt.plot(r_vals, Hprime_vals, label="H'(r)"); plt.plot(r_vals, H)
plt.figure(); plt.plot(r_vals, Hpp_vals,    label="H''(r)")
plt.show()

sqrt_detg_vals = r_vals * np.sqrt(1 + zp_vals**2)
phi_vals = (r_vals**2 * Hp_vals) / (r_vals * np.sqrt(1 + zp_vals**2))
nablaH_vals = np.gradient(phi_vals, r_vals) / (r_vals * np.sqrt(1 + zp_vals**2))
z4_vals = np.array([
  f_z4(rv, 0, zp_vals[i], zpp_vals[i], z3_vals[i], sigma_vals[i])
  for i,rv in enumerate(r_vals)
], dtype=float)
nablaH_vals = np.array([
  f_NablaNablaH(
    rv,
    0,
    zp_vals[i],
    zpp_vals[i],
    z3_vals[i],
    z4_vals[i],
    sigma_vals[i]
  )
  for i,rv in enumerate(r_vals)
], dtype=float)
K_vals = zp_vals * zpp_vals /( r_vals * (1+zp_vals**2)**2 )


plt.figure(); plt.plot(r_vals, z_vals);      plt.title('z(r)')
plt.figure(); plt.plot(r_vals, zp_vals);     plt.title("z'(r)")
plt.figure(); plt.plot(r_vals, v_vals);      plt.title('v(r)')
plt.figure(); plt.plot(r_vals, H_vals);      plt.title('H(r)')
plt.figure(); plt.plot(r_vals, Hp_vals);     plt.title("H'(r)")
plt.figure(); plt.plot(r_vals, nablaH_vals); plt.title('Nabla_i Nabla^i H')
plt.figure(); plt.plot(r_vals, sigma_vals);  plt.title('sigma(r)')
plt.show()


v_prime_vals = -C1_val*(
    1.0/(r_vals**2 * np.sqrt(1+zp_vals**2))
    + zp_vals*zpp_vals/(r_vals*(1+zp_vals**2)**1.5)
)

npts = len(r_vals)
dr = r_vals[1] - r_vals[0]
M = np.zeros((3*npts, 3*npts))

def idx_zeta(i): return i
def idx_u(i):    return npts + i
def idx_psi(i):  return 2*npts + i

kappa = eta = rho = 1.0
A_vals = 0.5*( -12*kappa/rho * H_vals**2
         -4*kappa/rho * K_vals
         +2/rho * sigma_vals )
B = eta/rho
for i in range(npts):
    M[idx_psi(i), idx_u(i)] += -2*v_vals[i]*zpp_vals[i]/np.sqrt(1+zp_vals[i]**2)
    M[idx_zeta(i), idx_psi(i)] += np.sqrt(1+zp_vals[i]**2)
for i in range(1, npts-1):
    M[idx_psi(i), idx_psi(i-1)] += v_vals[i]/(2*dr)
    M[idx_psi(i), idx_psi(i+1)] += -v_vals[i]/(2*dr)
    M[idx_psi(i), idx_zeta(i+1)] += -zpp_vals[i]*zp_vals[i]*v_vals[i]**2/(2*dr*(1+zp_vals[i]**2)**(3/2))
    M[idx_psi(i), idx_zeta(i-1)] += +zpp_vals[i]*zp_vals[i]*v_vals[i]**2/(2*dr*(1+zp_vals[i]**2)**(3/2))
for i in range(2, npts-2):
    M[idx_psi(i), idx_zeta(i+1)] += v_vals[i]**2/np.sqrt(1+zp_vals[i]**2)/(dr**2)
    M[idx_psi(i), idx_zeta(i-1)] += v_vals[i]**2/np.sqrt(1+zp_vals[i]**2)/(dr**2)
    M[idx_psi(i), idx_zeta(i)] += -2*v_vals[i]**2/np.sqrt(1+zp_vals[i]**2)/(dr**2)
for i in range(2, npts-2):
    M[idx_psi(i), idx_zeta(i+1)] += A_vals[i]/(dr**2*np.sqrt(1+zp_vals[i]**2)**(3/2))
    M[idx_psi(i), idx_zeta(i-1)] += A_vals[i]/(dr**2*np.sqrt(1+zp_vals[i]**2)**(3/2))
    M[idx_psi(i), idx_zeta(i)] += -2*A_vals[i]/(dr**2*np.sqrt(1+zp_vals[i]**2)**(3/2))

for i in range(1, npts-1):
    M[idx_psi(i), idx_zeta(i+1)] += A_vals[i]*(-zpp_vals[i]*3*zp_vals[i]/(1+zp_vals[i]**2)**(5/2)+ 1/(r_vals[i]*(1+zp_vals[i]**2)**(3/2)))
    M[idx_psi(i), idx_zeta(i-1)] += -A_vals[i]*(-zpp_vals[i]*3*zp_vals[i]/(1+zp_vals[i]**2)**(5/2)+ 1/(r_vals[i]*(1+zp_vals[i]**2)**(3/2)))
B = eta/rho
for i in range(2, npts-2):
    M[idx_psi(i), idx_zeta(i+1)] += B/(dr**2*np.sqrt(1+zp_vals[i]**2)**(3/2))*(v_prime_vals[i]+ 2*v_vals[i]*zp_vals[i]/(1+zp_vals[i]**2))
    M[idx_psi(i), idx_zeta(i-1)] += B/(dr**2*np.sqrt(1+zp_vals[i]**2)**(3/2))*(v_prime_vals[i]+ 2*v_vals[i]*zp_vals[i]/(1+zp_vals[i]**2))
    M[idx_psi(i), idx_zeta(i)] += -2*B/(dr**2*np.sqrt(1+zp_vals[i]**2)**(3/2))*(v_prime_vals[i]+ 2*v_vals[i]*zp_vals[i]/(1+zp_vals[i]**2))

for i in range(1, npts-1):
    M[idx_psi(i), idx_u(i+1)] += B*(zpp_vals[i]/(1+zp_vals[i]**2)**(3/2))/(2*dr)
    M[idx_psi(i), idx_u(i-1)] += -B*(zpp_vals[i]/(1+zp_vals[i]**2)**(3/2))/(2*dr)
    M[idx_psi(i), idx_zeta(i+1)] += B*(-v_prime_vals[i]*zpp_vals[i]*3*zp_vals[i]/(1+zp_vals[i]**2)**(5/2)+v_vals[i]*zpp_vals[i]*(1-2*zp_vals[i]**2)/(1+zp_vals[i]**2)**(5/2))/(2*dr)
    M[idx_psi(i), idx_zeta(i-1)] += B*(v_prime_vals[i]*zpp_vals[i]*3*zp_vals[i]/(1+zp_vals[i]**2)**(5/2)-v_vals[i]*zpp_vals[i]*(1-2*zp_vals[i]**2)/(1+zp_vals[i]**2)**(5/2))/(2*dr)

for i in range(0, npts):
    M[idx_psi(i), idx_u(i)] += B*(zpp_vals[i]/(1+zp_vals[i]**2)**(3/2))*zp_vals[i]*zpp_vals[i]/(1+zp_vals[i]**2)
    M[idx_psi(i), idx_psi(i)] += -2*B*(H_vals[i]**2-K_vals[i])

#2nd order term of 4H*delta K
for i in range(2, npts-2):
    M[idx_psi(i), idx_zeta(i+1)] += kappa/rho/(dr**2)*(-4*H_vals[i]*zp_vals[i]/(r_vals[i]*(1+zp_vals[i]**2)**2))
    M[idx_psi(i), idx_zeta(i-1)] += kappa/rho/(dr**2)*(-4*H_vals[i]*zp_vals[i]/(r_vals[i]*(1+zp_vals[i]**2)**2))
    M[idx_psi(i), idx_zeta(i)] += -2*kappa/rho/(dr**2)*(-4*H_vals[i]*zp_vals[i]/(r_vals[i]*(1+zp_vals[i]**2)**2))

# 1st order term of 4H*delta K
for i in range(1, npts-1):
    M[idx_psi(i), idx_zeta(i+1)] += kappa/rho/(2*dr)*(-4*H_vals[i])*(1-3*zp_vals[i]**2)*zpp_vals[i]/(r_vals[i]*(1+zp_vals[i]**2)**3)
    M[idx_psi(i), idx_zeta(i-1)] += -kappa/rho/(2*dr)*(-4*H_vals[i])*(1-3*zp_vals[i]**2)*zpp_vals[i]/(r_vals[i]*(1+zp_vals[i]**2)**3)

# implementation of delta(laplace H)

#part D
#for i in range(1, npts-1):

