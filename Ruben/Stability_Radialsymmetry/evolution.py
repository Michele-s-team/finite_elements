import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sympy as sp
from sympy import sqrt

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
solver = mp.odefun(ode_system, Rmin, y0, tol = 1e-10)

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
sigma_prime_vals = np.array([
    f_sigma_prime(
        float(r_vals[i]),
        z_vals[i],
        zp_vals[i],
        zpp_vals[i],
        z3_vals[i],
        sigma_vals[i]
    )
    for i in range(len(r_vals))
], dtype=float)

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
Hp_vals = np.gradient(H_vals, r_vals)
Hpp_vals = np.gradient(Hp_vals, r_vals)
grr_vals = 1.0 + zp_vals**2
g_rr_inv_vals = 1.0/grr_vals
b_rr_vals = zpp_vals/np.sqrt(1.0+zp_vals**2)

b_rr_inv_vals = g_rr_inv_vals**2 * b_rr_vals
'''
plt.figure()
plt.plot(r_vals, b_rr_vals, label='b_rr')
plt.plot(r_vals, b_rr_inv_vals, label='b_rr_inv')
plt.legend()
plt.show()
'''
# 4) evaluate them on your solver arrays
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

# 5) (optional) plot to check


sqrt_detg_vals = r_vals * np.sqrt(1 + zp_vals**2)
phi_vals = (r_vals**2 * Hp_vals) / (r_vals * np.sqrt(1 + zp_vals**2))
#nablaH_vals = np.gradient(phi_vals, r_vals) / (r_vals * np.sqrt(1 + zp_vals**2))

Hprime_vals = Hprime_fun(r_vals, zp_vals, zpp_vals, z3_vals)
Hpp_vals    = Hpp_fun(   r_vals, zp_vals, zpp_vals, z3_vals, z4_vals)
K_vals = zp_vals * zpp_vals /( r_vals * (1+zp_vals**2)**2 )
'''
plt.figure(); plt.plot(r_vals, Hp_vals, label="H'(r)")
plt.figure(); plt.plot(r_vals, Hpp_vals,    label="H''(r)")
plt.show()
plt.figure(); plt.plot(r_vals, z_vals);      plt.title('z(r)')
plt.figure(); plt.plot(r_vals, zp_vals);     plt.title("z'(r)")
plt.figure(); plt.plot(r_vals, v_vals);      plt.title('v(r)')
plt.figure(); plt.plot(r_vals, H_vals);      plt.title('H(r)')
plt.figure(); plt.plot(r_vals, Hp_vals);     plt.title("H'(r)")
plt.figure(); plt.plot(r_vals, nablaH_vals); plt.title('Nabla_i Nabla^i H')
plt.figure(); plt.plot(r_vals, sigma_vals);  plt.title('sigma(r)')
plt.show()
'''


v_prime_vals = -C1_val*(
    1.0/(r_vals**2 * np.sqrt(1+zp_vals**2))
    + zp_vals*zpp_vals/(r_vals*(1+zp_vals**2)**1.5)
)
r, C1 = symbols('r C1', positive=True)
zp, zpp = symbols('zp zpp')
v_expr = C1/(r * sqrt(1 + zp**2))

# 2) compute the second derivative symbolically
vpp_expr = sp.diff(v_expr, r, 2)

# 3) lambdify for numpy (set C1=1 if that’s your case)
f_vpp = sp.lambdify((r, zp, zpp, C1), vpp_expr, 'numpy')

# 4) evaluate on your arrays
C1_val = 1.0
v_pp_vals = f_vpp(r_vals, zp_vals, zpp_vals, C1_val)
'''
plt.figure()
plt.plot(r_vals, v_vals, label='v(r)')
plt.plot(r_vals, v_prime_vals, label="v'(r)")
plt.plot(r_vals, v_pp_vals, label="v''(r)")
plt.legend()
plt.show()
'''
print("Solved ODE")
# here we have the option to reset the grid coarsening
npts = len(r_vals)
npts = 200
r_vals = np.linspace(Rmin, Rmax, npts)
r_vals = np.linspace(Rmin, Rmax, npts)
print('Number of points:',npts)
dr = r_vals[1] - r_vals[0]
M = np.zeros((3*npts, 3*npts))
sol = [solver(float(rv)) for rv in r_vals]
z_vals   = np.array([ float(s[0]) for s in sol])
zp_vals  = np.array([ float(s[1]) for s in sol])
zpp_vals = np.array([ float(s[2]) for s in sol])
z3_vals  = np.array([ float(s[3]) for s in sol])
sigma_vals = np.array([float(s[4]) for s in sol])
v_vals = C1_val / (r_vals * np.sqrt(1 + zp_vals**2))
v_prime_vals = -C1_val*(
    1.0/(r_vals**2 * np.sqrt(1+zp_vals**2))
    + zp_vals*zpp_vals/(r_vals*(1+zp_vals**2)**1.5)
)
grr_vals = 1.0 + zp_vals**2
g_rr_inv_vals = 1.0/grr_vals
f_vpp = sp.lambdify((r, zp, zpp, C1), vpp_expr, 'numpy')
v_pp_vals = f_vpp(r_vals, zp_vals, zpp_vals, C1_val)
H_vals = H_fun(r_vals, zp_vals, zpp_vals)
Hp_vals = np.gradient(H_vals, r_vals)
Hpp_vals = np.gradient(Hp_vals, r_vals)
K_vals = zp_vals * zpp_vals /( r_vals * (1+zp_vals**2)**2 )
b_rr_vals = zpp_vals/np.sqrt(1.0+zp_vals**2)
b_rr_inv_vals = g_rr_inv_vals**2 * b_rr_vals
npts = len(r_vals)
dr = r_vals[1] - r_vals[0]
def neumann_D1(n,h):
    D1 = sp.lil_matrix((n, n))
    for i in range(1, n-1):
        D1[i, i-1] = -0.5 / h
        D1[i, i+1] = +0.5 / h
    D1[0, 0] = -3.0 / (2*h)
    D1[0, 1] = +4.0 / (2*h)
    D1[0, 2] = -1.0 / (2*h)
    D1[-1, -1] = +3.0 / (2*h)
    D1[-1, -2] = -4.0 / (2*h)
    D1[-1, -3] = +1.0 / (2*h)
    return D1.tocsr()

def neumann_D2(n,h):
    main = -2*np.ones(n)
    off  =  1*np.ones(n-1)
    D2   = sp.diags([off,main,off],[-1,0,1],format="lil")
    # ∂f/∂x=0 ⇒ f[-1]=f[1], f[n]=f[n-2]
    D2[0,0],   D2[0,1]   = -1,  +1
    D2[-1,-2],D2[-1,-1] = +1,  -1
    return (D2/h**2).tocsr()
D1 = neumann_D1(npts, dr)
D2 = neumann_D2(npts, dr)

v0 = np.zeros(npts)
z0 = np.zeros(npts)
w0 = np.zeros(npts)

def rhs(t, Y):
    v = Y[:npts]
    z = Y[npts:2*npts]
    w = Y[2*npts:3*npts]
    zr = D1.dot(z)
    zrr = D2.dot(z)
    z3 = D1.dot(zrr)
    z4 = D2.dot(zrr)
    dv = np.zeros_like(v)
    dz = w*np.sqrt(1+zr**2)
    dw = np.zeros_like(w)
    return np.concatenate([dv, dz, dw])
amp_noise = 0.001
Y0 = np.concatenate([
    v0 + amp_noise*np.random.randn(n),
    z0 + amp_noise*np.random.randn(n),
    w0 + amp_noise*np.random.randn(n),
])
t_span = (0, 10)
sol = solve_ivp(rhs, t_span, Y0, method = "RK23", atol=1e-6, rtol = 1e-6)
times = sol.t
Vt = sol.y[:npts]
nrom_v = np.linalg.norm(Vt-v0[:,None], axis=0)

plt.figure()
plt.semilogy(times, norm_v, label)

##Non-linear evolution of perturbation 
