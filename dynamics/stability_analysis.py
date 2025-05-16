import sympy as sp
import numpy as np
from scipy.integrate import solve_bvp
from scipy.interpolate import interp1d
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigs

r, theta = sp.symbols('r theta', real=True)
eta, kappa, rho, C1 = sp.symbols('eta kappa rho C1', positive=True)
sigma = sp.Function('sigma')
z = sp.Function('z')
v = sp.Function('v')

g = sp.Matrix([[1 + sp.diff(z(r), r)**2, 0],
               [0, r**2]])
g_inv = sp.simplify(g.inv())
detg = sp.simplify(g.det())
sqrt_detg = sp.sqrt(detg)

Gamma_rrr = sp.simplify((sp.diff(z(r),r)*sp.diff(z(r),r,2)) / (1 + sp.diff(z(r),r)**2))
Gamma_rtt = sp.simplify(-r / (1 + sp.diff(z(r),r)**2))
Gamma_trt = sp.simplify(1/r)

Delta_r_v_r     = sp.diff(v(r), r) + v(r) * Gamma_rrr
Delta_theta_v_theta = v(r) * Gamma_trt

e1 = sp.Matrix([sp.cos(theta), sp.sin(theta), sp.diff(z(r),r)])
e2 = sp.Matrix([-r*sp.sin(theta), r*sp.cos(theta), 0])
normal_vec = sp.simplify(e1.cross(e2) / sqrt_detg)

b = sp.Matrix([
    [sp.diff(e1, r).dot(normal_vec), sp.diff(e1, theta).dot(normal_vec)],
    [sp.diff(e2, r).dot(normal_vec), sp.diff(e2, theta).dot(normal_vec)]
])

H = sp.simplify(sp.Rational(1,2) * (g_inv * b).trace())
H_expr = H
normal_comp = normal_vec[:2]
K = sp.simplify(b.det() / detg)

phi = sp.simplify(r**2 * sp.diff(H, r) / sqrt_detg)
NablaNablaH = sp.simplify(sp.diff(phi, r) / sqrt_detg)

fVISCt = sp.simplify(
    1/(1 + sp.diff(z(r),r)**2)
    * 2*eta * (
        sp.diff(v(r), r, 2)
        + (sp.diff(v(r),r)*sp.diff(z(r),r)*sp.diff(z(r),r,2))/(1+sp.diff(z(r),r)**2)
        - (2*v(r)*sp.diff(z(r),r)**2*sp.diff(z(r),r,2)**2)/(1+sp.diff(z(r),r)**2)**2
        + (v(r)*sp.diff(z(r),r,2)**2)/(1+sp.diff(z(r),r)**2)
        + (-(v(r)/r) + sp.diff(v(r),r) + (v(r)*sp.diff(z(r),r)*sp.diff(z(r),r,2))/(1+sp.diff(z(r),r)**2))/r
        + (v(r)*sp.diff(z(r),r)**3)/(1+sp.diff(z(r),r)**2)
    )
)
fSigma_t = sp.simplify(g_inv[0,0] * sp.diff(sigma(r), r))
f_v = sp.simplify(rho * v(r) * Delta_r_v_r)

d = sp.diag(g[0,0]*Delta_r_v_r, g[1,1]*Delta_theta_v_theta)
Pi = sp.simplify(
    -sigma(r) * g_inv - 2*eta * sp.diag(g_inv[0,0], g_inv[1,1]) * d
)

dFdl_EtaSigma = sp.Matrix([
    [sp.simplify(sum(Pi[i,j]*g[j,k]*normal_vec[k] for j in range(2) for k in range(2))) for i in range(2)]
])
dFdl_Kappa_t = sp.simplify(
    sp.Matrix([-2 * kappa * H**2 * normal_vec[i] for i in range(2)])
)
dFdl_Kappa_n = sp.simplify(2*kappa*sp.diff(H, r) * normal_vec[0])

def repv_subs(expr):
    """Replace v, v', v'' by continuity expressions."""
    return expr.subs({
        v(r): C1/(r*sp.sqrt(1+sp.diff(z(r),r)**2)),
        sp.diff(v(r),r): -C1*(1+sp.diff(z(r),r)**2 + r*sp.diff(z(r),r)*sp.diff(z(r),r,2))/(r**2*(1+sp.diff(z(r),r)**2)**(sp.Rational(3,2))),
        sp.diff(v(r),r,2): C1*(2+2*sp.diff(z(r),r)**4 - r**2*sp.diff(z(r),r,2)**2 + 2*sp.diff(z(r),r)**2*(2+r**2*sp.diff(z(r),r,2)**2)
            + r*sp.diff(z(r),r)*(2*sp.diff(z(r),r,2)-r*sp.diff(z(r),r,3))
            + r*sp.diff(z(r),r)**3*(2*sp.diff(z(r),r,2)-r*sp.diff(z(r),r,3))
        )/(r**3*(1+sp.diff(z(r),r)**2)**(sp.Rational(5,2)))
    })

    eq1, eq2 = ...
eq2 = ...
eqs = [eq1, eq2]
eq1 = sp.solve(sp.simplify((rho*v(r)*Delta_r_v_r).subs({\
    }), sp.diff(sigma(r),r))[0])
eq2 = sp.solve(sp.simplify(eqs[1] - 4*C1*r*eta*(1+sp.diff(z(r),r)**2)**(sp.Rational(5,2))*sp.diff(z(r),r,2)), sp.diff(z(r),r,2))[0]

def fun(r_vals, Y):
    """
    ODE system for [z, z', sigma].
    Y[0]=z, Y[1]=z', Y[2]=sigma
    """
    zp = Y[1]
    sig = Y[2]
    zpp = sp.lambdify((r, z(r), sp.diff(z(r),r), sigma(r)), eq2_expr, 'numpy')(
        r_vals, Y[0], Y[1], Y[2]
    )
    sigp = sp.lambdify((r, z(r), sp.diff(z(r),r), sigma(r)), eq1_expr, 'numpy')(
        r_vals, Y[0], Y[1], Y[2]
    )
    return np.vstack((Y[1], zpp, sigp))

def bc(Y0, Y1):
    return np.array([
        Y0[1] - zpRmin,      
        Y1[1] - 0.0,           
        Y0[2] - sigma_Rmin     
    ])

Rmin, Rmax = 1.0, 2.0
eta_val, kappa_val, rho_val, C1_val = 1e-2, 3e-2, 1e-12, 1.0
zpRmin = 0.1

sigma_Rmin = float((sp.solve(eqs[1].subs(r, Rmin).subs({sp.diff(z(r),r): zpRmin, sp.diff(z(r),r,2):0}), sigma(r))[0])[sigma(r)])

t_r = np.linspace(Rmin, Rmax, 50)
Y_guess = np.zeros((3, t_r.size))
sol = solve_bvp(fun, bc, t_r, Y_guess, max_nodes=2000)

z_ifunc = interp1d(sol.x, sol.y[0], kind='cubic')
v_ifunc = lambda rr: C1_val/(rr*np.sqrt(1 + (np.gradient(z_ifunc(rr), rr))**2))
z1_ifunc = interp1d(sol.x, sol.y[1], kind='cubic')
z2_ifunc = lambda rr: np.gradient(z1_ifunc(rr), rr)

npts = 3
rGrid = np.linspace(Rmin, Rmax, npts)
dr = (Rmax - Rmin)/(npts-1)
zStar    = z_ifunc(rGrid)
zPrime   = z1_ifunc(rGrid)
zDouble  = z2_ifunc(rGrid)
vStar    = v_ifunc(rGrid)
brrStar  = zDouble/np.sqrt(1 + zPrime**2)


def idxZ(i): return i

def idxW(i): return npts + i

def idxV(i): return 2*npts + i


rows, cols, data = [], [], []
for i in range(npts):
    jm = i-1 if i>0 else 1
    jp = i+1 if i<npts-1 else npts-2

    rows += [idxZ(i), idxW(i), idxW(i), idxW(i), idxW(i)]
    cols += [idxW(i), idxV(i), idxW(jp), idxW(jm), idxV(i)]  
    data += [np.sqrt(1 + zPrime[i]**2),
             -2*vStar[i]*brrStar[i],
             -vStar[i]/(2*dr),
              vStar[i]/(2*dr),
             0  
            ]

size = 3*npts
'''
A = coo_matrix((data, (rows, cols)), shape=(size, size)).tocsc()+
eigvals = eigs(A, k=1, which='LR', return_eigenvectors=False)
lambda_max = np.real(eigvals[0])
print("Dominant eigenvalue:", lambda_max)
'''
