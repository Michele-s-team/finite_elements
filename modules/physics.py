from fenics import *
from mshr import *
import ufl as ufl
from geometry import *

#Pi(v, w, omega, sigma)[i, j] = \Pi^{ij}_notes, i.e., the momentum-flux tensor
def Pi(v, w, omega, sigma, eta):
    return as_tensor( - g_c(omega)[i, j] * sigma - 2.0 * eta * d_c(v, w, omega)[i ,j], (i, j) )

#dFdl(v, w, omega, sigma, eta, nu)[i] = dF^i/dl_notes, i.e., the force per unit length exerted on a line element with normal nu[i] = \nu^i_notes
def dFdl(v, w, omega, sigma, eta, nu):
    return as_tensor(Pi(v, w, omega, sigma, eta)[i, j] * g(omega)[j, k] * nu[k], (i))

#fel_n = f^{EL}_notes , i.e.,  part of the normal force due to the bending rigidity
def fel_n(omega, mu, nu, kappa):
    return (kappa * ( - 2.0 * g_c(omega)[i, j] * Nabla_f(nu, omega)[i, j] - 4.0 * mu * ( (mu**2) - K(omega) ) ))

#fvisc_n(v, w, omega, eta) = f^{VISC}_n_notes, i.e., viscous contribution to the normal force
def fvisc_n(v, w, omega, eta):
    return ( 2.0 * eta * ( g_c(omega)[i, k] * Nabla_v(v, omega)[j, k] * b(omega)[i, j] - 2.0 * w * ( 2.0 * ((H(omega))**2) - K(omega) )  )  )

#tforce coming from the Laplace preccure
def flaplace(sigma, omega):
    return (2.0 * sigma * H(omega))