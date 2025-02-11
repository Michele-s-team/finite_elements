from fenics import *
from mshr import *
import ufl as ufl

import geometry as geo

i, j, k, l = ufl.indices( 4 )


#Pi(v, w, omega, sigma)[i, j] = \Pi^{ij}_notes, i.e., the momentum-flux tensor
def Pi(v, w, omega, sigma, eta):
    return as_tensor( - geo.g_c(omega)[i, j] * sigma - 2.0 * eta * geo.d_c(v, w, omega)[i ,j], (i, j) )

#dFdl(v, w, omega, sigma, eta, nu)[i] = dF^i/dl_notes, i.e., the force per unit length exerted on a line element with normal nu[i] = \nu^i_notes
def dFdl(v, w, omega, sigma, eta, nu):
    return as_tensor(Pi(v, w, omega, sigma, eta)[i, j] * geo.g(omega)[j, k] * nu[k], (i))

#fel_n = f^{EL}_notes , i.e.,  part of the normal force due to the bending rigidity
def fel_n(omega, mu, tau, kappa):
    return (kappa * ( - 2.0 * tau - 4.0 * mu * ( (mu**2) - geo.K(omega) ) ))


#fvisc_n(v, w, omega, mu, eta) = f^{VISC}_n_notes, i.e., viscous contribution to the normal force
def fvisc_n(v, w, omega, mu, eta):
    return ( 2.0 * eta * ( geo.g_c(omega)[i, k] * geo.Nabla_v(v, omega)[j, k] * geo.b(omega)[i, j] - 2.0 * w * ( 2.0 * (mu**2) - geo.K(omega) )  )  )

#tforce coming from the Laplace preccure. flaplace ={ 2 * \sigma * H }_notes
def flaplace(sigma, omega):
    return (2.0 * sigma * geo.H(omega))

#fvisc_t[i] = f^{VISC i}_notes. here the argument d is defined in the same way as geo.d
def fvisc_t(v, w, d, omega, eta):
    return as_tensor(2.0 * eta * geo.g_c(omega)[i, j] * geo.g_c(omega)[k, l] * geo.Nabla_ff(d, omega)[j, l, k], (i))
