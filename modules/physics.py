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

'''
Left-hand side of the tangential component of Navier Stokes equation (5a) in notes (version in the discrete dynamics of Crank Nicholson (cn) discretization)
This function is called 'ma' because it is the analog of the right-hand side of Newton equation  F = ma
'''
def ma_cn_t(v_bar, v_n_1, v_n_2, w_bar, w_n_1, omega_n_12, rho, dt):
    return as_tensor( rho * (v_bar[i] - v_n_1[i]) / dt + conv_cn_t( v_bar, v_n_1, v_n_2, w_bar, w_n_1, omega_n_12, rho )[i], (i) )

# convective term in the left-hand side of Eq. (5a) in notes: the left-hand side of Eq. (5a) is rho * (v_bar[i] - v_n_1[i]) / dt  + conv_cn_t = ma_cn_t
def conv_cn_t(v_bar, v_n_1, v_n_2, w_bar, w_n_1, omega_n_12, rho):
    return as_tensor( rho * ( \
                + (3.0 / 2.0 * v_n_1[j] - 1.0 / 2.0 * v_n_2[j]) * geo.Nabla_v( (v_bar + v_n_1)/2.0, omega_n_12 )[i, j]  \
                - 2.0 * (v_bar[j] + v_n_1[j])/2.0 * (w_bar + w_n_1)/2.0 * geo.g_c(omega_n_12)[i, k] * geo.b(omega_n_12)[k, j]\
                - (w_bar + w_n_1)/2.0 * geo.g_c(omega_n_12)[i, j] * (((w_bar + w_n_1)/2.0).dx(j)) \
        ), (i) )

'''
Left-hand side of the normal component of Navier Stokes equation (5b) in notes (version in the discrete dynamics of Crank Nicholson (cn) discretization)
This function is called 'ma' because it is the analog of the right-hand side of Newton equation  F = ma
'''
def ma_cn_n(v_bar, v_n_1, v_n_2, w_bar, w_n_1, omega_n_12, rho, dt):
    return (rho * ((w_bar - w_n_1) / dt) + conv_cn_n( v_bar, v_n_1, v_n_2, w_bar, w_n_1, omega_n_12, rho ))

# convective term in the left-hand side of Eq. (5b) in notes: the left-hand side of Eq. (5b) is rho * (w_bar - w_n_1) / dt  + conv_cn_n = ma_cn_n
def conv_cn_n(v_bar, v_n_1, v_n_2, w_bar, w_n_1, omega_n_12, rho):
    return ( \
                rho * ( \
                    + (v_bar[i] + v_n_1[i]) / 2.0 * (v_bar[k] + v_n_1[k]) / 2.0 * geo.b( omega_n_12 )[k, i] \
                    + (3.0 / 2.0 * v_n_1[i] - 1.0 / 2.0 * v_n_2[i]) * (((w_bar + w_n_1) / 2.0).dx( i )) \
            ) \
        )


#fsigma_t[i] = {\nabla^i \sigma}_notes
def fsigma_t(sigma, omega):
    return as_tensor(geo.g_c(omega)[i, j] * (sigma.dx(j)) , (i) )

#fvisc_t[i] = f^{VISC i}_notes. here the argument d is defined in the same way as geo.d
def fvisc_t(d, omega, eta):
    return as_tensor(2.0 * eta * geo.g_c(omega)[i, j] * geo.g_c(omega)[k, l] * geo.Nabla_ff(d, omega)[j, l, k], (i))