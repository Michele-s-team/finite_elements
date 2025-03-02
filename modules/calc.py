import numpy as np
import scipy.integrate as spi


# return the radian angle of vector r by taking into account its quadrant
def atan_quad(r):
    if (r[0] > 0):
        angle = np.arctan( r[1] / r[0] )
    else:
        angle = np.pi + np.arctan( r[1] / r[0] )

    return angle - 2 * np.pi * np.floor( angle / (2 * np.pi) )

'''
a line in 2d joining the points x_a and x_b, parametrized with 0 <= t <= 1
it returns the curve and its gradient [[x[0](t), x[1](t)], [x[0]'(t), x[1]'(t)]]
'''
def line_x_a_x_b(x_a, x_b, t):
    return [x_a + np.subtract(x_b, x_a)*t, np.subtract(x_b, x_a)]


'''
return \int_a^b dx f(x)
'''
def integral_1d_segment(f, a, b):
    result = spi.quad(f, a, b)[0]
    return result


'''
return the curvilinear integral of the function f(x[0], x[1]) along the curve gamma_dgamma
gamma_dgamma must return [gamma, [grad_gamma]], where gamma = [x[0](t), x[1](t)] and grad_gamma = [x[0]'(t), x[1]'(t)], and the curve is defined for 0<= t <= 1
'''
def integral_2d_curve(f, gamma_dgamma):
   # result = quad(lambda t: (f(gamma_dgamma(t)[0]) * geo.my_norm((gamma_dgamma(t))[1])), 0, 1)[0]
   result = spi.quad(lambda t: (f(gamma_dgamma(t)[0]) * np.linalg.norm((gamma_dgamma( t ))[1])   ), 0, 1)[0]
   return result
