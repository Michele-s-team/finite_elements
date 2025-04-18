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

# return the matrix of a rotation by an angle 'theta' about the z axis
def R_z(theta):
    return [[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]]


'''
given a rectangle with its bottom-left corner at the origin and a point inscribed in it, return the minimal distance between the circle center and the rectangle boundary
Input values: 
- 'L', 'h': the length and  height of the rectangle
- 'p' : the coordinates of the point
Return values: 
- the minimal distance
'''

def min_dist_c_r_rectangle(L, h, p):
    if p[0] < L/2:
        min_x = p[0]
    else:
        min_x = L-p[0]

    if p[1] < h/2:
        min_y = p[1]
    else:
        min_y = h-p[1]

    return min(min_x, min_y)

