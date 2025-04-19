import numpy as np
import scipy.integrate as spi


# return the radian angle of vector r by taking into account its quadrant
def atan_quad(r):
    if (r[0] > 0):
        angle = np.arctan(r[1] / r[0])
    else:
        angle = np.pi + np.arctan(r[1] / r[0])

    return angle - 2 * np.pi * np.floor(angle / (2 * np.pi))


'''
a line in 2d joining the points x_a and x_b, parametrized with 0 <= t <= 1
it returns the curve and its gradient [[x[0](t), x[1](t)], [x[0]'(t), x[1]'(t)]]
'''


def line(x_a, x_b, t):
    return [x_a + np.subtract(x_b, x_a) * t, np.subtract(x_b, x_a)]


'''
a circle
Input values:
- 'r': the circle radius
- 'c_r': the circle center (an array of two points)
- 't' : the parameteric coordinate of the circle, 0<=t<1
Return values:
- the curve position and derivative: [x[0](t), x[1](t)], [x[0]'(t), x[1]'(t)]
'''


def circle(r, cr, t):
    return [np.add(cr, r * np.array([np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)])).tolist(),
            (r * 2 * np.pi * np.array([- np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)])).tolist()]





'''
return the curvilinear integral of a function  along a curve 
Input values:
- 'f': the function f(x[0], x[1])
- 'gamma_dgamma': the curve and its gradient: gamma_dgamma(t) = [[x[0](t), x[1](t)], [x[0]'(t), x[1]'(t)]]
Return values:
- 'integral': the integral 

Example of usage:
    line_test = lambda t: cal.line([np.sqrt(2),0.4], [1.2,1], t)
    def g(x):
        return np.sin( x[0]**2 +np.cos( x[1]**2))
    integral_line_test = cal.integral_curve(g, line_test)
    print(f'integral_line_test: {integral_line_test}')
    
Example of usage:
    circle_test = lambda t: cal.circle(1.34, [np.sqrt(2), -np.sqrt(3)],  t)
    def g(x):
        return np.sin( x[0]**2 +np.cos( x[1]**2))
    integral_line_test = cal.integral_curve(g, circle_test)
'''


def integral_curve(f, gamma_dgamma):
    # integral = quad(lambda t: (f(gamma_dgamma(t)[0]) * geo.my_norm((gamma_dgamma(t))[1])), 0, 1)[0]
    integral = spi.quad(lambda t: (f(gamma_dgamma(t)[0]) * np.linalg.norm((gamma_dgamma(t))[1])), 0, 1)[0]
    return integral

'''
compute the integral of a function of two variables over a rectangle
Input values:
- 'f': the function f([x, y])
- 'p_bl', 'p_rt': the bottom-left and top-right corner points of the rectangle, each is a list with two entries
Result: 
- the integral \int_{rectagnle} dx dy f(x,y)

Example of usage:
    def g(x):
        return np.sin(x[0] ** 2 + np.cos(x[1] ** 2))
    integral = integral_rectangle(g, [-2,0.1], [1,1])
'''
def integral_rectangle(f, p_bl, p_tr):
    f_swapped = lambda x, y: f([y, x])
    return spi.dblquad(f_swapped, p_bl[0], p_tr[0], lambda x: p_bl[1], lambda x: p_tr[1])[0]

'''
integate a function of two variables over a ring delimited by two concentric circles
Input values 
- 'f': the function f([x, y])
- 'r', 'R': radii of the inner and outer circle defining the ring
- 'c' : center of the circles (a list of two values)
Result:
- \int_ring dx dy f

Example of usage:
    def g(x):
        return np.sin(x[0] ** 2 + np.cos(x[1] ** 2))
    integral = cal.integral_ring(g, 1/np.sqrt(3), 2, [np.sqrt(11),-0.5])
'''
def integral_ring(f, r, R, c):
    f_swapped = lambda x, y: f([y, x])

    return spi.dblquad(lambda rho, theta: rho * f_swapped(c[1] + rho * np.sin(theta), c[0] + rho * np.cos(theta)), 0, 2 * np.pi, lambda rho: r, lambda rho: R)[0]


'''
integate a function of two variables over a disk
Input values 
- 'f': the function f([x, y])
- 'r': radius of the disk
- 'c' : center of the disk
Result:
- \int_disk dx dy f

Example of usage:
    def g(x):
        return np.sin(x[0] ** 2 + np.cos(x[1] ** 2))
    integral = cal.integral_disk(g, 1/np.sqrt(3), [np.sqrt(11),-0.5])
'''
def integral_disk(f, r, c):
    return integral_ring(f, 0, r, c)


'''
compute the integral of a function in the region between a disk and a rectangle (the rectangle must contain the disk)
Input values 
- 'f': the function f([x, y])
- 'p_bl', 'p_rt': the bottom-left and top-right corner points of the rectangle, each is a list with two entries
- 'r': radius of the disk
- 'c' : center of the disk
Return value: 
- \int_{rectangle - disk} dx dy f

Example of usage:
    def g(x):
        return np.sin(x[0] ** 2 + np.cos(x[1] ** 2))
    integral = cal.integral_rectangle_minus_disk(g, [-1,-2], [2,3], 0.3, [1,1])
'''

def integral_rectangle_minus_disk(f, p_bl, p_tr, r, c):
    return integral_rectangle(f, p_bl, p_tr) - integral_disk(f, r, c)

# return the matrix of a rotation by an angle 'theta' about the z axis
def R_z(theta):
    return [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]


'''
given a rectangle with its bottom-left corner at the origin and a point inscribed in it, return the minimal distance between the circle center and the rectangle boundary
Input values: 
- 'L', 'h': the length and  height of the rectangle
- 'p' : the coordinates of the point
Return values: 
- the minimal distance
'''


def min_dist_c_r_rectangle(L, h, p):
    if p[0] < L / 2:
        min_x = p[0]
    else:
        min_x = L - p[0]

    if p[1] < h / 2:
        min_y = p[1]
    else:
        min_y = h - p[1]

    return min(min_x, min_y)
