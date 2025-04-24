import numpy as np
import scipy.integrate as spi

small_number = 1e-3


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
    return circle_arc(r, cr, 0, 2*np.pi, t)


'''
a circle arc
Input values:
- 'r': the circle radius
- 'c_r': the circle center (an array of two points)
- 'theta_min', 'theta_max': the minimal and maxmimal values of the polar angles of the arg, repsectively
- 't' : the parameteric coordinate of the circle, 0<=t<1
Return values:
- the curve position and derivative: [x[0](t), x[1](t)], [x[0]'(t), x[1]'(t)]
'''


def circle_arc(r, cr, theta_min, theta_max, t):
    theta_t = theta_min + (theta_max - theta_min) * t

    return [np.add(cr, r * np.array([np.cos(theta_t), np.sin(theta_t)])).tolist(),
            (r * (theta_max - theta_min) * np.array([- np.sin(theta_t), np.cos(theta_t)])).tolist()]

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
    integral_line_test = cal.curve_integral(g, line_test)
    print(f'integral_line_test: {integral_line_test}')
    
Example of usage:
    circle_test = lambda t: cal.circle(1.34, [np.sqrt(2), -np.sqrt(3)],  t)
    def g(x):
        return np.sin( x[0]**2 +np.cos( x[1]**2))
    integral_line_test = cal.curve_integral(g, circle_test)
'''


def curve_integral(f, gamma_dgamma):
    integral = spi.quad(lambda t: (f(gamma_dgamma(t)[0]) * np.linalg.norm((gamma_dgamma(t))[1])), 0, 1)[0]
    return integral


'''
return the curve integral of a function  along a line 
Input values:
- 'f': the function f(x[0], x[1])
- 'x_a', 'x_b': the start and end points of the line
Return values: 
\int_line f dl

Example of usage:
    def g(x):
        return np.sin(x[0] ** 2 + np.cos(x[1] ** 2))
    
    integral_line = cal.curve_integral_line(g, [1,2],[4,3])
'''


def curve_integral_line(f, x_a, x_b):
    line_curve = lambda t: line(x_a, x_b, t)
    return curve_integral(f, line_curve)


'''
return the curve integral of a function  along a circle 
Input values:
- 'f': the function f(x[0], x[1])
- 'r': the circle radius
- 'c': the circle center (an array of two points)
Return values: 
\int_circle f dl

Example of usage:
    def g(x):
        return np.sin(x[0] ** 2 + np.cos(x[1] ** 2))
    
    integral_circle = cal.curve_integral_circle(g, 1, [1,np.sqrt(2)])
'''


def curve_integral_circle(f, r, c):
    circle_curve = lambda t: circle(r, c, t)
    return curve_integral(f, circle_curve)


'''
return the curve integral of a function  along a circle arc
Input values:
- 'f': the function f(x[0], x[1])
- 'r': the circle radius
- 'theta_min', 'theta_max': min and max values of the polar angles of the arc, repsectively
- 'c': the circle-arc center (an array of two points)
Return values: 
\int_{circle arc} f dl

'''

def curve_integral_circle_arc(f, r, theta_min, theta_max, c):
    circle_arc_curve = lambda t: circle_arc(r, c, theta_min, theta_max, t)
    return curve_integral(f, circle_arc_curve)

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
    integral = surface_integral_rectangle(g, [-2,0.1], [1,1])
'''


def surface_integral_rectangle(f, p_bl, p_tr):
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
    integral = cal.surface_integral_ring(g, 1/np.sqrt(3), 2, [np.sqrt(11),-0.5])
'''


def surface_integral_ring(f, r, R, c):
    return surface_integral_ring_slice(f, r, R, 0, 2 * np.pi, c)



'''
integate a function of two variables over the slice of a ring delimited by two concentric circles
Input values 
- 'f': the function f([x, y])
- 'r', 'R': radii of the inner and outer circle defining the ring
- 'theta_min', 'theta_max': the polar angles delimiting the ring slice
- 'c' : center of the circles (a list of two values)
Result:
- \int_{ring slice} dx dy f
'''


def surface_integral_ring_slice(f, r, R, theta_min, theta_max, c):
    f_swapped = lambda x, y: f([y, x])

    return spi.dblquad(lambda rho, theta: rho * f_swapped(c[1] + rho * np.sin(theta), c[0] + rho * np.cos(theta)), theta_min, theta_max, lambda rho: r, lambda rho: R)[0]


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
    integral = cal.surface_integral_dsk(g, 1/np.sqrt(3), [np.sqrt(11),-0.5])
'''


def surface_integral_disk(f, r, c):
    return surface_integral_ring(f, 0, r, c)


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
    integral = cal.surface_integral_integral_rectangle_minus_disk(g, [-1,-2], [2,3], 0.3, [1,1])
'''


def surface_integral_rectangle_minus_disk(f, p_bl, p_tr, r, c):
    return surface_integral_rectangle(f, p_bl, p_tr) - surface_integral_disk(f, r, c)


# return the matrix of a rotation by an angle 'theta' about the z axis
def R_z(theta):
    return [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]


'''
A rotation matrix in two dimensions
Input values: 
- 'theta': the rotation angle, in radians
Return values: 
- the rotation matrix
'''


def R(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


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


'''
checks whether a point lies on a line
Input values: 
- 'point': the coordinates of the point ( a tuple of two values)
- 'line': the parametric form of the line, as an output of cal.line
Return value:
- True (False) if 'point' lies on 'line' within accuracy 'small_number'

Example of usage:
gamma_top = lambda t: cal.line(r_2, r_3, t)
print(f'r_1 is on gamma_top: {cal.point_on_line(np.add(r_2, r_3), gamma_top)}')
'''


def point_on_line(point, line):
    p_start = (line(0))[0]
    delta_p = np.subtract((line(1))[0], p_start).tolist()

    num = (p_start[1] - point[1]) * delta_p[0] - (p_start[0] - point[0]) * delta_p[1]
    den = np.linalg.norm(delta_p)

    return np.isclose(num / den, 0, rtol=small_number)


'''
mirrors a point with respect to the symmetry axis given by a line
Input values: 
- 'point': the coordinates of the point ( a list with two entries)
- 'line': the parametric form of the line, as an output of cal.line
Return value:
- the mirrored point (a list with two entries) 

Example of usage:
gamma = lambda t: cal.line([0, 1/2], [1,1/2], t)
mirrored_point = cal.mirror_point_line([1/2,1], gamma)
'''


def mirror_point_line(point, line):
    p_start = (line(0))[0]
    p_end = (line(1))[0]
    delta = np.subtract(p_end, p_start)
    denominator = (np.linalg.norm(delta)) ** 2

    result = [-point[0] + (2 * (point[0] * delta[0] ** 2 + delta[1] * (-p_start[1] * delta[0] + point[1] * delta[0] + p_start[0] * delta[1]))) / denominator, point[1] + (2 * delta[0] * (p_start[1] * delta[0] - point[1] * delta[0] + (-p_start[0] + point[0]) * delta[1])) / denominator, 0]

    return result

'''
tells whether a line lies on an axis
Input values: 
- 'line': a line in a mesh
- 'gamma_axis': the parametric form of the line, as an output of cal.line
- 'mesh': the mesh
Return value:
- True (False) if 'line' lies (does not lie) on 'gamma_axis'

Example of usage:

for j in range(len(mesh.cells)):
    if mesh.cells[j].type == 'line':
        lines = np.copy(mesh.cells[j].data)
        for i in range(np.shape(lines)[0]):
            if (not cal.line_on_axis(lines[i], gamma_axis_of_symmetry, mesh)):
[...]
'''
def line_on_axis(line, gamma_axis, mesh):

    print(f'Calling line_on_axis with line = {line}')

    line_vertex_on_axis = [(point_on_line(mesh.points[line[k]], gamma_axis)) for k in range(len(line))]
    return (line_vertex_on_axis[0] and line_vertex_on_axis[1])




'''
given a ring mesh and multiple radial lines which start from the origin, check is a line lies on one of these radial lines
Input values:
- 'line': a line in the mesh, of type '<class 'numpy.ndarray'>'
- 'N': the number of radial lines: each line has polar angle theta = 2 \pi/N * i with i = 0, ..., N-1
- 'mesh': the mesh, a <meshio mesh object>
Return values: 
- True/False if 'line' lies on at least one of the 'N' radial lines 
'''
def line_is_radial(line_to_check, N, mesh):

    # the angular size of each slice delimited by the radial lines
    theta = 2 * np.pi / N

    is_radial = False

    for i in range(0, N):
        # loop through the radial lines

        # construct an axis given by the radial line under consideration
        point_O =[0,0]
        point_r = R(i * theta).dot([1, 0])
        radial_axis = lambda t: line(point_O, point_r, t)

        # check whether 'line_to_check' lies on the axis
        is_radial = line_on_axis(line_to_check, radial_axis, mesh)

        # if 'line_to_check' lies on the axis, stop
        if is_radial:
            break

    return is_radial