from fenics import *
import importlib
import input_output as io
import numpy as np
import os
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq


import calculus as cal
import constants.utils as const
import mesh.utils as msh
import parameters.read.solution as rpam
import runtime_arguments as rarg
import switch_problem as swi

fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)


mesh_0_parameters = io.read_parameters_from_csv_file(os.path.join(rarg.args.input_directory, f'mesh_{0}', 'mesh_metadata.csv')) 

shape_coordinates = np.array(mesh_0_parameters['shape_coordinates'])
N = len(shape_coordinates)

# parameter values: uniform in [0, 1), close the curve
t_values_closed = np.arange(N) / N
t_values_closed = np.append(t_values_closed, 1.0)

shape_coordinates_closed = np.vstack([shape_coordinates, shape_coordinates[0]])

# fit a periodic cubic spline for x(t) and y(t) separately
cspline = [CubicSpline(t_values_closed, shape_coordinates_closed[:, 0], bc_type='periodic'), CubicSpline(t_values_closed, shape_coordinates_closed[:, 1], bc_type='periodic')]

'''
the curve y_s in the reference configuration and its derivative 
Input values: 
    - 's' :  the parametric coordinate 0 < s < 1
Return values; 
    - 'y_s', 'd y_s/ds': y(s) and d y(s) / ds
'''
def y_s_dy_ds(s):
    return np.array([float(cspline[0](s)), float(cspline[1](s))]), np.array([float(cspline[0](s, 1)), float(cspline[1](s, 1))])





def theta(s):
    return cal.atan_quad([ y_s_dy_ds(s)[0][0] - rmsh.parameters['c'][0], y_s_dy_ds(s)[0][1] - rmsh.parameters['c'][1] ])



# s_0 is the value of the curvilinear coordinate at which the polar angle of y(s) with respect to c is 0
theta_0 = theta(0)
# print(f'*** theta(0) = {theta_0 * const.rad_to_deg}')

# 1. Determine  s_0
# 1.1

if (abs(theta_0) < const.epsilon) or (abs(theta_0 - 2*np.pi) < const.epsilon):
    # here s_0 = 1

    theta_0 = 0

    # print(f'** Setting s_0 = 1')

    s_0 = 1

else:
    # theta_0 != 0 -> need to solve for s_0

    # print(f'** Solving for s_0 ... ')

    # 1.1 find two values of s, s_0_L and s_0_R, that bracket s_0, by looking for a pair of values of s, s_i and s_i_plus_1 such that theta(s_i_plus_1) < theta(s_i). This is done by dividing the interval 0 < s < 1 into ~ N_scan interals. Note that N_scan needs to be large enough to resolve the root. 
    for i_s in range(rpam.parameters['N_scan']+1):
        
        if theta((i_s+1)/(rpam.parameters['N_scan']-1)) < theta(i_s/(rpam.parameters['N_scan']-1)): 
            break

    s_0_L = i_s/(rpam.parameters["N_scan"]-1)
    s_0_R = (i_s+1)/(rpam.parameters["N_scan"]-1)

    # print(f's_0 lies betwee {s_0_L} and {s_0_R}')

    # 1.2 solve for s_0 by using s_0_L and s_0_R
    s_0 = brentq(lambda s:  np.arctan((y_s_dy_ds(s)[0][1] - rmsh.parameters['c'][1]) / (y_s_dy_ds(s)[0][0] - rmsh.parameters['c'][0])), a=s_0_L, b=s_0_R)

    # print(f's_0 = {s_0}\t theta(s_0) = {theta(s_0)}')

    # print(f'... done. ')
# 


# return the parameter `s` of the curve `y(s)` corresponding to a point `y` on the plane
# Input values: 
#     - `y`: the point in the reference configuration
# Return values: 
#     - `s`: the value of the parametric coordinate of the curve `y(s)` such that the point y(s) forms the same polar angle as `y` with respect to `c`

def s_y(y):

    # the polar angle that `y` forms with respect to `c``, theta_y is in [0, 2 pi]
    theta_y = cal.atan_quad([ y[0] - rmsh.parameters['c'][0], y[1] - rmsh.parameters['c'][1] ])

    if (abs(theta_y) < const.epsilon) or (abs(theta_y - 2.0 * np.pi) < const.epsilon): 
        # theta_y takes the special values 0 or 2 pi -> set directly s = s_0

        s = s_0

    else:
        # theta_y is not equal to 0 nor to 2 pi -> solve for s

        if 0 < theta_y <= theta_0:
            
            # 0 < theta_y <= theta_0: the solution s must lie in the interval s_0 < s <= 1.0. 
            

            s_L = s_0 + const.epsilon
            s_R = 1 + const.epsilon
            
            s = brentq(lambda s: theta(s) - theta_y, a=s_L, b=s_R)
                            
            # print(f'Case I\n\ts_0 = {s_0}\n\ttheta_y = {theta_y} \n\ttheta_L = {theta(s_0+const.epsilon)}\n\ttheta_R = {theta(1 + const.epsilon)}', flush=True)

        else:
            # theta_0 <= theta_y < 2 pi,  the solution s must lie in the interval 0 <= s < s_0

            if theta(0 + const.epsilon) < theta_y: 

                s_L = 0 + const.epsilon

            else:

                s_L = 0

            s_R = s_0 - const.epsilon

            s = brentq(lambda s: theta(s) - theta_y, a=s_L, b=s_R)

    # print(f'\t s = {s}')

    return s


class dyds_expression(UserExpression):
    def eval(self, values, x):

        # obtain the value of the parametric coordinate `s` corresponding to `x`
        s = s_y(x)

        values[0] = y_s_dy_ds(s)[1][0]
        values[1] = y_s_dy_ds(s)[1][1]

    def value_shape(self):
        return (2,)
    

msh.interpolate_dg(fsp.dyds, dyds_expression())


