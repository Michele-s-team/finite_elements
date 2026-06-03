import csv
import importlib
import numpy as np
import os
from scipy.interpolate import CubicSpline


import runtime_arguments as rarg
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)



shape_coordinates = np.array(rmsh.parameters['shape_coordinates'])
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






