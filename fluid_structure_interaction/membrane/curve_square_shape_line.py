from fenics import *
import importlib
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

curve_coordinates = np.array(rmsh.parameters['curve_coordinates'])

'''
compute the arc length along the sub mesh: arc_length_tab[i] = [cumulative arc length along the sub mesh curve obtained from its beginning until curve_coordinates[i] included]
'''
arc_length = 0
arc_length_tab = [0]
for i in range(1, len(curve_coordinates)):

    arc_length += np.linalg.norm(np.subtract(curve_coordinates[i], curve_coordinates[i-1]))
    arc_length_tab.append(arc_length)


# fit a periodic cubic spline for x(t) and y(t) separately
cspline = [CubicSpline(arc_length_tab, curve_coordinates[:, 0], bc_type='natural'), CubicSpline(arc_length_tab, curve_coordinates[:, 1], bc_type='natural')]

'''
the curve X_ref  and its derivative 
Input values: 
    - 's' : the parametric coordinate (arc length)
Return values; 
    - 'X_ref(s)', 'd X_ref/ds': X_ref(s) and d X_ref(s) / ds
'''
def X_ref_s_dXref_ds(s):
    return np.array([float(cspline[0](s)), float(cspline[1](s))]), np.array([float(cspline[0](s, 1)), float(cspline[1](s, 1))])



