from fenics import *
import importlib
import numpy as np
from scipy.interpolate import CubicSpline

import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

shape_coordinates = np.array(rmsh.parameters['shape_coordinates'])

'''
compute the arc length along the 1d mesh: arc_length_tab[i] = [cumulative arc length along the 1d mesh curve obtained from its beginning until shape_coordinates[i] included]
'''
arc_length = 0
arc_length_tab = [0]
for i in range(1, len(shape_coordinates)):

    arc_length += np.linalg.norm(np.subtract(shape_coordinates[i], shape_coordinates[i-1]))
    arc_length_tab.append(arc_length)



mesh_len = sum(c.volume() for c in cells(rmsh.lmsh.mesh[1]))
spline_end =  arc_length_tab[-1]
print(f'*** check : {mesh_len - spline_end}')


# fit a periodic cubic spline for x(t) and y(t) separately
cspline = [CubicSpline(arc_length_tab, shape_coordinates[:, 0], bc_type='natural'), CubicSpline(arc_length_tab, shape_coordinates[:, 1], bc_type='natural')]

'''
the curve X_ref  and its derivative 
Input values: 
    - 's' : the parametric coordinate (arc length)
Return values; 
    - 'X_ref(s)', 'd X_ref/ds': X_ref(s) and d X_ref(s) / ds
'''
def X_ref_s_dXref_ds(s):
    return np.array([float(cspline[0](s)), float(cspline[1](s))]), np.array([float(cspline[0](s, 1)), float(cspline[1](s, 1))])



