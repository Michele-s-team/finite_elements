from fenics import *
import importlib
import numpy as np
from scipy.interpolate import CubicSpline

import switch_problem as swi

import function_spaces as fsp

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


# reference configuration of the manifold, a straight line which coincides with the mesh line
class X_ref_Expression(UserExpression):
    def eval(self, values, x):

        X, _ = X_ref_s_dXref_ds(x[0])

        values[0] = X[0]
        values[1] = X[1]

    def value_shape(self):
        return (2,)

# interpolate `X_ref` according to the analytical expression `X_ref_Expression`
fsp.X_ref.interpolate(X_ref_Expression(element=fsp.Q_X.ufl_element()))


mesh_len = sum(c.volume() for c in cells(fsp.Q_X.mesh()))
spline_end =  arc_length_tab[-1]
print(f'*** \n\t s_max = {arc_length_tab[-1]}\n\tcheck : {mesh_len - spline_end} \n\t coordinate[-1][0] - 1 = {shape_coordinates[-1][0]-1}\n\tX_ref_s_dXref_ds(s_max)[0][0]-1 = {X_ref_s_dXref_ds(arc_length_tab[-1])[0][0]-1}\n\tX_ref(s_max) = {fsp.X_ref(arc_length_tab[-1])}')
