import importlib
import numpy as np

import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

'''
the curve y_s in the reference configuration and its derivative 
Input values: 
    - 's' :  the parametric coordinate 0 < s < 1
Return values; 
    - 'y_s', 'd y_s/ds': y(s) and d y(s) / ds
'''
def y_s_dy_ds(s):

    return np.add(rmsh.parameters['c'], [
               rmsh.parameters['a'] * np.cos(2.0 * np.pi * s) * np.sin(np.pi / 8.0) / (2.0 + np.cos(2.0 * np.pi * s)) + rmsh.parameters['b'] * np.cos(np.pi / 8.0) * np.sin(2.0 * np.pi * s),
               - rmsh.parameters['a'] * np.cos(np.pi / 8.0) * np.cos(2.0 * np.pi * s) / (2.0 + np.cos(2.0 * np.pi * s)) + rmsh.parameters['b'] * np.sin(np.pi / 8.0) * np.sin(2.0 * np.pi * s)
           ]), \
           np.array([
               2.0 * np.pi * (rmsh.parameters['b'] * np.cos(np.pi / 8.0) * np.cos(2.0 * np.pi * s) - 2.0 * rmsh.parameters['a'] * np.sin(np.pi / 8.0) * np.sin(2.0 * np.pi * s) / (2.0 + np.cos(2.0 * np.pi * s))**2),
               2.0 * rmsh.parameters['b'] * np.pi * np.cos(2.0 * np.pi * s) * np.sin(np.pi / 8.0) + np.sqrt(2.0) * rmsh.parameters['a'] * np.pi * np.sin(2.0 * np.pi * s) / (np.sin(np.pi / 8.0) * (2.0 + np.cos(2.0 * np.pi * s))**2)
           ])

