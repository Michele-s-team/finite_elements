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

    x = rmsh.parameters['a'] * np.cos(2.0 * np.pi * s) / (2.0 + np.cos(2.0 * np.pi * s))
    y = rmsh.parameters['b'] * np.sin(2.0 * np.pi * s)

    dx = -4.0 * np.pi * rmsh.parameters['a'] * np.sin(2.0 * np.pi * s) / (2.0 + np.cos(2.0 * np.pi * s))**2
    dy = 2.0 * np.pi * rmsh.parameters['b'] * np.cos(2.0 * np.pi * s)
    
    return np.add(rmsh.parameters['c'], [(x - y) / np.sqrt(2.0), (x + y) / np.sqrt(2.0)]), \
           np.array([(dx - dy) / np.sqrt(2.0), (dx + dy) / np.sqrt(2.0)])

