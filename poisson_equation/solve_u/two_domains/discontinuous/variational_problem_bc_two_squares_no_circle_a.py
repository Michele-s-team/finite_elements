'''
here 'a' anb 'b' refer to the two domains
    - 'a' is the label for the left square
    - 'b' is the label for the right square
'''


from fenics import *
import importlib
import numpy as np
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import mesh.utils as msh
import parameters.read.solution as rpam
import switch_problem as swi

fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)

i, j = ufl.indices(2)

'''
# test interpolate_dg - start
import input_output as io
import solution_paths as solpath
import sys

def g_l(x):
    return np.cos(2*np.pi*(x[0]+x[1]))

def g_r(x):
    return np.sin(4*np.pi*(x[0]-x[1]))

msh.interpolate_dg(fsp.u, g_l, rmsh.sf, rmsh.lmsh.parameters['l_surface_id'])
msh.interpolate_dg(fsp.u, g_r, rmsh.sf, rmsh.lmsh.parameters['r_surface_id'])


io.full_print(fsp.u, 'u', solpath.xdmf_file_path, solpath.h5_file_path, solpath.csv_files_path,
              solpath.nodal_values_path)

sys.exit(1)
# test interpolate_dg - end
'''



'''
# test case 1
def u_exact_l_expression(x):
   return 1 + x[0] ** 2 + 2 * x[1] ** 2 + rpam.parameters['A'] * (x[0]-rmsh.lmsh.parameters['L_m'])

def u_exact_r_expression(x):
   return 1 + x[0] ** 2 + 2 * x[1] ** 2

def f_l_expression(x): 
    return 6.0

def f_r_expression(x):
    return 6.0

def d_expression(x):
    return rpam.parameters['A']
'''

'''
# test case 2
def u_exact_l_expression(x):
   return  1 + x[0] ** 2 + 2 * x[1] ** 4 + rpam.parameters['A'] * (x[0]-rmsh.lmsh.parameters['L_m'])

def u_exact_r_expression(x):
   return 1 + x[0] ** 2 + 2 * x[1] ** 4 

def f_l_expression(x): 
    return 2.0 + 24.0 * x[1]**2

def f_r_expression(x):
    return f_l_expression(x)

def d_expression(x):
    return rpam.parameters['A']
'''


# test case 3

def f_l_expression(x):
    return 2.0 + 24.0 * x[1]**2 + 2 * rpam.parameters['A'] * np.pi**2/ rmsh.lmsh.parameters['L']**2 * (13 * np.sin((2 * np.pi * (rmsh.lmsh.parameters['L_m'] - 3 * x[0] - 2 * x[1])) / rmsh.lmsh.parameters['L']) +  5 * np.sin((2 * np.pi * (rmsh.lmsh.parameters['L_m'] + x[0] + 2 * x[1])) / rmsh.lmsh.parameters['L']) )

def f_r_expression(x):
    return 2.0 + 24.0 * x[1]**2

def u_exact_l_expression(x):
   return 1 + x[0] ** 2 + 2 * x[1] ** 4 + rpam.parameters['A'] * np.sin(2 * np.pi / rmsh.lmsh.parameters['L'] * (x[0] - rmsh.lmsh.parameters['L_m'])) * np.cos(4 * np.pi * (x[0] + x[1]) / rmsh.lmsh.parameters['L'])

def u_exact_r_expression(x):
   return 1 + x[0] ** 2 + 2 * x[1] ** 4 

def d_expression(x):
    return (2 * rpam.parameters['A'] * np.pi * np.cos((4 * np.pi * (rmsh.lmsh.parameters['L_m'] + x[1])) / rmsh.lmsh.parameters['L'])) / rmsh.lmsh.parameters['L']



msh.interpolate_dg(fsp.u_exact, u_exact_l_expression, rmsh.sf, rmsh.lmsh.parameters['l_surface_id'])
msh.interpolate_dg(fsp.u_exact, u_exact_r_expression, rmsh.sf, rmsh.lmsh.parameters['r_surface_id'])

msh.interpolate_dg(fsp.f, f_l_expression, rmsh.sf, rmsh.lmsh.parameters['l_surface_id'])
msh.interpolate_dg(fsp.f, f_r_expression, rmsh.sf, rmsh.lmsh.parameters['r_surface_id'])

msh.interpolate_dg(fsp.d, d_expression, rmsh.sf)


bcs = []


# variational functional for the original problem (poisson equation)
F_0 =   (fsp.u.dx(i) * fsp.nu_u.dx(i)) * rmsh.dx + \
        (fsp.f * fsp.nu_u) * rmsh.dx + \
        - bgeo.facet_normal[i] * (fsp.u.dx(i)) * fsp.nu_u * rmsh.ds

# here I put the average for d because d is the same on both sides (it is a jump)
F_a = - (msh.average(fsp.d)* msh.average(fsp.nu_u)) * rmsh.dS_m

F_I = (
        - msh.average(fsp.u.dx(i)) * msh.jump(fsp.nu_u, bgeo.facet_normal)[i] + \
        rpam.parameters['alpha']/rmsh.r_mesh * ( msh.jump(fsp.u, bgeo.facet_normal)[i] * msh.jump(fsp.nu_u, bgeo.facet_normal)[i] )
        ) * rmsh.dS

F_b =   rpam.parameters['alpha']/rmsh.r_mesh * (\
            (fsp.u - fsp.u_exact) * fsp.nu_u * (rmsh.ds_l + rmsh.ds_lt + rmsh.ds_lb) + \
            (fsp.u - fsp.u_exact) * fsp.nu_u * (rmsh.ds_r + rmsh.ds_rt + rmsh.ds_rb)
        )


F = F_0 + F_I + F_a + F_b
