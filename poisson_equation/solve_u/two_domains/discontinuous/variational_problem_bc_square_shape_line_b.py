from fenics import *
import importlib
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import mesh.utils as msh
import parameters.read.solution as rpam
import switch_problem as swi

fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)

i, j = ufl.indices(2)


# test case 1

def f_shape_expression(x):
    return 6.0 + rpam.parameters['A'] * 4.0

def f_square_expression(x):
    return 6.0

def u_exact_shape_expression(x):
   return 1 + x[0] ** 2 + 2 * x[1] ** 2 + rpam.parameters['A'] * ((x[0]-rmsh.lmsh.parameters['c'][0])**2 + (x[1]-rmsh.lmsh.parameters['c'][1])**2 - rmsh.lmsh.parameters['r']**2) + rpam.parameters['B']

def u_exact_square_expression(x):
   return 1 + x[0] ** 2 + 2 * x[1] ** 2

def d_expression(x):
    return rpam.parameters['A'] * 2.0 * rmsh.lmsh.parameters['r']


msh.interpolate_dg(fsp.u_exact, u_exact_shape_expression, rmsh.sf[0], rmsh.lmsh.parameters['sub_mesh_0_0_id'])
msh.interpolate_dg(fsp.u_exact, u_exact_square_expression, rmsh.sf[0], rmsh.lmsh.parameters['sub_mesh_0_1_id'])

msh.interpolate_dg(fsp.f, f_shape_expression, rmsh.sf[0], rmsh.lmsh.parameters['sub_mesh_0_0_id'])
msh.interpolate_dg(fsp.f, f_square_expression, rmsh.sf[0], rmsh.lmsh.parameters['sub_mesh_0_1_id'])

msh.interpolate_dg(fsp.d, d_expression, rmsh.sf[0])


import input_output as io
import os
import solution_paths as solpath

io.print_scalar_to_csvfile(fsp.u_exact, os.path.join(solpath.csv_files_path, 'u_exact.csv'), rmsh.lmsh.sf[0])




bcs = []


# variational functional for the original problem (poisson equation)
F_0 =   (fsp.u.dx(i) * fsp.nu_u.dx(i)) * rmsh.dx_mesh[0]['dx'] + \
        (fsp.f * fsp.nu_u) * rmsh.dx_mesh[0]['dx'] + \
        - bgeo.facet_normal[0][i] * (fsp.u.dx(i)) * fsp.nu_u * rmsh.ds_mesh[0]['ds']

# here I put the average for d because d is the same on both sides (it is a jump)
F_a = - (msh.average(fsp.d)* msh.average(fsp.nu_u)) * rmsh.ds_mesh[0]['dS_shape']

F_I = (
        - msh.average(fsp.u.dx(i)) * msh.jump(fsp.nu_u, bgeo.facet_normal[0])[i]
        ) * rmsh.ds_mesh[0]['ds_I'] + \
        rpam.parameters['alpha']/rmsh.r_mesh[0] * (\
            ( msh.jump(fsp.u, bgeo.facet_normal[0])[i] * msh.jump(fsp.nu_u, bgeo.facet_normal[0])[i] ) * (rmsh.ds_mesh[0]['dS_I_shape'] + rmsh.ds_mesh[0]['dS_I_square']) + \
            ( msh.jump(fsp.u, bgeo.facet_normal[0])[i]  - msh.jump(fsp.u_exact, bgeo.facet_normal[0])[i] ) *  msh.jump(fsp.nu_u, bgeo.facet_normal[0])[i] * rmsh.ds_mesh[0]['dS_shape']
            )

F_b =   rpam.parameters['alpha']/rmsh.r_mesh[0] * (fsp.u - fsp.u_exact) * fsp.nu_u * rmsh.ds_mesh[0]['ds']


F = F_0 + F_I + F_a + F_b
