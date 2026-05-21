'''
this module solves for u_0
'''

from fenics import *
import importlib
import ufl as ufl


import decompose_u as dec_u
import differential_geometry.boundary.geometry as bgeo
import mesh.utils as msh
import physics.fluid_mechanics as flu
import physics.elasticity as ela
import parameters.read.solution as rpam
import switch_problem as swi


fsp = importlib.import_module(swi.fsp)
rmsh = importlib.import_module(swi.rmsh)

i, j, k, l, m = ufl.indices(5)


sub_mesh_0_label, sub_mesh_1_label = msh.plus_minus(rmsh.lmsh.mesh[0], rmsh.sf[0], rmsh.lmsh.parameters["sub_mesh_0_0_id"], rmsh.lmsh.parameters["sub_mesh_0_1_id"], rmsh.ds_mesh[0]['dS_shape'])





bcs = []

# variational problem


F_u_0 = msh.ufl_conditional_form(
                                        rmsh.lmsh.mesh[0],
                                        rmsh.sf[0], 
                                        (fsp.u_0[i] - (fsp.phi_0[i] - fsp.y[i])) * fsp.nu_u_0[i], 
                                        - ela.P(fsp.u_0, ela.K(fsp.u_0, rpam.parameters['exponent']), ela.mu(fsp.u_0, rpam.parameters['exponent']))[k, i] * (fsp.nu_u_0[k].dx(i)), 
                                        rmsh.lmsh.parameters['sub_mesh_0_0_id'],
                                        rmsh.lmsh.parameters['sub_mesh_0_1_id']
                                ) * rmsh.dx_mesh[0]['dx'] \
        + (\
            msh.jump(fsp.nu_u_0[k], bgeo.facet_normal[0])[i] * msh.average( ela.P(fsp.u_0, ela.K(fsp.u_0, rpam.parameters['exponent']), ela.mu(fsp.u_0, rpam.parameters['exponent']))[k, i] )   
        ) * rmsh.ds_mesh[0]['dS_I_square'] \
        + bgeo.facet_normal[0][i] * ela.P(fsp.u_0, ela.K(fsp.u_0, rpam.parameters['exponent']), ela.mu(fsp.u_0, rpam.parameters['exponent']))[k, i] * fsp.nu_u_0[k] * rmsh.ds_mesh[0]['ds'] \
        + bgeo.facet_normal[0](sub_mesh_1_label)[i] * ela.P(fsp.u_0(sub_mesh_1_label), ela.K(fsp.u_0(sub_mesh_1_label), rpam.parameters['exponent']), ela.mu(fsp.u_0(sub_mesh_1_label), rpam.parameters['exponent']))[k, i] * fsp.nu_u_0(sub_mesh_1_label)[k] * rmsh.ds_mesh[0]['dS_shape'] \
        + rpam.parameters['alpha']/rmsh.r_mesh[0] * (\
            msh.jump(fsp.u_0[i], bgeo.facet_normal[0])[j] * msh.jump(fsp.nu_u_0[i], bgeo.facet_normal[0])[j] * rmsh.ds_mesh[0]['dS_I_square'] \
            + fsp.u_0[i] * fsp.nu_u_0[i] * rmsh.ds_mesh[0]['ds'] \
        ) \
        + rpam.parameters['alpha_ellipse']/rmsh.r_mesh[0] * (\
            msh.jump(fsp.u_0[i], bgeo.facet_normal[0])[j] * msh.jump(fsp.nu_u_0[i], bgeo.facet_normal[0])[j] * rmsh.ds_mesh[0]['dS_shape'] \
        ) \



