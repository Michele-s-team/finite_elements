from fenics import *
import importlib
import ufl as ufl


import command as cmd
import differential_geometry.boundary.geometry as bgeo
import differential_geometry.manifold.geometry as geo
import function_spaces as fsp
import mesh.load as lmsh
import parameters.read.solution as rpam
import switch_problem as swi

cu = importlib.import_module(swi.cu)
rmsh = importlib.import_module(swi.rmsh)

cmd.set_gauge('arc_length')


i, j = ufl.indices( 2 )


bcs_f_el = []

F_f_el = (
            fsp.f_el *  fsp.nu_f_el \
            - rpam.parameters['kappa'] * (
                2.0 * geo.g_c( fsp.psi_n_12, fsp.nu_n_12 )[i, j] * ((fsp.mu_n_12).dx(i)) * (fsp.nu_f_el.dx(j)) \
                - 4.0 * fsp.mu_n_12 * (((fsp.mu_n_12) ** 2) - geo.K( fsp.psi_n_12, fsp.nu_n_12 )) * fsp.nu_f_el
            )
        ) * geo.sqrt_detg( fsp.psi_n_12, fsp.nu_n_12 ) * rmsh.dx_mesh[1] \
        + 2.0 * rpam.parameters['kappa'] * ( \
                      ( (bgeo.n_lr( fsp.psi_n_12, fsp.nu_n_12, lmsh.mesh[1]))[i] * ((fsp.mu_n_12).dx( i )) * fsp.nu_f_el) * bgeo.sqrt_deth_lr( fsp.psi_n_12 ) * rmsh.ds_mesh[1]['ds'] \
          )





