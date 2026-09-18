import importlib
from fenics import *
import numpy as np
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import differential_geometry.manifold.geometry as geo
import mesh_quality as msh_qu
import parameters.read.solution as rpam
import phi_lb as phi_lb
import switch_problem as swi

import function_spaces as fsp
fi = importlib.import_module(swi.fi)
rmsh = importlib.import_module(swi.rmsh)

alpha = ufl.indices(1)


# this method prints out the residuals of BCs for all sectors
def print_data(step):
    
    fi.writer_data.writerows([{

        fi.fieldnames_data[0]: \
            step,\
        fi.fieldnames_data[1]: \
            f"{msh_qu.quality:.{rpam.parameters['print_out_digits']}e}",\
        fi.fieldnames_data[2]: \
            f"{np.max([rmsh.parameters['shape_coordinates'][i][1] for i in range(len(rmsh.parameters['shape_coordinates']))]):.{rpam.parameters['print_out_digits_u_max']}e}",\
        fi.fieldnames_data[3]: \
            f"{assemble(rpam.parameters['rho_fluid'] * geo.ufl_norm(fsp.U_dot_n_12) * geo.ufl_norm((fsp.X_ref[0] + fsp.U_n_12[0]).dx(0)) * rmsh.dx_mesh[1]):.{rpam.parameters['print_out_digits']}e}",\
        fi.fieldnames_data[4]: \
            f"{assemble(- rpam.parameters['rho_fluid'] * fsp.v_fl_bar[alpha] * (bgeo.facet_normal[0])[alpha] * rmsh.ds_mesh[0]['ds_b']):.{rpam.parameters['print_out_digits']}e}",\
        fi.fieldnames_data[5]: \
            f"{(assemble(rpam.parameters['rho_fluid'] * geo.ufl_norm(fsp.U_dot_n_12) * geo.ufl_norm((fsp.X_ref[0] + fsp.U_n_12[0]).dx(0)) * rmsh.dx_mesh[1]) - assemble(- rpam.parameters['rho_fluid'] * fsp.v_fl_bar[alpha] * (bgeo.facet_normal[0])[alpha] * rmsh.ds_mesh[0]['ds_b']))/(assemble(- rpam.parameters['rho_fluid'] * fsp.v_fl_bar[alpha] * (bgeo.facet_normal[0])[alpha] * rmsh.ds_mesh[0]['ds_b'])):.{rpam.parameters['print_out_digits']}e}",\
        fi.fieldnames_data[6]: \
            f"{float(phi_lb.value):.{rpam.parameters['print_out_digits']}e}"
    }])

    fi.csvfile_data.flush()

