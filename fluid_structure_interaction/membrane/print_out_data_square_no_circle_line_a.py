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

    dMdt_t = assemble(rpam.parameters["rho_fluid"] * fsp.w_n * geo.ufl_norm((fsp.X_ref + fsp.U_n_12).dx(0)) * rmsh.dx_mesh[1])
    dMdt_b = assemble(rpam.parameters["rho_fluid"] * fsp.v_fl_bar_b[alpha] * (bgeo.facet_normal[0])[alpha] * rmsh.ds_mesh[0]["ds_b"])

    # maximal value of the y component of X_cur
    _, _, _, _, _, U_n_12_output, _, _, _ = fsp.psi_mem.split( deepcopy=True )
    X_cur_y_max = np.max([(fsp.X_ref(fsp.U_n_12_coordinates[i])[1] + U_n_12_output(fsp.U_n_12_coordinates[i])[1]) for i in range(len(fsp.U_n_12_coordinates))])


    fi.writer_data.writerows([{

        fi.fieldnames_data[0]: \
            step,\
        fi.fieldnames_data[1]: \
            f"{msh_qu.quality:.{rpam.parameters['print_out_digits']}e}",\
        fi.fieldnames_data[2]: \
            f"{X_cur_y_max:.{rpam.parameters['print_out_digits_u_max']}e}",\
        fi.fieldnames_data[3]: \
            f"{dMdt_t:.{rpam.parameters['print_out_digits']}e}",\
        fi.fieldnames_data[4]: \
            f"{dMdt_b:.{rpam.parameters['print_out_digits']}e}",\
        fi.fieldnames_data[5]: \
            f"{(dMdt_t + dMdt_b)/dMdt_b:.{rpam.parameters['print_out_digits']}e}",\
        fi.fieldnames_data[6]: \
            f"{float(phi_lb.value):.{rpam.parameters['print_out_digits']}e}"
    }])

    fi.csvfile_data.flush()

