'''
this module prints some useful data  to monitor the time iteration
'''

import importlib
from fenics import *
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import differential_geometry.manifold.geometry as geo
import parameters.read.solution as rpam
import switch_problem as swi

import files as fi
import function_spaces as fsp

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)



i, j, k = ufl.indices(3)


def print_data(step):

     # I estimate sigma_n from sigma_n_12 by adding the estimated increment across dt/2
    sigma_n = fsp.sigma_n_12 + (fsp.sigma_n_12 - fsp.sigma_n_32)/2.0
    
    dTdt_L = assemble( ( \
                    rpam.parameters['rho']/2.0 * (fsp.v_n[i] * geo.g(fsp.omega_n_12)[i, j] * fsp.v_n[j] - fsp.v_n_1[i] * geo.g(fsp.omega_n_12)[i, j] * fsp.v_n_1[j])/vp.dt \
                    + fsp.v_n[i] * ( rpam.parameters['rho']/2.0 * fsp.v_n[j] * geo.g(fsp.omega_n_12)[j, k] * fsp.v_n[k] ).dx(i)
                ) * geo.sqrt_detg( fsp.omega_n_12 ) * rmsh.dx)
    
    dTdt_R = (assemble(  \
                                bgeo.n_lr( fsp.omega_n_12 )[i] * fsp.v_n[j] * ( sigma_n * geo.g(fsp.omega_n_12)[i, j] + 2 * rpam.parameters['mu'] * geo.d(fsp.v_n, fsp.w_n, fsp.omega_n_12 )[i, j] )  * bgeo.sqrt_deth_lr( fsp.omega_n_12 ) * rmsh.ds_lr \
                                + bgeo.n_tb( fsp.omega_n_12 )[i] * fsp.v_n[j] * ( sigma_n * geo.g(fsp.omega_n_12)[i, j] + 2 * rpam.parameters['mu'] * geo.d(fsp.v_n, fsp.w_n, fsp.omega_n_12 )[i, j] )  * bgeo.sqrt_deth_tb( fsp.omega_n_12 ) * rmsh.ds_tb \
                                + bgeo.n_circle( fsp.omega_n_12 )[i] * fsp.v_n[j] * ( sigma_n * geo.g(fsp.omega_n_12)[i, j] + 2 * rpam.parameters['mu'] * geo.d(fsp.v_n, fsp.w_n, fsp.omega_n_12 )[i, j] )  * bgeo.sqrt_deth_circle( fsp.omega_n_12, rmsh.parameters["c_r"] ) * ( 1.0 / rmsh.parameters["r"]) * rmsh.ds_circle \
                            ) \
                            - assemble( ( 2.0 * rpam.parameters['mu'] * geo.d(fsp.v_n, fsp.w_n, fsp.omega_n_12)[j, i] * geo.d_c(fsp.v_n, fsp.w_n, fsp.omega_n_12)[i, j] ) * geo.sqrt_detg( fsp.omega_n_12 ) * rmsh.dx))

    fi.writer_data.writerows([{
        fi.fieldnames_data[0]: \
            f"{step:.{rpam.parameters['print_out_digits']}e}",\
        fi.fieldnames_data[1]: \
            f'{sqrt(assemble((fsp.v_bar[i] * geo.g( fsp.omega_n_12_n_12 )[i, j] * (bgeo.n_circle( fsp.omega_n_12_n_12 ))[j])**2 * bgeo.sqrt_deth_circle( fsp.omega_n_12_n_12, rmsh.parameters["c_r"] ) * (1.0 / rmsh.parameters["r"]) * rmsh.ds_circle)):.{rpam.parameters["print_out_digits"]}e}', \
        fi.fieldnames_data[2]: \
            f'{sqrt(assemble( (geo.Nabla_v( fsp.v_n, fsp.omega_n_12_n_12 )[i, i] - 2 * fsp.mu_n_12 * fsp.w_n_n )**2 * geo.sqrt_detg( fsp.omega_n_12_n_12 ) * rmsh.dx)):.{rpam.parameters["print_out_digits"]}e}', \
        fi.fieldnames_data[3]: \
            f'{sqrt(assemble( 1 * geo.sqrt_detg( fsp.omega_n_12_n_12 ) * rmsh.dx)):.{rpam.parameters["print_out_digits"]}e}'   
        }])

    fi.csvfile_data.flush()
