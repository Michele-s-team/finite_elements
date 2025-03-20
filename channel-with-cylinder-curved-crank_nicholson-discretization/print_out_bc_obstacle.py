from fenics import *
from mshr import *
import ufl as ufl
import csv

import boundary_geometry as bgeo
import files as fi
import function_spaces as fsp
import geometry as geo
import input_output as io
import read_mesh_bc_obstacle as rmsh
import runtime_arguments as rarg
import variational_problem_bc_obstacle as vp

i, j, k, l = ufl.indices( 4 )


csvfile = open( (rarg.args.output_directory) + '/bcs.csv', 'a', newline='' )
fieldnames = [ \
    '<<(l_profile_v_bar^i - v_bar^i)(l_profile_v_bar_i - v_bar_i)>>_{l + t + b + circle}',\
    '<<(phi - r_profile_phi)^2>>_r' ,\
    '<<(n^i Nabla_i phi)^2>>_{l + t + b + circle}' \
    ]
writer = csv.DictWriter( csvfile, fieldnames=fieldnames )
writer.writeheader()


# this function prints out the residuals of BCs
def print_bcs():
    # get the solution and write it to file

    # write the residual of natural BCs on step 2 to file
    writer.writerows( [{ \
        fieldnames[0]: \
            (sqrt( assemble( (fsp.v_[i] - vp.v__profile_l[i]) * geo.g( fsp.omega )[i, j] * (fsp.v_[j] - vp.v__profile_l[j]) * rmsh.ds_l ) + assemble( fsp.v_[i] * geo.g( fsp.omega )[i, j] * fsp.v_[j] * (rmsh.ds_t + rmsh.ds_b + rmsh.ds_circle) ) ) / \
             assemble( Constant( 1.0 ) * (rmsh.ds_l + rmsh.ds_t + rmsh.ds_b + rmsh.ds_circle) )), \
        fieldnames[1]: \
            sqrt( (assemble( ((bgeo.n_lr( fsp.omega ))[i] * (fsp.phi.dx( i ))) ** 2 * rmsh.ds_l ) \
                   + assemble( ((bgeo.n_tb( fsp.omega ))[i] * (fsp.phi.dx( i ))) ** 2 * (rmsh.ds_t + rmsh.ds_b) ) \
                   + assemble( ((bgeo.n_circle( fsp.omega ))[i] * (fsp.phi.dx( i ))) ** 2 * rmsh.ds_circle )) \
                  / assemble( Constant( 1.0 ) * (rmsh.ds_l + rmsh.ds_t + rmsh.ds_b + rmsh.ds_circle) ) ), \
        fieldnames[2]: \
            sqrt( assemble( (fsp.phi) ** 2 * rmsh.ds_r ) /
                  assemble( Constant( 1.0 ) * rmsh.ds_r ) ) \
        }] )

    csvfile.flush()
