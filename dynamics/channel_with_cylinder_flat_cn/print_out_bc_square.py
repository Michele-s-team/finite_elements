import csv
import importlib
from fenics import *
import os
import ufl as ufl

import differential_geometry.boundary.geometry as bgeo
import function_spaces as fsp
import differential_geometry.manifold.geometry as geo
import runtime_arguments as rarg
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)


i, j, k, l = ufl.indices( 4 )

# create the path for the csv file if it does not exist
filename_bcs = rarg.args.output_directory + '/bcs.csv'
os.makedirs(os.path.dirname(filename_bcs), exist_ok=True)

csvfile = open(filename_bcs, 'a', newline='' )
fieldnames = [ \
    '<<(l_profile_v_bar^i - v_bar^i)(l_profile_v_bar_i - v_bar_i)>>_{l + t + b + circle}',\
    '<<(phi - r_profile_phi)^2>>_r' ,\
    '<<(n^i  \partial_i phi)^2>>_{l + t + b + circle}' \
    ]
writer = csv.DictWriter( csvfile, fieldnames=fieldnames )
writer.writeheader()


# this function prints out the residuals of BCs
def print_bcs():
    # get the solution and write it to file

    # write the residual of natural BCs on step 2 to file
    writer.writerows( [{ \
        fieldnames[0]: \
            (sqrt( assemble( (fsp.v_[i] - vp.v__profile_l[i])  * (fsp.v_[i] - vp.v__profile_l[i]) * rmsh.ds_l ) + assemble( fsp.v_[i] * fsp.v_[i] * (rmsh.ds_t + rmsh.ds_b + rmsh.ds_circle) ) ) / \
             assemble( Constant( 1.0 ) * (rmsh.ds_l + rmsh.ds_t + rmsh.ds_b + rmsh.ds_circle) )), \
        fieldnames[1]: \
            sqrt( (assemble( (bgeo.facet_normal[i] * (fsp.phi.dx( i ))) ** 2 * rmsh.ds_l ) \
                   + assemble( (bgeo.facet_normal[i] * (fsp.phi.dx( i ))) ** 2 * (rmsh.ds_t + rmsh.ds_b) ) \
                   + assemble( (bgeo.facet_normal[i] * (fsp.phi.dx( i ))) ** 2 * rmsh.ds_circle )) \
                  / assemble( Constant( 1.0 ) * (rmsh.ds_l + rmsh.ds_t + rmsh.ds_b + rmsh.ds_circle) ) ), \
        fieldnames[2]: \
            sqrt( assemble( (fsp.phi) ** 2 * rmsh.ds_r ) /
                  assemble( Constant( 1.0 ) * rmsh.ds_r ) ) \
        }] )

    csvfile.flush()
