import csv
import importlib
from fenics import *
import os
import ufl as ufl

import boundary_geometry as bgeo
import function_spaces as fsp
import differential_geometry.manifold.geometry as geo
import mesh as msh
import runtime_arguments as rarg
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

i, j, k, l = ufl.indices(4)

# create the path for the csv file if it does not exist
filename_bcs = rarg.args.output_directory + '/bcs.csv'
os.makedirs(os.path.dirname(filename_bcs), exist_ok=True)

csvfile = open(filename_bcs, 'a', newline='')
fieldnames = [ \
    '<<(l_profile_u_bar^i - u_bar^i)(l_profile_u_bar_i - u_bar_i)>>_le',\
    '<<|u_bar|^2>>_{tobo}',\
    '<<|u_bar|^2>>_{frba}',\
    '<<|u_bar|^2>>_sphere',\
    '<<p>>_ri',\
    '<(n^j \partial_j ((u_bar^i + u_n^i)/2)) (n^k \partial_k ((u_bar^i + u_n^i)/2))>>_ri'
]
writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
writer.writeheader()


# this function prints out the residuals of BCs
def print_bcs():
    # get the solution and write it to file

    # write the residual of natural BCs on step 2 to file
    writer.writerows([{ \
        fieldnames[0]: \
            msh.abs_wrt_measure(sqrt((fsp.u_bar[i] - vp.u_bar_le_profile[i]) * (fsp.u_bar[i] - vp.u_bar_le_profile[i])), rmsh.ds_le) ,\
        fieldnames[1]: \
            msh.abs_wrt_measure(sqrt(fsp.u_bar[i] * fsp.u_bar[i]),rmsh.ds_tobo),\
        fieldnames[2]: \
            msh.abs_wrt_measure(sqrt(fsp.u_bar[i] * fsp.u_bar[i]),rmsh.ds_frba),\
        fieldnames[3]: \
            msh.abs_wrt_measure(sqrt(fsp.u_bar[i] * fsp.u_bar[i]),rmsh.ds_sphere),
        fieldnames[4]: \
            msh.abs_wrt_measure(fsp.p_, rmsh.ds_ri),
        fieldnames[5]:
        msh.abs_wrt_measure(sqrt((bgeo.facet_normal[i] * ((fsp.u_bar[j] + fsp.u_n[j])/2).dx(i)) * (bgeo.facet_normal[k] * ((fsp.u_bar[j] + fsp.u_n[j])/2).dx(k))),rmsh.ds_ri)
    }])

    csvfile.flush()
