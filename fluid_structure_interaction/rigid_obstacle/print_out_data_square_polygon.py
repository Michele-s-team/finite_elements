import importlib
from fenics import *
import ufl as ufl

import switch_problem as swi


import mesh_quality as msh_qu

fi = importlib.import_module(swi.fi)
rmsh = importlib.import_module(swi.rmsh)
vp_mesh = importlib.import_module(swi.vp_mesh)
vp_fluid = importlib.import_module(swi.vp_fluid)



# this function prints out the residuals of BCs
def print_data(step):

    # write data to file
    fi.writer_data.writerows([{
        fi.fieldnames_data[0]: \
            step, \
        fi.fieldnames_data[1]: \
            msh_qu.quality
        }])

    fi.csvfile_data.flush()
