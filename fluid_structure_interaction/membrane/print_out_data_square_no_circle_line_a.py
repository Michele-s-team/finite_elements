import importlib
from fenics import *
import numpy as np
import ufl as ufl

import mesh_quality as msh_qu
import parameters.read.solution as rpam
import switch_problem as swi

fi = importlib.import_module(swi.fi)
rmsh = importlib.import_module(swi.rmsh)

# compute the maximal value of the y coordinate of the top boundary of mesh[0], in order to have an idea of how much the edge has been displaced vertically



# this method prints out the residuals of BCs for all sectors
def print_data(step):
    
    fi.writer_data.writerows([{

        fi.fieldnames_bcs[0]: \
            step,\
        fi.fieldnames_data[1]: \
            f"{msh_qu.quality:.{rpam.parameters['print_out_digits']}e}",\
        fi.fieldnames_data[2]: \
            f"{np.max([rmsh.parameters['shape_coordinates'][i][1] for i in range(len(rmsh.parameters['shape_coordinates']))]):.{rpam.parameters['print_out_digits']}e}"
            
    }])

    fi.csvfile_data.flush()

