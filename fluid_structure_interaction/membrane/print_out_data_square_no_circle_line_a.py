import importlib
from fenics import *
import ufl as ufl

import mesh_quality as msh_qu
import parameters.read.solution as rpam
import switch_problem as swi

fi = importlib.import_module(swi.fi)



# this method prints out the residuals of BCs for all sectors
def print_data(step):
    
    fi.writer_data.writerows([{

        fi.fieldnames_bcs[0]: \
            step,\
        fi.fieldnames_data[1]: \
            f"{msh_qu.quality:.{rpam.parameters['print_out_digits']}e}"
            
    }])

    fi.csvfile_data.flush()

