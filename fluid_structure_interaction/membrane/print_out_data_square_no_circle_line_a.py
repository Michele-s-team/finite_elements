import importlib
from fenics import *
import ufl as ufl

import switch_problem as swi

fi = importlib.import_module(swi.fi)



# this method prints out the residuals of BCs for all sectors
def print_data(step):
    
    fi.writer_data.writerows([{

        fi.fieldnames_bcs[0]: \
            step
    }])

    fi.csvfile_bcs.flush()

