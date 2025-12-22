# this module prints out the time taken to solve the variational problem in solve.pt

import importlib
import os
import pandas as pd

import runtime_arguments as rarg
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)


def print_time(elapsed_time):
    
    errors = pd.DataFrame({
        'num_cells_mesh': [rmsh.lmsh.mesh.num_cells()],
        'time': [elapsed_time]
        })
    errors.to_csv(os.path.join(rarg.args.output_directory, 'time.csv'), index=False, float_format='%.3e')

