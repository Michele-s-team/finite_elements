from fenics import *

import mesh as msh
import runtime_arguments as rarg

mesh = msh.read_from_file(rarg.args.input_directory)
