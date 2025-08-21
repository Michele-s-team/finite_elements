from fenics import *
import runtime_arguments as rarg

import input_output as io

parameters = io.read_parameters_from_csv_file("parameters_bc_line.csv")

mesh = IntervalMesh(parameters['N'], parameters['x_min'], parameters['x_max'])
