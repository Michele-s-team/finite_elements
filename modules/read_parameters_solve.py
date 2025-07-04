import input_output as io
import runtime_arguments as rarg

parameters_file_path = 'parameters_bc_' + rarg.args.problem + '.csv'
parameters = io.read_parameters_from_csv_file(parameters_file_path)