import input_output as io
import runtime_arguments_generate_mesh as rarg

parameters_file_path = io.add_trailing_slash(rarg.args.parameter_directory) + 'mesh_parameters.csv'
parameters = io.read_parameters_from_csv_file(parameters_file_path)