import input_output as io

parameters_file = 'mesh_parameters.csv'

print(f'Reading parameters from file: {parameters_file} ...')
L = io.read_parameter_from_csv_file(parameters_file, "L", float)
x_p = io.read_parameter_from_csv_file(parameters_file, "x_p", float)
print(f'... done.')

print(f'L = {L}, x_p = {x_p}')
