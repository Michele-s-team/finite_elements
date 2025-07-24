import input_output as io

# CHANGE PARAMETERS HERE
rho = io.read_parameter_from_csv_file("parameters.csv", "rho", float)
mu = io.read_parameter_from_csv_file("parameters.csv", "mu", float)
v_l = io.read_parameter_from_csv_file("parameters.csv", "v_l", float)
T = io.read_parameter_from_csv_file("parameters.csv", "T", float)
num_steps = io.read_parameter_from_csv_file("parameters.csv", "num_steps", int)
exponent = io.read_parameter_from_csv_file("parameters.csv", "exponent", float)


print(f'rho = {rho}, mu = {mu}, v_l = {v_l}, T = {T},  num_steps = {num_steps}, exponent = {exponent}')