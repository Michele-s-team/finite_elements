import input_output as io

# CHANGE PARAMETERS HERE
rho = io.read_parameter_from_csv_file("parameters.csv", "rho", float)
mu = io.read_parameter_from_csv_file("parameters.csv", "mu", float)
T = io.read_parameter_from_csv_file("parameters.csv", "T", float)
num_steps = io.read_parameter_from_csv_file("parameters.csv", "num_steps", int)
exponent = io.read_parameter_from_csv_file("parameters.csv", "exponent", float)
I_ellipse = io.read_parameter_from_csv_file("parameters.csv", "I_ellipse", float)


print(f'rho = {rho}, mu = {mu}, T = {T}, I_ellipse = {I_ellipse}, num_steps = {num_steps}, exponent = {exponent}')