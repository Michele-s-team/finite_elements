import input_output as io

# CHANGE PARAMETERS HERE
rho_fluid = io.read_parameter_from_csv_file("parameters.csv", "rho_fluid", float)
mu_fluid = io.read_parameter_from_csv_file("parameters.csv", "mu_fluid", float)
v_l = io.read_parameter_from_csv_file("parameters.csv", "v_l", float)
T = io.read_parameter_from_csv_file("parameters.csv", "T", float)
num_steps = io.read_parameter_from_csv_file("parameters.csv", "num_steps", int)
exponent = io.read_parameter_from_csv_file("parameters.csv", "exponent", float)
K_elastic = io.read_parameter_from_csv_file("parameters.csv", "K_elastic", float)
mu_elastic = io.read_parameter_from_csv_file("parameters.csv", "mu_elastic", float)


print(f'rho_fluid = {rho_fluid}, mu_fluid = {mu_fluid}, v_l = {v_l}, T = {T},  num_steps = {num_steps}, exponent = {exponent}, K_elastic = {K_elastic}, mu_elastic = {mu_elastic}')