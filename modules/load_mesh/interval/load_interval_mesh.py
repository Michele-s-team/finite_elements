import colorama as col
from fenics import *
import math
import numpy as np

import calculus
import input_output as io
import mesh as msh

# CHANGE PARAMETERS HERE
c_test = [0.3]
r_test = 0.345
# CHANGE PARAMETERS HERE


parameters = io.read_parameters_from_csv_file("parameters_bc_line.csv")
mesh = IntervalMesh(parameters['N'], parameters['x_l'], parameters['x_r'])

# create a function for the lines
cf = MeshFunction("size_t", mesh, mesh.topology().dim())
cf.set_all(parameters['line_id'])  # Tag entire line as region parameters['line_id']

# creat a function for the vertices
vf = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
for vertex in vertices(mesh):
    x = vertex.point().x()  # Get x-coordinate

    if math.isclose(x, parameters['x_l']):
        vf[vertex] = parameters['vertex_l_id']

    if math.isclose(x, parameters['x_r']):
        vf[vertex] = parameters['vertex_r_id']


# Save the mesh components to files
with XDMFFile("line_mesh.xdmf") as outfile:
    outfile.write(cf)

with XDMFFile("vertex_mesh.xdmf") as outfile:
    outfile.write(vf)

def read_mesh_components_new(mesh, dim, filename):
    mf = MeshFunction("size_t", mesh, dim)
    with XDMFFile(filename) as infile:
        infile.read(mf)  # Remove the "name_to_read" parameter
    return mf

cf_read = read_mesh_components_new(mesh, mesh.topology().dim(), "line_mesh.xdmf")
vf_read = read_mesh_components_new(mesh, mesh.topology().dim()-1, "vertex_mesh.xdmf")


dx = Measure("dx", domain=mesh, subdomain_data=cf, subdomain_id=parameters['line_id'])
ds_l = Measure("ds", domain=mesh, subdomain_data=vf, subdomain_id=parameters['vertex_l_id'])
ds_r = Measure("ds", domain=mesh, subdomain_data=vf, subdomain_id=parameters['vertex_r_id'])

ds = ds_l + ds_r

dx_read = Measure("dx", domain=mesh, subdomain_data=cf_read, subdomain_id=parameters['line_id'])
ds_l_read = Measure("ds", domain=mesh, subdomain_data=vf_read, subdomain_id=parameters['vertex_l_id'])


# a function space used solely to define function_test_integrals_fenics
Q_test = FunctionSpace(mesh, 'P', 2)


# function_test_integrals_fenics is a function of two variables, that will be used to test whether the boundary elements ds_circle, ds_inflow, ds_outflow, .. are defined correclty . This will be done by computing an integral of f_test_ds over these boundary terms and comparing with the exact result
def function_test_integrals(x):
    return (np.cos(np.linalg.norm(np.subtract(x, c_test)) - r_test) ** 2.0)


function_test_integrals_fenics = Function(Q_test)


class FunctionTestIntegrals(UserExpression):
    def eval(self, values, x):
        values[0] = function_test_integrals(x)

    def value_shape(self):
        return (1,)


function_test_integrals_fenics.interpolate(FunctionTestIntegrals(element=Q_test.ufl_element()))

print(f'int dx = {assemble(Constant(1) * dx)}')
print(f'int ds_l = {assemble(function_test_integrals_fenics * ds_l)}')

integral_exact_dx = calculus.curve_integral_line(function_test_integrals, parameters['x_l'], parameters['x_r'])

integral_exact_ds_l = function_test_integrals_fenics(parameters['x_l'])
integral_exact_ds_r = function_test_integrals_fenics(parameters['x_r'])

integral_exact_ds = integral_exact_ds_l + integral_exact_ds_r

test_mesh_integral_errors = []

test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_dx, function_test_integrals_fenics, dx, '\int f dx'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_dx, function_test_integrals_fenics, dx_read, '\int f dx_read'))

test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_l, function_test_integrals_fenics, ds_l, '\int f ds_l'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_r, function_test_integrals_fenics, ds_r, '\int f ds_r'))

test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_l, function_test_integrals_fenics, ds_l_read, '\int f ds_l_read'))


test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds, function_test_integrals_fenics, ds, '\int f ds'))

print(f'Maximum relative error of mesh integrals = {col.Fore.BLUE}{max(test_mesh_integral_errors):.{io.number_of_decimals}e}{col.Fore.RESET}')

boundary = 'on_boundary'
boundary_l = f'near(x[0], {parameters["x_l"]})'
boundary_r = f'near(x[1], {parameters["x_r"]})'
