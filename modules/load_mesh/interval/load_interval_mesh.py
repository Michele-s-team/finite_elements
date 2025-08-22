import colorama as col
from fenics import *
import math
import numpy as np

import calculus
import input_output as io
import runtime_arguments as rarg
import mesh as msh

# CHANGE PARAMETERS HERE
c_test = [0.3]
r_test = 0.345
# CHANGE PARAMETERS HERE


parameters = io.read_parameters_from_csv_file("parameters_bc_line.csv")
mesh_t = IntervalMesh(parameters['N'], parameters['x_l'], parameters['x_r'])

# create a function for the lines
cf_t = MeshFunction("size_t", mesh_t, mesh_t.topology().dim())
cf_t.set_all(parameters['line_id'])  # Tag entire line as region parameters['line_id']

# creat a function for the vertices
vf_t = MeshFunction("size_t", mesh_t, mesh_t.topology().dim() - 1)
for vertex in vertices(mesh_t):
    x = vertex.point().x()  # Get x-coordinate

    if math.isclose(x, parameters['x_l']):
        vf_t[vertex] = parameters['vertex_l_id']

    if math.isclose(x, parameters['x_r']):
        vf_t[vertex] = parameters['vertex_r_id']

# Save the mesh_t components to files
# Save using HDF5
with HDF5File(mesh_t.mpi_comm(), io.add_trailing_slash(rarg.args.output_directory) + "line_mesh.h5", "w") as outfile:
    outfile.write(mesh_t, "mesh")
    outfile.write(cf_t, "cf")

with HDF5File(mesh_t.mpi_comm(), io.add_trailing_slash(rarg.args.output_directory) + "vertex_mesh.h5", "w") as outfile:
    outfile.write(mesh_t, "mesh")
    outfile.write(vf_t, "vf")


def read_mesh_from_file_new(filename, mesh_name):
    mesh = Mesh()
    with HDF5File(mesh.mpi_comm(), filename, "r") as infile:
        infile.read(mesh, mesh_name, False)
    return mesh


def read_mesh_function_from_file(mesh, dim, filename, mf_name="name_to_read", file_format=None):
    """
    Read mesh function from file with unified interface.

    Parameters:
    -----------
    mesh : dolfin.Mesh
        The mesh object
    dim : int
        Dimension of the mesh function
    filename : str
        Path to the file
    mf_name : str, optional
        Name of the mesh function to read (default: "name_to_read")
    file_format : str, optional
        File format: "hdf5" or "xdmf" (auto-detected if None)

    Returns:
    --------
    MeshFunction or MeshFunctionSizet
        The mesh function read from file
    """
    if file_format is None:
        # Auto-detect format from file extension
        if filename.endswith('.h5') or filename.endswith('.hdf5'):
            file_format = "hdf5"
        elif filename.endswith('.xdmf'):
            file_format = "xdmf"
        else:
            raise ValueError(f"Cannot determine file format from extension: {filename}")

    if file_format.lower() == "hdf5":
        mf = MeshFunction("size_t", mesh, dim)
        with HDF5File(mesh.mpi_comm(), filename, "r") as infile:
            infile.read(mf, mf_name)
        return mf

    elif file_format.lower() == "xdmf":
        mesh_value_collection = MeshValueCollection("size_t", mesh, dim)
        with XDMFFile(filename) as infile:
            infile.read(mesh_value_collection, mf_name)
            infile.close()
        return cpp.mesh.MeshFunctionSizet(mesh, mesh_value_collection)

    else:
        raise ValueError(f"Unsupported file format: {file_format}")


# Read meshes from files
mesh = read_mesh_from_file_new(io.add_trailing_slash(rarg.args.output_directory) + "line_mesh.h5", "mesh")

print(f"Original mesh dimension: {mesh.topology().dim()}")
print(f"Original mesh num vertices: {mesh.num_vertices()}")
print(f"Original mesh num cells: {mesh.num_cells()}")
print(f"Original mesh coordinates shape: {mesh.coordinates().shape}")

print(f"Read mesh dimension: {mesh.topology().dim()}")
print(f"Read mesh num vertices: {mesh.num_vertices()}")
print(f"Read mesh num cells: {mesh.num_cells()}")
print(f"Read mesh coordinates shape: {mesh.coordinates().shape}")

# Check if coordinates are identical
print(f"Coordinates match: {np.allclose(mesh.coordinates(), mesh.coordinates())}")

# Build mesh functions from meshes loaded from files
cf = read_mesh_function_from_file(mesh, mesh.topology().dim(), io.add_trailing_slash(rarg.args.output_directory) + "line_mesh.h5", "cf")
vf = read_mesh_function_from_file(mesh, mesh.topology().dim() - 1, io.add_trailing_slash(rarg.args.output_directory) + "vertex_mesh.h5", "vf")

dx = Measure("dx", domain=mesh, subdomain_data=cf, subdomain_id=parameters['line_id'])
ds_l = Measure("ds", domain=mesh, subdomain_data=vf, subdomain_id=parameters['vertex_l_id'])
ds_r = Measure("ds", domain=mesh, subdomain_data=vf, subdomain_id=parameters['vertex_r_id'])
ds = Measure("ds", domain=mesh)

# a function space used solely to define function_test_integrals_fenics
Q_test_read = FunctionSpace(mesh, 'P', 2)


# function_test_integrals_fenics is a function of two variables, that will be used to test whether the boundary elements ds_circle, ds_inflow, ds_outflow, .. are defined correclty . This will be done by computing an integral of f_test_ds over these boundary terms and comparing with the exact result
def function_test_integrals(x):
    return (np.cos(np.linalg.norm(np.subtract(x, c_test)) - r_test) ** 2.0)


function_test_integrals_fenics_read = Function(Q_test_read)


class FunctionTestIntegrals(UserExpression):
    def eval(self, values, x):
        values[0] = function_test_integrals(x)

    def value_shape(self):
        return (1,)


function_test_integrals_fenics_read.interpolate(FunctionTestIntegrals(element=Q_test_read.ufl_element()))

integral_exact_dx = calculus.curve_integral_line(function_test_integrals, parameters['x_l'], parameters['x_r'])

integral_exact_ds_l = function_test_integrals_fenics_read(parameters['x_l'])
integral_exact_ds_r = function_test_integrals_fenics_read(parameters['x_r'])
integral_exact_ds = integral_exact_ds_l + integral_exact_ds_r

test_mesh_integral_errors = []

test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_dx, function_test_integrals_fenics_read, dx, '\int f dx_read'))

test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_l, function_test_integrals_fenics_read, ds_l, '\int f ds_l_read'))
test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds_r, function_test_integrals_fenics_read, ds_r, '\int f ds_r_read'))

test_mesh_integral_errors.append(msh.test_mesh_integral(integral_exact_ds, function_test_integrals_fenics_read, ds, '\int f ds'))

print(f'Maximum relative error of mesh integrals = {col.Fore.RED}{max(test_mesh_integral_errors):.{io.number_of_decimals}e}{col.Fore.RESET}')

boundary = 'on_boundary'
boundary_l = f'near(x[0], {parameters["x_l"]})'
boundary_r = f'near(x[1], {parameters["x_r"]})'
