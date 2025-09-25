'''
this is a minimal working example for the solution of Poisson equation on a line with an inner vertex, where everything is in one file, 
to be used on the Fenics forum for example

Run with 
    python3 solve.py
'''

from fenics import *
import numpy as np
import ufl
import sys

mesh_path = 'mesh_files/'


mesh = Mesh()
with HDF5File(mesh.mpi_comm(), mesh_path + "line_mesh.h5", "r") as infile:
    infile.read(mesh, 'mesh', False)


# read lines and vertices
cf = MeshFunction("size_t", mesh, mesh.topology().dim())
with HDF5File(mesh.mpi_comm(), mesh_path + "line_mesh.h5", "r") as infile:
    infile.read(cf, 'cf')
    
vf = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
with HDF5File(mesh.mpi_comm(), mesh_path + "vertex_mesh.h5", "r") as infile:
    infile.read(vf, 'vf')




# radius of the smallest cell in the mesh
r_mesh = mesh.hmin()

# measures for the line, for left and right points and for middle point
dx = Measure("dx", domain=mesh, subdomain_data=cf, subdomain_id=1)
ds_l = Measure("ds", domain=mesh, subdomain_data=vf, subdomain_id=2)
ds_r = Measure("ds", domain=mesh, subdomain_data=vf, subdomain_id=3)
ds_m = Measure("dS", domain=mesh, subdomain_data=vf, subdomain_id=4)
ds_lr = Measure("ds", domain=mesh)


# define boundaries for left and right points, and for middle point
boundary = 'on_boundary'
boundary_l = f'near(x[0], {0.0})'
boundary_r = f'near(x[0], {1.0})'
boundary_m = f'near(x[0], {0.5})'

Q = FunctionSpace(mesh, 'P', 4)

u = Function(Q)
nu_u = TestFunction(Q)
f = Function(Q)
J_u = TrialFunction(Q)
u_exact = Function(Q)


# consider a Poisson equation for which I know the exact solution
class u_exact_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] = np.cos(2 * np.pi * x[0]) 

    def value_shape(self):
        return (1,)


class grad_u_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] = - 2 * np.pi / 1 * np.sin(2 * np.pi * x[0]) 

    def value_shape(self):
        return (1,)


class laplacian_u_expression(UserExpression):
    def eval(self, values, x):
        # test case 1
        values[0] = -( 2 * np.pi )**2 * np.cos(2 * np.pi * x[0]) 
          
    def value_shape(self):
        return (1,)


u_exact.interpolate(u_exact_expression(element=Q.ufl_element()))
f.interpolate(laplacian_u_expression(element=Q.ufl_element()))


# Dirichlet boundary conditions on left and right edge
bc_l = DirichletBC(Q, u_exact, boundary_l )
bc_r = DirichletBC(Q, u_exact, boundary_r )

# tentative condition to enforce u = u _exact at the middle (m) point
bc_m = DirichletBC(Q, u_exact, boundary_m )

#case A: this works
bcs = [bc_l, bc_r]
# case B: this does not work
# bcs = [bc_l, bc_m]

i = ufl.indices(1)
facet_normal = FacetNormal(mesh)


# variational problem
F = (u.dx(i) * nu_u.dx(i) + f * nu_u) * dx \
    - facet_normal[i] * (u.dx(i)) * nu_u * ds_lr



J = derivative(F, u, J_u)
problem = NonlinearVariationalProblem(F, u, bcs, J)
solver = NonlinearVariationalSolver(problem)

# set the solver parameters here
params = {'nonlinear_solver': 'newton',
          'newton_solver':
              {
                  'linear_solver': 'superlu',
                  'absolute_tolerance': 1e-12,
                  'relative_tolerance': 1e-12,
                  'maximum_iterations': 1000000,
                  'relaxation_parameter': 0.95,
              }
          }
solver.parameters.update(params)

solver.solve()

print(f'\nerror = {assemble((u - u_exact)**2 * dx)/assemble(Constant(1)* dx)}')