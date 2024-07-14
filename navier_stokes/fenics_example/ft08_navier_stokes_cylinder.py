"""
FEniCS tutorial demo program: Incompressible Navier-Stokes equations
for flow around a cylinder using the Incremental Pressure Correction
Scheme (IPCS).

  u' + u . nabla(u)) - div(sigma(u, p)) = f
                                 div(u) = 0
"""

from __future__ import print_function
from fenics import *
from mshr import *
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input_directory")
parser.add_argument("output_directory")
args = parser.parse_args()

set_log_level(30)


kappa = 1.0


# # Create mesh
# channel = Rectangle(Point(0, 0), Point(2.2, 0.41))
# cylinder = Circle(Point(0.2, 0.2), 0.05)
# domain = channel - cylinder
# mesh = generate_mesh(domain, 64)

L = 2.2
h = 0.41

#create mesh
mesh=Mesh()
with XDMFFile((args.input_directory) + "/triangle_mesh.xdmf") as infile:
    infile.read(mesh)
mvc = MeshValueCollection("size_t", mesh, 2)
with XDMFFile((args.input_directory) + "/line_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
#sub = cpp.mesh.MeshFunctionSizet(mesh, mvc)


# Define function spaces
P_z = FiniteElement('P', triangle, 2)
P_omega = FiniteElement('P', triangle, 1)
element = MixedElement([P_z, P_omega])
Q_z_omega = FunctionSpace(mesh, element)
Q_z = Q_z_omega.sub(0).collapse()
Q_omega = Q_z_omega.sub(1).collapse()



# Define boundaries and obstacle
#CHANGE PARAMETERS HERE
inflow   = 'near(x[0], 0)'
outflow  = 'near(x[0], 2.2)'
walls    = 'near(x[1], 0) || near(x[1], 0.41)'
#CHANGE PARAMETERS HERE

# Define inflow profile
# z_profile = 'x[1]'
# omega_profile = 'x[1]'


bc_z = DirichletBC(Q_z_omega.sub(0), Expression('x[1]', degree=1), walls)
bc_omega = DirichletBC(Q_z_omega.sub(1), Expression('x[1]', degree=1), walls)

bc_z_omega = [bc_z, bc_omega]

# Define trial and test functions
nu_z, nu_omega = TestFunctions(Q_z_omega)

# Define functions for solutions at previous and current time steps
z_omega = Function(Q_z_omega)
z, omega = split(z_omega)


# Define expressions used in variational forms
n  = FacetNormal(mesh)
kappa = Constant(kappa)




# Define variational problem for step 1
F_z = ( -1.0/(2.0*sqrt_detg(z))* atan(z.dx(1)) * (nu_z.dx(1)) - omega * nu_z ) * dx
F_omega = ( kappa * (1.0/detg(z)) * (omega.dx(1))* (nu_omega.dx(1)) - 2 * (omega**3) * nu_omega ) * dx
F12 = F_z + F_omega

# Define variational problem for step 3
F3 = (dot(us, vs) - (dot(z, vs) - k*dot(nabla_grad(omega - p_n), vs))) * dx

# Create XDMF files for visualization output
xdmffile_u = XDMFFile((args.output_directory) + '/v.xdmf')
xdmffile_p = XDMFFile((args.output_directory) + '/p.xdmf')

# Create VTK files for visualization output
vtkfile_u = File('v.pvd')
vtkfile_p = File('p.pvd')


# Save mesh to file (for use in reaction_system.py)
File('cylinder.xml.gz') << mesh


# Time-stepping
print("Starting time iteration ...", flush=True)
t = 0
for n in range(N):

    # Save solution to file (XDMF/HDF5)
    xdmffile_u.write(u_n, t)
    xdmffile_p.write(p_n, t)

    # Update current time
    t += dt

    # Step 1+2
    # A12 = assemble(a12)
    # b12 = assemble(L12)
    # [bc.apply(A12) for bc in bc_up]
    # [bc.apply(b12) for bc in bc_up]

    solve(F12 == 0, z_omega, bc_up)
    

    # Step 3: Velocity correction step
    # ps_.assign(project(p_, Q))
    # A3 = assemble(a3)
    # b3 = assemble(L3)
    solve(F3 == 0, us)

    # Update previous solution
    #u_n has been already updated by  solve(A3, u_n.vector(), b3,  'cg', 'sor')
    #this step writes the numerical data of up_ into u_n, p_n -> I am interested only in writing into p_n with this line
    u_n.assign(us)
    _u_, p_n = z_omega.split(deepcopy=True)

    print("\t%.2f %%" % (100.0*(t/T)), flush=True)

print("... done.", flush=True)