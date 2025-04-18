"""
FEniCS tutorial demo program: Incompressible Navier-Stokes equations
for flow around a cylinder using the Incremental Pressure Correction
Scheme (IPCS).

  u' + u . nabla(u)) - div(sigma(u, p)) = f
                                 div(u) = 0


run with
python ft08_navier_stokes_cylinder.py [path where to read the mesh]
clear; clear; rm -r solution-ft08_navier_stokes_cylinder; python3 ft08_navier_stokes_cylinder.py /home/fenics/shared/channel-with-cylinder-flat-incremental-pressure-correction-scheme/mesh/solution
"""

from __future__ import print_function
from fenics import *
from mshr import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument( "input_directory" )
args = parser.parse_args()

T = 1.0  # final time
num_steps = 1024  # number of time steps
dt = T / num_steps  # time step size
mu = 0.001  # dynamic viscosity
rho = 1  # density

# read mesh
mesh = Mesh()
with XDMFFile( (args.input_directory) + "/triangle_mesh.xdmf" ) as infile:
    infile.read( mesh )
mvc = MeshValueCollection( "size_t", mesh, 2 )
with XDMFFile( (args.input_directory) + "/line_mesh.xdmf" ) as infile:
    infile.read( mvc, "name_to_read" )

# Define function spaces
V = VectorFunctionSpace( mesh, 'P', 2 )
Q = FunctionSpace( mesh, 'P', 1 )

# Define boundaries
inflow = 'near(x[0], 0)'
outflow = 'near(x[0], 2.2)'
walls = 'near(x[1], 0) || near(x[1], 0.41)'
cylinder = 'on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3'

# Define inflow profile
inflow_profile = ('4.0*1.5*x[1]*(0.41 - x[1]) / pow(0.41, 2)', '0')

# Define boundary conditions
bcu_inflow = DirichletBC( V, Expression( inflow_profile, degree=2 ), inflow )
bcu_walls = DirichletBC( V, Constant( (0, 0) ), walls )
bcu_cylinder = DirichletBC( V, Constant( (0, 0) ), cylinder )
bcp_outflow = DirichletBC( Q, Constant( 0 ), outflow )
bcu = [bcu_inflow, bcu_walls, bcu_cylinder]
bcp = [bcp_outflow]

# Define trial and test functions
u = TrialFunction( V )
v = TestFunction( V )
p = TrialFunction( Q )
q = TestFunction( Q )

# Define functions for solutions at previous and current time steps
u_n = Function( V )
u_ = Function( V )
p_n = Function( Q )
p_ = Function( Q )

# Define expressions used in variational forms
U = 0.5 * (u_n + u)
n = FacetNormal( mesh )
f = Constant( (0, 0) )
k = Constant( dt )
mu = Constant( mu )
rho = Constant( rho )


# Define symmetric gradient
def epsilon(u):
    return sym( nabla_grad( u ) )


# Define stress tensor
def sigma(u, p):
    return 2 * mu * epsilon( u ) - p * Identity( len( u ) )


# Define variational problem for step 1
F1 = rho * dot( (u - u_n) / k, v ) * dx \
     + rho * dot( dot( u_n, nabla_grad( u_n ) ), v ) * dx \
     + inner( sigma( U, p_n ), epsilon( v ) ) * dx \
     + dot( p_n * n, v ) * ds - dot( mu * nabla_grad( U ) * n, v ) * ds \
     - dot( f, v ) * dx
a1 = lhs( F1 )
L1 = rhs( F1 )

# Define variational problem for step 2
a2 = dot( nabla_grad( p ), nabla_grad( q ) ) * dx
L2 = dot( nabla_grad( p_n ), nabla_grad( q ) ) * dx - (1 / k) * div( u_ ) * q * dx

# Define variational problem for step 3
a3 = dot( u, v ) * dx
L3 = dot( u_, v ) * dx - k * dot( nabla_grad( p_ - p_n ), v ) * dx

# Assemble matrices
A1 = assemble( a1 )
A2 = assemble( a2 )
A3 = assemble( a3 )

# Apply boundary conditions to matrices
[bc.apply( A1 ) for bc in bcu]
[bc.apply( A2 ) for bc in bcp]

# Create XDMF files for visualization output
xdmffile_u = XDMFFile( 'solution-ft08_navier_stokes_cylinder/v_n.xdmf' )
xdmffile_p = XDMFFile( 'solution-ft08_navier_stokes_cylinder/p_n.xdmf' )

# Time-stepping
t = 0
for n in range( num_steps ):
    # Update current time
    t += dt

    # Step 1: Tentative velocity step
    b1 = assemble( L1 )
    [bc.apply( b1 ) for bc in bcu]
    solve( A1, u_.vector(), b1, 'bicgstab', 'hypre_amg' )

    # Step 2: Pressure correction step
    b2 = assemble( L2 )
    [bc.apply( b2 ) for bc in bcp]
    solve( A2, p_.vector(), b2, 'bicgstab', 'hypre_amg' )

    # Step 3: Velocity correction step
    b3 = assemble( L3 )
    solve( A3, u_.vector(), b3, 'cg', 'sor' )

    # Save solution to file (XDMF/HDF5)
    xdmffile_u.write( u_, t )
    xdmffile_p.write( p_, t )

    # Update previous solution
    u_n.assign( u_ )
    p_n.assign( p_ )

    print( "\t%.2f %%" % (100.0 * (t / T)), flush=True )
