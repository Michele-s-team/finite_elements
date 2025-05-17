'''
this code solves for a flow on a flat rectangular channel with a circular obstacle with the incremental pressure correction scheme

run with

clear; clear; python3 run.py [path where to read the mesh] [path where to store the solution] T N
clear; clear; SOLUTION_PATH="solution"; rm -rf $SOLUTION_PATH; mkdir -p /home/fenics/shared/channel-with-cylinder-flat-incremental-pressure-correction-scheme/$SOLUTION_PATH/snapshots/csv; python3 run.py /home/fenics/shared/channel-with-cylinder-flat-incremental-pressure-correction-scheme/mesh/solution /home/fenics/shared/channel-with-cylinder-flat-incremental-pressure-correction-scheme/$SOLUTION_PATH 0.001 1
'''

from fenics import *
from mshr import *
import numpy as np
import ufl as ufl
import argparse
import csv
import numpy as np

import sys


# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append( module_path )

import geometry as geo
import input_output as io
import mesh as msh


parser = argparse.ArgumentParser()
parser.add_argument("input_directory")
parser.add_argument("output_directory")
parser.add_argument("T")
parser.add_argument("N")
args = parser.parse_args()

print( "Input directory", args.input_directory )
print( "Output directory", args.output_directory )

#CHANGE PARAMETERS HERE
#r, R must be the same as in generate_mesh.py
tol = 1E-3
L = 2.2
h = 0.41
r = 0.05
c_r = [0.2, 0.2]
#CHANGE PARAMETERS HERE

i, j, k, l = ufl.indices(4)


T = (float)( args.T )
num_steps = (int)( args.N )

dt = T / num_steps  # time step size
mu = 0.001
rho = 1.0

print("c_r = ", c_r)
print( "L = ", L )
print( "h = ", h )
print("r = ", r)
print( "T = ", T )
print( "N = ", num_steps )
print( "mu = ", mu )
print("input directory = ", args.input_directory)
print("output directory = ", args.output_directory)

# Create XDMF files for visualization output
xdmffile_v = XDMFFile( (args.output_directory) + "/v.xdmf" )
xdmffile_p = XDMFFile( (args.output_directory) + "/p.xdmf" )

xdmffile_geo = XDMFFile( (args.output_directory) + "/geo.xdmf" )
# this is needed to write multiple data series to xdmffile_geo
xdmffile_geo.parameters.update(
    {
        "functions_share_mesh": True,
        "rewrite_function_mesh": False
    } )

csvfile = open( (args.output_directory) + '/bcs.csv', 'w', newline='' )
fieldnames = [ \
    'n', \
    '<<(v_bar^i - v_boundary^i)(v_bar_i - v_boundary_i)>>_{l U tb U circle}', \
    '<<(n^i \partial_i phi)^2>>_{l U tb U circle}', \
    '<<phi^2>>_r' \
    ]
writer = csv.DictWriter( csvfile, fieldnames=fieldnames )
writer.writeheader()

#create mesh
mesh=Mesh()
with XDMFFile((args.input_directory) + "/triangle_mesh.xdmf") as infile:
    infile.read(mesh)
mvc = MeshValueCollection("size_t", mesh, 2)
with XDMFFile((args.input_directory) + "/line_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")

#read an object with label subdomain_id from xdmf file and assign to it the ds `ds_inner`
mf = dolfin.cpp.mesh.MeshFunctionSizet(mesh, mvc)
ds_l = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=2)
ds_r = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=3)
ds_t = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=4)
ds_b = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=5)
ds_circle = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=6)



#a function space used solely to define f_test_ds
Q_test = FunctionSpace( mesh, 'P', 2 )

# f_test_ds is a scalar function defined on the mesh, that will be used to test whether the boundary elements ds_circle, ds_inflow, ds_outflow, .. are defined correclty . This will be done by computing an integral of f_test_ds over these boundary terms and comparing with the exact result
f_test_ds = Function( Q_test )

#analytical expression for a  scalar function used to test the ds
class FunctionTestIntegrals( UserExpression ):
    def eval(self, values, x):
        c_test = [0.3, 0.76]
        r_test = 0.345
        values[0] = np.cos(geo.my_norm(np.subtract(x, c_test)) - r_test)**2.0
    def value_shape(self):
        return (1,)

f_test_ds.interpolate( FunctionTestIntegrals( element=Q_test.ufl_element() ) )


#here I integrate \int ds 1 over the circle and store the result of the integral as a double in inner_circumference
integral_l = assemble(f_test_ds*ds_l)
integral_r = assemble(f_test_ds*ds_r)
integral_t = assemble(f_test_ds*ds_t)
integral_b = assemble(f_test_ds*ds_b)
integral_circle = assemble(f_test_ds*ds_circle)

#print out the integrals on the surface elements and compare them with the exact values to double check that the elements are tagged correctly
print("Integral l = ", integral_l, " should be 0.373168")
print("Integral r = ", integral_r, " should be 0.00227783")
print("Integral t = ", integral_t, " should be 1.36562")
print("Integral b = ", integral_b, " should be 1.02837")
print("Integral circle = ", integral_circle, " 0.298174")

# Define function spaces
#the '2' in ''P', 2)' is the order of the polynomials used to describe these spaces: if they are low, then derivatives high enough of the functions projected on thee spaces will be set to zero !
O = VectorFunctionSpace(mesh, 'P', 2, dim=2)
O3d = VectorFunctionSpace(mesh, 'P', 2, dim=3)
Q = FunctionSpace(mesh, 'P', 1)
#I will use Q4 for functions which involve high order derivatives

# Define boundaries and obstacle
#CHANGE PARAMETERS HERE
inflow   = 'near(x[0], 0)'
outflow  = 'near(x[0], 2.2)'
walls    = 'near(x[1], 0) || near(x[1], 0.41)'
cylinder = 'on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3'
#CHANGE PARAMETERS HERE



# Define functions for solutions at previous and current time steps
v_n = Function( O )
v_ = Function( O )
sigma_ = Function( Q )
p_n = Function( Q )


#trial analytical expression for a vector
class TangentVelocityExpression(UserExpression):
    def eval(self, values, x):
        values[0] = 0.0
        values[1] = 0.0
    def value_shape(self):
        return (2,)

#trial analytical expression for the height function z(x,y)
class ManifoldExpression(UserExpression):
    def eval(self, values, x):
        values[0] = 0
        # values[0] =  1E-3 * np.cos(x[0]/L * 2*(math.pi))
    def value_shape(self):
        return (1,)

# trial analytical expression for the  surface tension sigma(x,y)
class SurfaceTensionExpression(UserExpression):
        def eval(self, values, x):
            # values[0] = 4*x[0]*x[1]*sin(8*(norm(np.subtract(x, c_r)) - r))*sin(8*(norm(np.subtract(x, c_R)) - R))
            # values[0] = cos(norm(np.subtract(x, c_r)) - r) * sin(norm(np.subtract(x, c_R)) - R)
            values[0] = 0.0

        def value_shape(self):
            return (1,)

#trial analytical expression for w
class NormalVelocityExpression(UserExpression):
    def eval(self, values, x):
        # values[0] = (np.subtract(x, c_r)[0])*(np.subtract(x, c_r)[1])*cos(norm(np.subtract(x, c_r)) - r) * sin(norm(np.subtract(x, c_R)) - R)
        values[0] = 0.0
    def value_shape(self):
        return (1,)


#analytical expression for a general scalar function
class ScalarFunctionExpression(UserExpression):
    def eval(self, values, x):
        values[0] = np.cos(geo.my_norm(np.subtract(x, c_r)) - r) * np.cos(gep.my_norm(np.subtract(x, c_R)) - R)
    def value_shape(self):
        return (1,)

facet_normal = FacetNormal( mesh )

# Define symmetric gradient
def epsilon(u):
    # nabla_grad(u)_{i,j} = (u[j]).dx[i]
    #sym(nabla_grad(u)) =  nabla_grad(u)_{i,j} + nabla_grad(u)_{j,i}
    # return sym(nabla_grad(u))
    return as_tensor(0.5*(u[i].dx(j) + u[j].dx(i)), (i,j))

#the normal vector on the inflow and outflow
def n_inout():
    x = ufl.SpatialCoordinate(mesh)
    u = as_tensor([conditional(lt(x[0], L/2), -1.0, 1.0), 0.0] )
    return as_tensor(u[k], (k))



# Define trial and test functions
v = Function(O)
v_ = Function(O)
sigma = Function(Q)
phi = Function(Q)
nu = TestFunction(O)
q = TestFunction(Q)



# the vector  or function is interpolated  and written into a Function() object
v_n = interpolate( TangentVelocityExpression( element=O.ufl_element() ), O )
sigma_n = interpolate( SurfaceTensionExpression( element=Q.ufl_element() ), Q )

inflow_profile_v = Expression( ('4.0*1.5*x[1]*(0.41 - x[1]) / pow(0.41, 2)', '0'), degree=2, h=h )

# Define boundary conditions
bcv_inflow = DirichletBC( O, inflow_profile_v, inflow )
bcv_walls = DirichletBC( O, Constant( (0, 0) ), walls )
bcv_cylinder = DirichletBC( O, Constant( (0, 0) ), cylinder )

bc_phi_outflow = DirichletBC( Q, Constant( 0 ), outflow )

# boundary conditions for the surface_tension p
bc_v = [bcv_inflow, bcv_walls, bcv_cylinder]
bc_phi = [bc_phi_outflow]

# Define expressions used in variational forms
V = 0.5 * (v_n + v_)
Deltat = Constant( dt )
mu = Constant( mu )
rho = Constant( rho )

# Define variational problem for step 1
# step 1 for v
F1v = rho * ((v_[i] - v_n[i]) / Deltat + v_n[j] * ((v_n[i]).dx( j ))) * nu[i] * dx \
      + (2.0 * mu * (epsilon( V ))[i, j] * (nu[i]).dx( j ) + sigma_n * (nu[i]).dx( i )) * dx \
      + (- sigma_n * n_inout()[i] * nu[i] - mu * (V[j]).dx( i ) * n_inout()[j] * nu[i]) * ds

# step 2
# natural BCs are imposed here
F2 = ((phi.dx( i )) * (q.dx( i )) + (rho / Deltat) * ((v_[i]).dx( i )) * q) * dx

# Define variational problem for step 3
# step 3 for v
F3v = ((v_[i] - v[i]) - (Deltat / rho) * (phi.dx( i ))) * nu[i] * dx

print( "Starting time iteration ...", flush=True )
# Time-stepping
t = 0
step = 0
for n in range( num_steps ):

    # Write the solution to file
    xdmffile_v.write( v_n, t )
    p_n.assign( -sigma_n )
    xdmffile_p.write( p_n, t )

    io.print_vector_to_csvfile( v_n, (args.output_directory) + '/snapshots/csv/v_n_' + str( step ) + '.csv' )
    io.print_scalar_to_csvfile( sigma_n, (args.output_directory) + '/snapshots/csv/sigma_' + str( step ) + '.csv' )

    # Update current time
    t += dt
    step += 1

    # Step 1: Tentative velocity step
    solve( F1v == 0, v_, bc_v )

    # Step 2: surface_tension correction step
    solve( F2 == 0, phi, bc_phi )

    # Step 3: Velocity correction step
    solve( F3v == 0, v )

    # Update previous solution
    v_n.assign( v )
    sigma_n.assign( -phi + sigma_n )

    data = [
        {'n': n, \
         '<<(v_bar^i - v_boundary^i)(v_bar_i - v_boundary_i)>>_{l U tb U circle}': \
             sqrt( \
                 ( \
                             assemble( (v_[i] - inflow_profile_v[i]) * (v_[i] - inflow_profile_v[i]) * ds_l ) \
                             + assemble( v_[i] * v_[i] * (ds_t + ds_b + ds_circle) ) \
                     ) \
                 / assemble( Constant( 1.0 ) * (ds_l + ds_t + ds_b + ds_circle) ) )
            ,
         '<<(n^i \partial_i phi)^2>>_{l U tb U circle}': \
             sqrt( \
                 (assemble( (facet_normal[i] * (phi.dx( i ))) ** 2 * (ds_l + ds_r + ds_t + ds_b + ds_circle) ) \
                  ) / assemble( Constant( 1.0 ) * (ds_l + ds_r + ds_t + ds_b + ds_circle) ) ), \
         '<<phi^2>>_r': \
             sqrt( assemble( phi ** 2 * ds_r ) / assemble( Constant( 1.0 ) * ds_r ) ) \
         }]
    writer.writerows( data )
    csvfile.flush()

    print( "\t%.2f %%" % (100.0 * (t / T)), flush=True )

print( "... done.", flush=True )
