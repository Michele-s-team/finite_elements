from __future__ import print_function
from fenics import *
from mshr import *
import numpy as np
import argparse
import ufl as ufl



i, j, k, l = ufl.indices(4)


parser = argparse.ArgumentParser()
parser.add_argument("input_directory")
parser.add_argument("output_directory")
args = parser.parse_args()

tol = 1E-3
set_log_level(30)

# Create mesh
# channel = Rectangle(Point(0, 0), Point(2.2, 0.41))
# cylinder = Circle(Point(0.2, 0.2), 0.05)
# domain = channel - cylinder
# mesh = generate_mesh(domain, 64)

print("input directory = ", (args.input_directory))
print("output directory = ", (args.output_directory))

L = 1
H = 1
# r = 0.05
# c_r = [0.2, 0.2]

#create mesh
mesh=Mesh()
with XDMFFile((args.input_directory) + "/triangle_mesh.xdmf") as infile:
    infile.read(mesh)
mvc = MeshValueCollection("size_t", mesh, 2)
with XDMFFile((args.input_directory) + "/line_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
    


    
    
# Create XDMF files for visualization output
xdmffile_u = XDMFFile((args.output_directory) + "/u.xdmf")
xdmffile_v = XDMFFile((args.output_directory) + "/v.xdmf")
xdmffile_check = XDMFFile((args.output_directory) + "/check.xdmf")
# this is needed to write multiple data series to xdmffile_geo
xdmffile_check.parameters.update(
    {
        "functions_share_mesh": True,
        "rewrite_function_mesh": False
    })


# Define function spaces
P_U = FiniteElement('P', triangle, 4)
P_V = FiniteElement('P', triangle, 4)
element = MixedElement([P_U, P_V])
UV = FunctionSpace(mesh, element)
U = UV.sub(0).collapse()
V = UV.sub(1).collapse()

#analytical expression for a general scalar function
class test_function_expression(UserExpression):
    def eval(self, values, x):
        values[0] =  np.sin((x[1]+x[0])/L) * np.cos(((x[0]+x[1]**2+1)/H)**2)
    def value_shape(self):
        return (1,)
    

#analytical expression for a general scalar function
class f_expression(UserExpression):
    def eval(self, values, x):
        # values[0] =  sin(np.pi * x[0]/L)
        values[0] =  1.0
    def value_shape(self):
        return (1,)

# #trial analytical expression for w
# class h_expression(UserExpression):
#     def eval(self, values, x):
#         values[0] = conditional(lt(abs(x[0] - 0.0), tol), 0.0, 1.0) * \
#                     conditional(lt(abs(x[0] - L), tol), (L**3)/12.0, 1.0) * \
#                     conditional(lt(abs(x[1] - 0.0), tol), 0.0, 1.0) * \
#                     conditional(lt(abs(x[1] - H), tol), (H**3)/12.0, 1.0)
        
#     def value_shape(self):
#         return (1,)
    
    
    


#read an object with label subdomain_id from xdmf file and assign to it the ds `ds_inner`
mf = dolfin.cpp.mesh.MeshFunctionSizet(mesh, mvc)

# ds_circle = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=2)
ds_inflow = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=2)
ds_outflow = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=3)
ds_top_wall = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=4)
ds_bottom_wall = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=5)


#f_test_ds is a scalar function defined on the mesh, that will be used to test whether the boundary elements ds_circle, ds_inflow, ds_outflow, .. are defined correclty . This will be done by computing an integral of f_test_ds over these boundary terms and comparing with the exact result 
f_test_ds = Function(U)
f_test_ds = interpolate(test_function_expression(element=U.ufl_element()), U)

#here I integrate \int ds 1 over the circle and store the result of the integral as a double in inner_circumference
# circle_length = assemble(f_test_ds*ds_circle)
inflow_integral = assemble(f_test_ds*ds_inflow)
outflow_integral = assemble(f_test_ds*ds_outflow)
top_wall_integral = assemble(f_test_ds*ds_top_wall)
bottom_wall_integral = assemble(f_test_ds*ds_bottom_wall)
# print("Circle length = ", circle_length, "exact value = 0.02756453133593419.")
print("Inflow integral = ", inflow_integral, " exact value = -0.19296108663371084")
print("Outflow integral = ", outflow_integral, " exact value = 0.07291330718050365")
print("Top-wall integral = ", top_wall_integral, " exact value = 0.2506982061292486")
print("Bottom-wall integral = ", bottom_wall_integral, " exact value = -0.322029857257521")

    


# Define boundaries and obstacle
#CHANGE PARAMETERS HERE
inflow   = 'near(x[0], 0)'
outflow  = 'near(x[0], 1.0)'
walls    = 'near(x[1], 0) || near(x[1], 1.0)'
# cylinder = 'on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3'
#CHANGE PARAMETERS HERE

# Define inflow profile
g = ('(x[0]*x[0]*x[0]*x[0] + x[1]*x[1]*x[1]*x[1])/48.0')


bcu_inflow = DirichletBC(UV.sub(0), Expression(g, degree=4), inflow)
bcu_outflow = DirichletBC(UV.sub(0), Expression(g, degree=4), outflow)
bcu_walls = DirichletBC(UV.sub(0), Expression(g, degree=4), walls)
# bcu_cylinder = DirichletBC(UV.sub(0), Constant((0, 0)), cylinder)


bc_u = [bcu_inflow, bcu_walls, bcu_outflow]

# Define trial and test functions
nu_u, nu_v = TestFunctions(UV)


# Define functions for solutions at previous and current time steps
uv = TrialFunction(UV)
u, v = split(uv)
uv_ = Function(UV)
u_ = Function(U)
v_ = Function(V)
# h = Function(V)

H = Constant(H)
L = Constant(L)
f = interpolate(f_expression(element=V.ufl_element()), V)

Fu = ( (u.dx(i))*(nu_u.dx(i)) + v*nu_u ) * dx
Fv = ( (v.dx(i)) * (nu_v.dx(i)) + f*nu_v) * dx - ( (L**3)/12.0 * nu_v * ds_outflow + (H**3)/12.0 * nu_v * ds_top_wall )
Fuv = Fu + Fv

a = lhs(Fuv)
L = rhs(Fuv)


xdmffile_check.write(f, 0)
# xdmffile_check.write(h, 0)



# Step 1+2
A = assemble(a)
b = assemble(L)
[bc.apply(A) for bc in bc_u]
[bc.apply(b) for bc in bc_u]

solve(A, uv_.vector(), b, 'bicgstab', 'hypre_amg')
    
u_, v_ = uv_.split(deepcopy=True)
   
# Save solution to file (XDMF/HDF5)
xdmffile_u.write(u_, 0)
xdmffile_v.write(v_, 0)