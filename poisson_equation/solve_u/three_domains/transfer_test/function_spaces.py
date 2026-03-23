from fenics import *
import importlib
import numpy as np

import mesh.load as lmsh
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

'''
here Q[i][j] is the scalar function space for the j-th submesh of the i-th mesh, and similarly of other spaces
'''

# This enforces periodic boundary conditions which map the l vertex into the r vertex or mesh 1
class PeriodicBoundary(SubDomain):
    # Identify the "target domain": the left vertex
    def inside(self, x, on_boundary):
        return near(x[0], lmsh.mesh_parameters[1]['x_l']) and on_boundary

    # Map the other boundaries to the "target domain"
    def map(self, x, y):
        if near(x[0], lmsh.mesh_parameters[1]['x_r']):
            # right vertex → left vertex
            y[0] = lmsh.mesh_parameters[1]['x_l']
        else:
            # Required: set unmapped points to identity
            y[0] = x[0]
            

periodic_boundary = PeriodicBoundary()

Q, V, T = [], [], []
u, nu_u, f, grad_u, J_u, u_exact, hess_u, nu_hess_u, hess_u_exact, J_hess_u = [], [], [], [], [], [], [], [], [], []

for i in range(len(lmsh.mesh)):

    if "n_sub_meshes" not in lmsh.mesh_parameters[i]:
        # mesh i has no sub-meshes 

        Q.append(FunctionSpace(lmsh.mesh[i], 'P', rpam.parameters['function_space_degree'], constrained_domain=periodic_boundary))
        V.append(VectorFunctionSpace(lmsh.mesh[i], 'P', rpam.parameters['function_space_degree'], constrained_domain=periodic_boundary))
        T.append(TensorFunctionSpace(lmsh.mesh[i], 'P', rpam.parameters['function_space_degree'], shape=(lmsh.mesh[i].topology().dim(), lmsh.mesh[i].topology().dim()), constrained_domain=periodic_boundary))

        # Define variational problem
        u.append(Function(Q[i]))
        nu_u.append(TestFunction(Q[i]))
        f.append(Function(Q[i]))
        grad_u.append(Function(V[i]))
        J_u.append(TrialFunction(Q[i]))
        u_exact.append(Function(Q[i]))


    else:
        # mesh i has sub-meshes -> run through all sub-meshes and define function spaces and fields

        Q.append([])
        V.append([])
        T.append([])

        u.append([])
        nu_u.append([])
        f.append([])
        grad_u.append([])
        J_u.append([])
        u_exact.append([])

        for j in range(len(lmsh.sub_meshes[i])):

            Q[i].append(FunctionSpace(lmsh.sub_meshes[i][j], 'P', rpam.parameters['function_space_degree']))
            V[i].append(VectorFunctionSpace(lmsh.sub_meshes[i][j], 'P', rpam.parameters['function_space_degree']))
            T[i].append(TensorFunctionSpace(lmsh.sub_meshes[i][j], 'P', rpam.parameters['function_space_degree'], shape=(lmsh.sub_meshes[i][j].topology().dim(), lmsh.sub_meshes[i][j].topology().dim())))

            # Define variational problem
            u[i].append(Function(Q[i][j]))
            nu_u[i].append(TestFunction(Q[i][j]))
            f[i].append(Function(Q[i][j]))
            grad_u[i].append(Function(V[i][j]))
            J_u[i].append(TrialFunction(Q[i][j]))
            u_exact[i].append(Function(Q[i][j]))



u[0][1].set_allow_extrapolation(True)


# a function which allows to bridge between sub_mesh[0][1] and mesh[1], and thus to impose the BCs for problem on mesh[1] in terms of the solution of the problem on sub_mesh[0][1]
u_0_1_on_1 = Function(Q[1])
# a function which allows to bridge between mesh[1] and sub_mesh[0][0], and thus to impose the BCs for problem on sub_mesh[0][0] in terms of the solution of the problem on mesh[1]
u_1_on_0_0 = Function(Q[0][0])

# expressions for the exact solution 
class u_exact_expression(UserExpression):
    def eval(self, values, x):

        values[0] = x[0]**2 + np.cos(2.0*np.pi*x[1]/rmsh.lmsh.parameters['h'])

    
    def value_shape(self):
        return (1,)
    
class laplacian_u_exact_expression(UserExpression):
    def eval(self, values, x):

        values[0] = 2 - (2.0*np.pi/rmsh.lmsh.parameters['h'])**2 * np.cos(2.0*np.pi*x[1]/rmsh.lmsh.parameters['h'])

    def value_shape(self):
        return (1,)


'''
#  for testing trasnfer - start
# scalar
Q_sub_mesh_0_0 = FunctionSpace(lmsh.sub_meshes[0][0], 'P', rpam.parameters['function_space_degree'])
Q_sub_mesh_0_1 = FunctionSpace(lmsh.sub_meshes[0][1], 'P', rpam.parameters['function_space_degree'])
Q_mesh_1 = FunctionSpace(lmsh.mesh[1], 'P', rpam.parameters['function_space_degree'], constrained_domain=periodic_boundary)


f_sub_mesh_0_0 = Function(Q_sub_mesh_0_0)
f_sub_mesh_0_1 = Function(Q_sub_mesh_0_1)
f_mesh_1 = Function(Q_mesh_1)


# vector
V_sub_mesh_0_0 = VectorFunctionSpace(lmsh.sub_meshes[0][0], 'P', rpam.parameters['function_space_degree'])
V_sub_mesh_0_1 = VectorFunctionSpace(lmsh.sub_meshes[0][1], 'P', rpam.parameters['function_space_degree'])
V_mesh_1 = VectorFunctionSpace(lmsh.mesh[1], 'P', rpam.parameters['function_space_degree'], constrained_domain=periodic_boundary, dim=2)

v_sub_mesh_0_0 = Function(V_sub_mesh_0_0)
v_sub_mesh_0_1 = Function(V_sub_mesh_0_1)
v_mesh_1 = Function(V_mesh_1)


# tensor
T_sub_mesh_0_0 = TensorFunctionSpace(lmsh.sub_meshes[0][0], 'P', rpam.parameters['function_space_degree'], shape=(2,3))
T_sub_mesh_0_1 = TensorFunctionSpace(lmsh.sub_meshes[0][1], 'P', rpam.parameters['function_space_degree'], shape=(2,3))
T_mesh_1 = TensorFunctionSpace(lmsh.mesh[1], 'P', rpam.parameters['function_space_degree'], constrained_domain=periodic_boundary, shape=(2,3))

t_sub_mesh_0_0 = Function(T_sub_mesh_0_0)
t_sub_mesh_0_1 = Function(T_sub_mesh_0_1)
t_mesh_1 = Function(T_mesh_1)
'''
#  for testing trasnfer - end


