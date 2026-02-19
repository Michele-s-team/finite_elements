from fenics import *
import importlib

import mesh.load as lmsh
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

'''
here Q[i][j] is the scalar function space for the j-th submesh of the i-th mesh, and similarly of other spaces
'''


Q, V, T = [[]], [[]], [[]]
u, nu_u, f, grad_u, J_u, u_exact, hess_u, nu_hess_u, hess_u_exact, J_hess_u = [[]], [[]], [[]], [[]], [[]], [[]], [[]], [[]], [[]], [[]]

for i in range(len(rmsh.lmsh.meshes)):

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

        # Define post-processing (pp) variational problem
        # hess_u is a tensor which is the Hessian matrix of u: hess_u[i, j] = \partial_i \partial_j u
        hess_u[i].append(Function(T[i][j]))
        nu_hess_u[i].append(TestFunction(T[i][j]))
        hess_u_exact[i].append(Function(T[i][j]))
        J_hess_u[i].append(TrialFunction(T[i][j]))

u[1].set_allow_extrapolation(True)

# a function which allows to bridge between sub_mesh[0][1] and sub_mesh[0][0], and thus to impose the BCs for problem on sub_mesh[0][0] in terms of the solution of the problem on sub_mesh[0][1]
v = Function(Q[1])
u_0_1_on_0_0 = Function(Q[0])

