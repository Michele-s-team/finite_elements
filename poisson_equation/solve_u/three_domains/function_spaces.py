from fenics import *
import importlib

import mesh.load as lmsh
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

'''
here Q[i][j] is the scalar function space for the j-th submesh of the i-th mesh, and similarly of other spaces
'''


Q, V, T = [], [], []
u, nu_u, f, grad_u, J_u, u_exact, hess_u, nu_hess_u, hess_u_exact, J_hess_u = [], [], [], [], [], [], [], [], [], []

for i in range(len(lmsh.mesh)):

    if "n_sub_meshes" not in lmsh.mesh_parameters[i]:
        # mesh i has no sub-meshes 

        Q.append(FunctionSpace(lmsh.mesh[i], 'P', rpam.parameters['function_space_degree']))
        V.append(VectorFunctionSpace(lmsh.mesh[i], 'P', rpam.parameters['function_space_degree']))
        T.append(TensorFunctionSpace(lmsh.mesh[i], 'P', rpam.parameters['function_space_degree'], shape=(lmsh.mesh[i].topology().dim(), lmsh.mesh[i].topology().dim())))

        # Define variational problem
        u.append(Function(Q[i]))
        nu_u.append(TestFunction(Q[i]))
        f.append(Function(Q[i]))
        grad_u.append(Function(V[i]))
        J_u.append(TrialFunction(Q[i]))
        u_exact.append(Function(Q[i]))

        # Define post-processing (pp) variational problem
        # hess_u is a tensor which is the Hessian matrix of u: hess_u[i, j] = \partial_i \partial_j u
        hess_u.append(Function(T[i]))
        nu_hess_u.append(TestFunction(T[i]))
        hess_u_exact.append(Function(T[i]))
        J_hess_u.append(TrialFunction(T[i]))


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
        hess_u.append([])
        nu_hess_u.append([])
        hess_u_exact.append([])
        J_hess_u.append([])

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


u[0][1].set_allow_extrapolation(True)

# a function which allows to bridge between sub_mesh[0][1] and sub_mesh[0][0], and thus to impose the BCs for problem on sub_mesh[0][0] in terms of the solution of the problem on sub_mesh[0][1]
v = Function(Q[0][1])
u_0_1_on_0_0 = Function(Q[0][0])


