from fenics import *
import importlib

import mesh.load as lmsh
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

function_space_degree = 4

Q, V, T = [], [], []
u, nu_u, f, grad_u, J_u, u_exact, hess_u, nu_hess_u, hess_u_exact, J_hess_u = [], [], [], [], [], [], [], [], [], []
for i in range(len(lmsh.sub_meshes)):
    Q.append(FunctionSpace(lmsh.sub_meshes[i], 'P', function_space_degree))
    V.append(VectorFunctionSpace(lmsh.sub_meshes[i], 'P', function_space_degree))
    T.append(TensorFunctionSpace(lmsh.sub_meshes[i], 'P', function_space_degree, shape=(lmsh.sub_meshes[i].topology().dim(), lmsh.sub_meshes[i].topology().dim())))

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

u[1].set_allow_extrapolation(True)

# a function which allows to bridge between sub_mesh[1] and sub_mesh[0], and thus to impose the BCs for problem on sub_mesh[0] in terms of the solutoin of the problem on sub_mesh[1]
v = Function(Q[1])
u_1_on_0 = Function(Q[0])

