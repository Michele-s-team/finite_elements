from fenics import *
import importlib

import mesh.load as lmsh
import mesh.utils as msh
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

'''
The fields in this problem are

* disk (fluid on disk):

    - 'v_disk_n' = {\textrm{v}_disk^n}_notes
    - 'v_disk_n_1' = {\textrm{v}_disk^{n-1}}_notes
    - 'v_disk_n_2' = {\textrm{v}_disk^{n-2}}_notes
    - 'v_disk__' = {\overline{\textrm{v}}_disk}_notes
    - 'phi_disk' = {\phi_disk}_notes
    - 'omega' = omega_notes
    - 'sigma_n_12_disk' = {\varsigma_disk^{n-1/2}}_notes
    - 'sigma_n_32_disk' = {\varsigma_disk^{n-3/2}}_notes

* square (fluid on square):

    - 'v_square_n' = {\textrm{v}_square^n}_notes
    - 'v_square_n_1' = {\textrm{v}_square^{n-1}}_notes
    - 'v_square_n_2' = {\textrm{v}_square^{n-2}}_notes
    - 'v_square__' = {\overline{\textrm{v}}_square}_notes
    - 'phi_square' = {\phi_square}_notes
    - 'sigma_n_12_square' = {\varsigma_square^{n-1/2}}_notes
    - 'sigma_n_32_square' = {\varsigma_square^{n-3/2}}_notes

* D (domain)

    - square: 

        x 'u_n_sq' = {u^n}_notes in \Omega_square
        x 'u_n_1_sq' = {u^{n-1}}_notes in \Omega_square
        x 'u_n_2_sq' = {u^{n-2}}_notes in \Omega_square

        x 'u_dot_n_sq' = {\dot{u}^n}_notes in \Omega_square
        x 'u_dot_n_1_sq' = {\dot{u}^{n-1}}_notes in \Omega_square
        x 'u_dot_n_2_sq' = {\dot{u}^{n-2}}_notes in \Omega_square
        
    - disk: 

        x 'u_n_di' = {u^n}_notes in \Omega_disk
        x 'u_n_1_di' = {u^{n-1}}_notes in \Omega_disk
        x 'u_n_2_di' = {u^{n-2}}_notes in \Omega_disk

        x 'u_dot_n_di' = {\dot{u}^n}_notes in \Omega_disk
        x 'u_dot_n_1_di' = {\dot{u}^{n-1}}_notes in \Omega_disk
        x 'u_dot_n_2_di' = {\dot{u}^{n-2}}_notes in \Omega_disk


* I (interface): 

    - 'U_n_12' = {U^{n-1/2}}_notes 
    - 'U_n_32' = {U^{n-3/2}}_notes 

    
* M: 

    - 'c_n' = \textrm{c^n}_notes
    - 'c_n_1' = \textrm{c^{n-1}}_notes

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

# 1. function spaces for disk
# 1.1 function space for v and v_
Q_v_disk = VectorFunctionSpace(lmsh.sub_meshes[0][0], 'P', 2)
Q_v__disk = VectorFunctionSpace(lmsh.sub_meshes[0][0], 'P', 2)
Q_sigma_disk = FunctionSpace(lmsh.sub_meshes[0][0], 'P', rpam.parameters['phi_function_space_degree'])


# 1.2  mixed function space for phi_disk and omega
P_phi_disk = FiniteElement('P', msh.element_geometry(lmsh.sub_meshes[0][0]), rpam.parameters['phi_function_space_degree'])
P_omega = VectorElement('P', msh.element_geometry(lmsh.sub_meshes[0][0]), rpam.parameters['phi_function_space_degree'])
phi_disk_omega_element = MixedElement([P_phi_disk, P_omega])
Q_phi_omega_disk = FunctionSpace(lmsh.sub_meshes[0][0], phi_disk_omega_element)
Q_phi_disk = Q_phi_omega_disk.sub(0).collapse()
Q_phi_omega_disk = Q_phi_omega_disk.sub(1).collapse()


# 2. function spaces for square
# 2.1 function space for v and v_
Q_v_square = VectorFunctionSpace(lmsh.sub_meshes[0][1], 'P', 2)
Q_v__square = VectorFunctionSpace(lmsh.sub_meshes[0][1], 'P', 2)
Q_sigma_square = FunctionSpace(lmsh.sub_meshes[0][1], 'P', rpam.parameters['phi_function_space_degree'])


# 3. function spaces for D
# 3.1 D in disk
Q_u_di = VectorFunctionSpace(lmsh.sub_meshes[0][0], 'P', 1)
Q_u_di_dot = VectorFunctionSpace(lmsh.sub_meshes[0][0], 'P', 1)

# 3.2 D in square
Q_u_sq = VectorFunctionSpace(lmsh.sub_meshes[0][1], 'P', 1)
Q_u_sq_dot = VectorFunctionSpace(lmsh.sub_meshes[0][1], 'P', 1)

# 4 function spaces for I 
Q_U = VectorFunctionSpace(lmsh.mesh[1], 'P', rpam.parameters['U_function_space_degree'], constrained_domain=periodic_boundary)

'''

Q, V, T = [], [], []
u, nu_u, f, grad_u, J_u, u_exact, hess_u, nu_hess_u, hess_u_exact, J_hess_u = [], [], [], [], [], [], [], [], [], []

for i in range(len(lmsh.mesh)):

    if "n_sub_meshes" not in lmsh.mesh_parameters[i]:
        # the mesh under consideration has no sub-meshes 

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

        # Define post-processing (pp) variational problem
        # hess_u is a tensor which is the Hessian matrix of u: hess_u[i, j] = \partial_i \partial_j u
        hess_u.append(Function(T[i]))
        nu_hess_u.append(TestFunction(T[i]))
        hess_u_exact.append(Function(T[i]))
        J_hess_u.append(TrialFunction(T[i]))
        


    else:
        # the mesh under consideration has sub-meshes -> run through all sub-meshes and define function spaces and fields

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


# a function which allows to bridge between sub_mesh[0][1] and mesh[1], and thus to impose the BCs for problem on mesh[1] in terms of the solution of the problem on sub_mesh[0][1]
u_0_1_on_1 = Function(Q[1])
# a function which allows to bridge between mesh[1] and sub_mesh[0][0], and thus to impose the BCs for problem on sub_mesh[0][0] in terms of the solution of the problem on mesh[1]
u_1_on_0_0 = Function(Q[0][0])


#  for testing trasnfer - start
# scalar
Q_sub_mesh_0_1 = FunctionSpace(lmsh.sub_meshes[0][1], 'P', rpam.parameters['function_space_degree'])
Q_mesh_1 = FunctionSpace(lmsh.mesh[1], 'P', rpam.parameters['function_space_degree'], constrained_domain=periodic_boundary)

f_sub_mesh_0_1 = Function(Q_sub_mesh_0_1)
f_mesh_1 = Function(Q_mesh_1)


# vector
V_sub_mesh_0_1 = VectorFunctionSpace(lmsh.sub_meshes[0][1], 'P', rpam.parameters['function_space_degree'])
V_mesh_1 = VectorFunctionSpace(lmsh.mesh[1], 'P', rpam.parameters['function_space_degree'], constrained_domain=periodic_boundary, dim=2)

v_sub_mesh_0_1 = Function(V_sub_mesh_0_1)
v_mesh_1 = Function(V_mesh_1)


# tensor
T_sub_mesh_0_1 = TensorFunctionSpace(lmsh.sub_meshes[0][1], 'P', rpam.parameters['function_space_degree'], shape=(2,3))
T_mesh_1 = TensorFunctionSpace(lmsh.mesh[1], 'P', rpam.parameters['function_space_degree'], constrained_domain=periodic_boundary, shape=(2,3))

t_sub_mesh_0_1 = Function(T_sub_mesh_0_1)
t_mesh_1 = Function(T_mesh_1)

#  for testing trasnfer - end


'''