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

    - 'sigma_disk_n_12' = {\varsigma_disk^{n-1/2}}_notes
    - 'sigma_disk_n_32' = {\varsigma_disk^{n-3/2}}_notes

    - 'phi_disk' = {\phi_disk}_notes
    - 'omega_disk' = omega_notes

    - 'f_di_n' = \textrm{f}^{disk n}

* square (fluid on square):

    - 'v_square_n' = {\textrm{v}_square^n}_notes
    - 'v_square_n_1' = {\textrm{v}_square^{n-1}}_notes
    - 'v_square_n_2' = {\textrm{v}_square^{n-2}}_notes
    - 'v_square__' = {\overline{\textrm{v}}_square}_notes

    - 'sigma_square_n_12' = {\varsigma_square^{n-1/2}}_notes
    - 'sigma_square_n_32' = {\varsigma_square^{n-3/2}}_notes

    - 'phi_square' = {\phi_square}_notes

    - 'f_sq_n' = \textrm{f}^{square n}


* D (domain)

    - disk: 

        x 'u_n_di' = {u^n}_notes in \Omega_disk
        x 'u_n_1_di' = {u^{n-1}}_notes in \Omega_disk
        x 'u_n_2_di' = {u^{n-2}}_notes in \Omega_disk

        x 'u_dot_n_di' = {\dot{u}^n}_notes in \Omega_disk
        x 'u_dot_n_1_di' = {\dot{u}^{n-1}}_notes in \Omega_disk
        x 'u_dot_n_2_di' = {\dot{u}^{n-2}}_notes in \Omega_disk

    - square: 

        x 'u_n_sq' = {u^n}_notes in \Omega_square
        x 'u_n_1_sq' = {u^{n-1}}_notes in \Omega_square
        x 'u_n_2_sq' = {u^{n-2}}_notes in \Omega_square

        x 'u_dot_n_sq' = {\dot{u}^n}_notes in \Omega_square
        x 'u_dot_n_1_sq' = {\dot{u}^{n-1}}_notes in \Omega_square
        x 'u_dot_n_2_sq' = {\dot{u}^{n-2}}_notes in \Omega_square
        


* I (interface): 

    - 'U_n_12' = {U^{n-1/2}}_notes 
    - 'U_n_32' = {U^{n-3/2}}_notes 

    - 'n_n_12' = {\hat{n}^{n-1/2}}_notes

    
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



#A) function-space definitions


# 1. disk

# 1.1 v and v_ on disk
Q_v_disk = VectorFunctionSpace(lmsh.sub_meshes[0][0], 'P', 2)
Q_v__disk = VectorFunctionSpace(lmsh.sub_meshes[0][0], 'P', 2)

# 1.2 sigma_disk
Q_sigma_disk = FunctionSpace(lmsh.sub_meshes[0][0], 'P', rpam.parameters['phi_function_space_degree'])


# 1.3  mixed function space for phi_disk and omega
P_phi_disk = FiniteElement('P', msh.element_geometry(lmsh.sub_meshes[0][0]), rpam.parameters['phi_function_space_degree'])
P_omega = VectorElement('P', msh.element_geometry(lmsh.sub_meshes[0][0]), rpam.parameters['phi_function_space_degree'])
phi_disk_omega_element = MixedElement([P_phi_disk, P_omega])
Q_phi_omega_disk = FunctionSpace(lmsh.sub_meshes[0][0], phi_disk_omega_element)
Q_phi_disk = Q_phi_omega_disk.sub(0).collapse()
Q_omega_disk = Q_phi_omega_disk.sub(1).collapse()


# 2. square

# 2.1 v and v_ on disk
Q_v_square = VectorFunctionSpace(lmsh.sub_meshes[0][1], 'P', 2)
Q_v__square = VectorFunctionSpace(lmsh.sub_meshes[0][1], 'P', 2)

# 2.2 sigma_square
Q_sigma_square = FunctionSpace(lmsh.sub_meshes[0][1], 'P', rpam.parameters['phi_function_space_degree'])


# 3. D (domain)

# 3.1 D in disk
Q_u_di = VectorFunctionSpace(lmsh.sub_meshes[0][0], 'P', 2)
Q_u_di_dot = VectorFunctionSpace(lmsh.sub_meshes[0][0], 'P', 2)

# 3.2 D in square
Q_u_sq = VectorFunctionSpace(lmsh.sub_meshes[0][1], 'P', 2)
Q_u_sq_dot = VectorFunctionSpace(lmsh.sub_meshes[0][1], 'P', 2)


# 4 I 

Q_U = VectorFunctionSpace(lmsh.mesh[1], 'P', 2, dim=2, constrained_domain=periodic_boundary)


# 5 M

Q_c = FunctionSpace(lmsh.sub_meshes[0][1], 'P', rpam.parameters['c_function_space_degree'])







#B) field definitions

# 1 disk fluid

# 1.1 v and v_ on disk
v_disk_n = Function(Q_v_disk)
v_disk_n_1 = Function(Q_v_disk)
v_disk_n_2 = Function(Q_v_disk)
v_disk__ = Function(Q_v__disk)

# 1.2 sigma_disk
sigma_disk_n_12 = Function(Q_sigma_disk)
sigma_disk_n_32 = Function(Q_sigma_disk)

# 1.3 phi and omega
phi_omega_disk = Function(Q_phi_omega_disk)
phi_disk, omega_disk = split(phi_omega_disk)

# 1.4 test functions
nu_v_disk_n = TestFunction(Q_v_disk)
nu_v_disk__ = TestFunction(Q_v__disk)
nu_phi_disk, nu_omega_disk = TestFunctions(Q_phi_omega_disk)


# 1.5 jacobians
J_v_disk = TrialFunction(Q_v_disk)
J_v__disk = TrialFunction(Q_v__disk)
J_phi_omega_disk = TrialFunction(Q_phi_omega_disk)

# 1.6 other fields
V_di = 0.5 * (v_disk_n_1 + v_disk__)
U_n_12_1_on_0_0 = Function(Q_u_di)
f_di_n = Function(Q_v_disk)

# this field stores the values of sigma_square_n_32 (defined on sub_mes[0][1]) on sub_mes[0][0]
sigma_square_n_32_0_1_on_0_0 = Function(Q_sigma_disk)


# 2 square fluid

# 2.1 v and v_ on square
v_square_n = Function(Q_v_square)
v_square_n_1 = Function(Q_v_square)
v_square_n_2 = Function(Q_v_square)
v_square__ = Function(Q_v__square)

v_square_n_1.set_allow_extrapolation(True)

# 2.2 sigma_square
sigma_square_n_12 = Function(Q_sigma_square)
sigma_square_n_32 = Function(Q_sigma_square)

sigma_square_n_32.set_allow_extrapolation(True)

# 2.3 phi
phi_square = Function(Q_sigma_square)

# 2.4 test functions
nu_v_square_n = TestFunction(Q_v_square)
nu_v_square__ = TestFunction(Q_v__square)
nu_phi_square = TestFunction(Q_sigma_square)


# 2.5 jacobians
J_v_square = TrialFunction(Q_v_square)
J_v__square = TrialFunction(Q_v__square)
J_phi_square = TrialFunction(Q_sigma_square)

# 2.6 other fields
V_sq = 0.5 * (v_square_n_1 + v_square__)
U_n_12_1_on_0_1 = Function(Q_u_sq)
f_sq_n = Function(Q_v_square)
# this field is used to store the Dirichlet BCs for v_square__
v_square__bc = Function(Q_v__square)



# 3 D

# 3.1 disk

# 3.1.1 u 
u_n_di = Function(Q_u_di)
u_n_1_di = Function(Q_u_di)
u_n_2_di = Function(Q_u_di)

# 3.1.2 u_dot
u_n_di_dot = Function(Q_u_di_dot)
u_n_1_di_dot = Function(Q_u_di_dot)
u_n_2_di_dot = Function(Q_u_di_dot)

# 3.1.3 test functions
nu_u_n_di = TestFunction(Q_u_di)
nu_u_n_di_dot = TestFunction(Q_u_di_dot)

# 3.1.4 jacobians
J_u_di = TrialFunction(Q_u_di)
J_u_dot_di = TrialFunction(Q_u_di_dot)

# 3.1.5 other fields
n_n_12_1_on_0_0 = Function(Q_u_di)
# this field stores the values of v_square_n_1 (defined on sub_mes[0][1]) on sub_mes[0][0]
v_square_n_1_0_1_on_0_0 = Function(Q_u_di_dot)
# this field stores the value [\textrm{v}_square^{n-1} . \hat{n}^{n-1/2}] \hat{n}^{n-1/2} coming from the I sector, to be used as a BC for u_n_di_dot on \partial \Omega_O
u_n_di_dot_bc_di = Function(Q_u_di_dot)

# 3.2 square

# 3.2.1 u 
u_n_sq = Function(Q_u_sq)
u_n_1_sq = Function(Q_u_sq)
u_n_2_sq = Function(Q_u_sq)

# 3.2.2 u_dot
u_n_sq_dot = Function(Q_u_sq_dot)
u_n_1_sq_dot = Function(Q_u_sq_dot)
u_n_2_sq_dot = Function(Q_u_sq_dot)

# 3.2.3 test functions
nu_u_n_sq = TestFunction(Q_u_sq)
nu_u_n_sq_dot = TestFunction(Q_u_sq_dot)

# 3.2.4 jacobians
J_u_sq = TrialFunction(Q_u_sq)
J_u_dot_sq = TrialFunction(Q_u_sq_dot)

# 3.2.5 other fields
n_n_12_1_on_0_1 = Function(Q_u_sq)
# this field stores the value [\textrm{v}_square^{n-1} . \hat{n}^{n-1/2}] \hat{n}^{n-1/2} coming from the I sector, to be used as a BC for u_n_sq_dot on \partial \Omega_O
u_n_sq_dot_bc_di = Function(Q_u_sq_dot)


# 4 I 

# 4.1 U
U_n_12 = Function(Q_U)
U_n_32 = Function(Q_U)

n_n_12 = Function(Q_U)

# 4.2 test functions
nu_U = TestFunction(Q_U)

# 4.3 jacobian
J_U = TrialFunction(Q_U)

# 4.4 other fields 
# fluid velocity on the square at step n-1, which lives on sub-mesh[0][1], transferred on the 1d mesh (mesh[1])
v_square_n_1_0_1_on_1 = Function(Q_U)
# two-dimensional vector field containing the reference configuration of I as a function of its parameteric coordinate s
ys = Function(Q_U)




# 5 M

# 5.1 c
c_n = Function(Q_c)
c_n_1 = Function(Q_c)

# 5.2 test functions
nu_c = TestFunction(Q_c)

# 5.3 jacobian
J_c = TrialFunction(Q_c)





