from fenics import *
import importlib

import mesh.load as lmsh
import mesh.utils as msh
import parameters.read.solution as rpam
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

'''
The fields in this problem are

* disk fluid:

    - 'v_disk_n' = {\textrm{v}_disk^n}_notes
    - 'v_disk_n_1' = {\textrm{v}_disk^{n-1}}_notes
    - 'v_disk_n_2' = {\textrm{v}_disk^{n-2}}_notes
    - 'v_disk__' = {\overline{\textrm{v}}_disk}_notes

    - 'sigma_disk_n_12' = {\varsigma_disk^{n-1/2}}_notes
    - 'sigma_disk_n_32' = {\varsigma_disk^{n-3/2}}_notes

    - 'phi_disk' = {\phi_disk}_notes
    - 'omega_disk' = omega_notes

    - 'f_di_n' = {\textrm{f}^{disk n}}_notes

* square fluid:

    - 'v_square_n' = {\textrm{v}_square^n}_notes
    - 'v_square_n_1' = {\textrm{v}_square^{n-1}}_notes
    - 'v_square_n_2' = {\textrm{v}_square^{n-2}}_notes
    - 'v_square__' = {\overline{\textrm{v}}_square}_notes

    - 'sigma_square_n_12' = {\varsigma_square^{n-1/2}}_notes
    - 'sigma_square_n_32' = {\varsigma_square^{n-3/2}}_notes

    - 'phi_square' = {\phi_square}_notes

    - 'f_sq_n' = {\textrm{f}^{square n}}_notes
    - 't_sq_n' = {\textrm{t}^n}_notes


* D (domain)

    - disk: 

        x 'u_n_di' = {u^n}_notes in \Omega_disk
        x 'u_n_1_di' = {u^{n-1}}_notes in \Omega_disk
        x 'u_n_2_di' = {u^{n-2}}_notes in \Omega_disk

        x 'u_n_di_dot' = {\dot{u}^n}_notes in \Omega_disk
        x 'u_n_1_di_dot' = {\dot{u}^{n-1}}_notes in \Omega_disk
        x 'u_n_2_di_dot' = {\dot{u}^{n-2}}_notes in \Omega_disk

    - square: 

        x 'u_n_sq' = {u^n}_notes in \Omega_square
        x 'u_n_1_sq' = {u^{n-1}}_notes in \Omega_square
        x 'u_n_2_sq' = {u^{n-2}}_notes in \Omega_square

        x 'u_n_sq_dot' = {\dot{u}^n}_notes in \Omega_square
        x 'u_n_1_sq_dot' = {\dot{u}^{n-1}}_notes in \Omega_square
        x 'u_n_2_sq_dot' = {\dot{u}^{n-2}}_notes in \Omega_square
        


* I (interface): 

    - 'U_n_12' = {U^{n-1/2}}_notes 
    - 'U_n_32' = {U^{n-3/2}}_notes 

    - 'nu_n_12': the stretching field of I 
    - 'psi_n_12': tangent angle of the I. 
        psi_n_12 is decomposed into two parts
            * 'psi_0 =  -2*np.pi*x[0]/rmsh.lmsh.mesh_parameters[1]['L'] is a reference, non-periodic par of psi_n_12, which takes account of the winding of the tangent angle on a closed curve. Note that psi_0 is *not* periodic
            * 'dpsi_n_12': 'psi_n_12'-'psi_0': the deviation of the tangent angle from 'psi_0'. Note that dpsi_n_12 is periodic

    - 'mu_n_12': mean curvature of I 

    - 'n_n_12' = {\hat{n}^{n-1/2}}_notes

    
* M: 

    - 'c_n' = \textrm{c^n}_notes
    - 'c_n_1' = \textrm{c^{n-1}}_notes

    - 'D_c' = {\cal{D}}_notes

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

# function space for the shape curvature projected on sub_mesh[0][0] 
Q_mu_di = FunctionSpace(lmsh.sub_meshes[0][0], 'P', 1)

# this assigner is used to write values into phi_omega_disk (mixed field)
assigner_phi_omega_disk = FunctionAssigner(Q_phi_omega_disk, [Q_phi_disk, Q_omega_disk])

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

# 4.1 displacement field
Q_U = VectorFunctionSpace(lmsh.mesh[1], 'P', 4, dim=2, constrained_domain=periodic_boundary)

# 4.2 nu and psi
# note that the function space on which psi_0 is defined is not periodic
Q_psi_0 = FunctionSpace(lmsh.mesh[1], 'P', 2)


P_nu = FiniteElement('P', interval, 2)
P_dpsi = FiniteElement('P', interval, 2)
element_nu_and_dpsi = MixedElement( [P_nu, P_dpsi] )

Q_nu_and_dpsi = FunctionSpace(lmsh.mesh[1], element_nu_and_dpsi, constrained_domain=periodic_boundary)

Q_nu = Q_nu_and_dpsi.sub(0).collapse()
Q_dpsi = Q_nu_and_dpsi.sub(1).collapse()

# 4.3 curvature
Q_mu = FunctionSpace(lmsh.mesh[1], 'P', 1, constrained_domain=periodic_boundary)

# 5 M

Q_c = FunctionSpace(lmsh.sub_meshes[0][1], 'P', rpam.parameters['c_function_space_degree'])

# this assigner is used to write values into nu_and_dpsi_n_12 (mixed field)
assigner_nu_dpsi = FunctionAssigner(Q_nu_and_dpsi, [Q_nu, Q_dpsi])






#B) field definitions

# 1 disk fluid

# 1.1 v and v_ on disk
v_disk_n = Function(Q_v_disk)
v_disk_n_1 = Function(Q_v_disk)
v_disk_n_2 = Function(Q_v_disk)
v_disk__ = Function(Q_v__disk)

v_disk_n.set_allow_extrapolation(True)
v_disk_n_1.set_allow_extrapolation(True)

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
mu_n_12_1_on_0_0 = Function(Q_mu_di)
f_di_n = Function(Q_v_disk)
# function used to project phi_disk on the function space Q_sigma_disk
phi_disk_on_Q_sigma_disk = Function(Q_sigma_disk)

# this field stores the values of sigma_square_n_32 (defined on sub_mes[0][1]) on sub_mes[0][0]
sigma_square_n_32_0_1_on_0_0 = Function(Q_sigma_disk)

# two auxiliary fields for remeshing
phi_disk_aux = Function(Q_phi_disk)
omega_disk_aux = Function(Q_omega_disk) 


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
t_sq_n = Function(Q_v_square)
# this field is used to store the Dirichlet BCs for v_square__
v_square__bc = Function(Q_v__square)
# this field stores the values of v_disk_n (defined on sub_mes[0][0]) on sub_mesh[0][1]
v_disk_n_0_0_on_0_1 = Function(Q_v__square)
# this field stores the values of v_disk_n_1 (defined on sub_mes[0][0]) on sub_mesh[0][1]
v_disk_n_1_0_0_on_0_1 = Function(Q_v__square)


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

nu_and_dpsi_n_12 = Function(Q_nu_and_dpsi)
nu_n_12, dpsi_n_12 = split( nu_and_dpsi_n_12 )

psi_0 = Function(Q_psi_0)

mu_n_12 = Function(Q_mu)

n_n_12 = Function(Q_U)

U_n_12.set_allow_extrapolation(True)

# 4.2 test functions
nu_U = TestFunction(Q_U)
nu_nu, nu_dpsi = TestFunctions( Q_nu_and_dpsi )
nu_mu = TestFunction(Q_mu)


# 4.3 jacobian
J_U = TrialFunction(Q_U)
J_nu_and_dpsi = TrialFunction(Q_nu_and_dpsi)
J_mu = TrialFunction(Q_mu)


# 4.4 other fields 
# fluid velocity on the disk fluid at step n-1, which lives on sub-mesh[0][0], transferred on the 1d mesh (mesh[1])
v_disk_n_1_0_0_on_1 = Function(Q_U)
# fluid velocity on the square at step n-1, which lives on sub-mesh[0][1], transferred on the 1d mesh (mesh[1])
v_square_n_1_0_1_on_1 = Function(Q_U)

# two-dimensional vector field containing the reference configuration of I as a function of its parameteric coordinate s
ys = Function(Q_U)

U_n_12_smooth = Function(Q_U)

# fields used to set nu_and_dpsi_n_12 after remeshing 
nu_n_12_input = Function(Q_nu)
dpsi_n_12_input = Function(Q_dpsi)





# 5 M

# 5.1 c
c_n = Function(Q_c)
c_n_1 = Function(Q_c)

D_c = Function(Q_c)

# 5.2 test functions
nu_c = TestFunction(Q_c)

# 5.3 jacobian
J_c = TrialFunction(Q_c)





