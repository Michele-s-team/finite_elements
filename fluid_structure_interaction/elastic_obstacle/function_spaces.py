from fenics import *
import importlib

import load_mesh as lmsh
import switch_problem as swi

rmsh = importlib.import_module(swi.rmsh)

'''
the variables for the problem are
- 'u_n', 'u_n_1': u^n, u^{n-1} in notes
- 'u_dot_n', 'u_dot_n_1', 'u_dot_n_2' : u^n, u^{n-1}, u^{n-2} in notes
- 'v^n' = \textrm{v}^n_notes
- 'v_' = \textrm{v}^*_notes
- 'phi' = \varphi_notes
- 'sigma' = \varsigma_notes
'''

# Define function spaces
# function spaces for the fluid problem
Q_v = VectorFunctionSpace(lmsh.sub_meshes[1], 'P', 2)
Q_v_ = VectorFunctionSpace(lmsh.sub_meshes[1], 'P', 2)
Q_phi = FunctionSpace(lmsh.sub_meshes[1], 'P', 1)

# function spaces for the elastic problem
Q_u_el = VectorFunctionSpace(lmsh.sub_meshes[0], 'P', 1)
Q_u_dot_el = VectorFunctionSpace(lmsh.sub_meshes[0], 'P', 1)
Q_rho_el = FunctionSpace(lmsh.sub_meshes[0], 'P', 1)


# function spaces for the mesh-motion problem
Q_u_msh = VectorFunctionSpace(lmsh.sub_meshes[1], 'P', 1)
Q_u_msh_dot = VectorFunctionSpace(lmsh.sub_meshes[1], 'P', 1)

# function space for the vector dy(s)/ds which represents the tangent to the ellipse curve
Q_ys = VectorFunctionSpace(lmsh.sub_meshes[0], 'P', 2)
Q_dyds = VectorFunctionSpace(lmsh.sub_meshes[0], 'P', 2)

# Define functions for solutions at previous and current time steps
v_n = Function(Q_v)
v_n_1 = Function(Q_v)
v_n_2 = Function(Q_v)
v_ = Function(Q_v_)
# sigma^{n-1/2}
sigma_n_12 = Function(Q_phi)
# sigma^{n-3/2}
sigma_n_32 = Function(Q_phi)
phi = Function(Q_phi)

# fields for the elastic problem
u_el_n = Function(Q_u_el)
u_el_n_1 = Function(Q_u_el)
u_el_n_2 = Function(Q_u_el)

u_el_dot_n = Function(Q_u_dot_el)
u_el_dot_n_1 = Function(Q_u_dot_el)
u_el_dot_n_2 = Function(Q_u_dot_el)

# density field of the elastic body in the reference configuration
rho_el = Function(Q_rho_el)
#deformation field of the elastic body at the inner circular boundary
u_el_circle = Function(Q_u_el)


# fields for the mesh-motion problem
u_msh_n = Function(Q_u_msh)
u_msh_n_1 = Function(Q_u_msh)
u_msh_n_2 = Function(Q_u_msh)

u_msh_dot_n = Function(Q_u_msh_dot)
u_msh_dot_n_1 = Function(Q_u_msh_dot)
u_msh_dot_n_2 = Function(Q_u_msh_dot)

u_ellipse = Function(Q_u_el)
u_square = Function(Q_u_el)
u_dot_ellipse = Function(Q_u_dot_el)
u_dot_square = Function(Q_u_dot_el)

# y_ellipse = {y^s}_notes
ys_ellipse = Function(Q_ys)
# dyds_ellipse = {dy^s/ds}_notes
dyds_ellipse = Function(Q_dyds)

# Define test functions
nu_v_n = TestFunction(Q_v)
nu_v_ = TestFunction(Q_v_)
nu_phi = TestFunction(Q_phi)
nu_u = TestFunction(Q_u_el)
nu_u_dot = TestFunction(Q_u_dot_el)

# Jacobians
J_v_ = TrialFunction(Q_v_)
J_v_n = TrialFunction(Q_v)
J_phi = TrialFunction(Q_phi)
J_u = TrialFunction(Q_u_el)
J_u_dot = TrialFunction(Q_u_dot_el)

V = 0.5 * (v_n_1 + v_)
