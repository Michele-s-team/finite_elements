from fenics import *
import load_mesh as lmsh

# -------------------------------------------------------------------
# Mixed Function Space for steady Navier–Stokes (u, p)
# -------------------------------------------------------------------
# Mesh cell type
triangle = lmsh.mesh.ufl_cell()

# Finite elements: P2 for velocity, P1 for pressure
P_v = VectorElement('P', triangle, 2)
P_p = FiniteElement('P', triangle, 1)

# Mixed element and FunctionSpace
Mixed = MixedElement([P_v, P_p])
W       = FunctionSpace(lmsh.mesh, Mixed)

# Subspaces: do not collapse for BCs
Q_v = W.sub(0)   # Velocity subspace (P2)
Q_p = W.sub(1)   # Pressure subspace (P1)

# Mixed unknown, test, and trial
up    = Function(W)            # combined (u,p)
u, p  = split(up)          # u, p components
nu, q  = TestFunctions(W)       # test functions for u,p
J_up  = TrialFunction(W)       # trial for Jacobian

# for stability solving
RT1_elem = FiniteElement("RT", triangle, 1)
Q_div = FunctionSpace(lmsh.mesh, RT1_elem)
# Legacy-Interop brauchen die privaten UFL-Attribute:
Q_div._ufl_element = Q_div.ufl_element()
Q_div._ufl_cell    = Q_div.ufl_cell()