"""
This script performs stability analysis for the Navier–Stokes steady-state solution using the pressure-projection operator P·L.

Main steps:
 1. Solve the nonlinear steady Navier–Stokes problem for a range of Reynolds numbers.
 2. Assemble the linearized operator L and projector matrices D, G for velocity and pressure spaces.
 3. Form the Schur complement M = L - G·(D·G)^{-1}·D·L and compute its largest eigenvalue using SciPy.
 4. Plot the dependence of the largest eigenvalue λ_max on the Reynolds number R.

Usage:
    python3 <script_name> <problem_name> <mesh_path> <solution_path>

"""


import sys, importlib
from fenics import NonlinearVariationalProblem, NonlinearVariationalSolver, parameters, DirichletBC, Constant
import colorama as col

import runtime_arguments as rarg
import switch_problem   as swi
import function_spaces_steady  as fsp
import print_out_solution as pr_sol  
import stability_operators as ops
import numpy as np
from mpi4py import MPI
from dolfin import *
from petsc4py import PETSc
from slepc4py import SLEPc
from fenics import Constant as Cnst
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.sparse.linalg import splu, eigsh


def solve_steady():
    # Load mesh and variational problem
    rmsh = importlib.import_module(swi.rmsh)
    vp   = importlib.import_module(f"variational_problem_bc_{rarg.args.problem}_steady")

    # Pull mixed-space and problem data
    W    = vp.W
    up   = vp.up
    bcs  = vp.bcs
    F    = vp.F
    J    = vp.J

    # Setup and solve nonlinear variational problem
    problem = NonlinearVariationalProblem(F, up, bcs, J)
    solver  = NonlinearVariationalSolver(problem)
    solver.parameters["newton_solver"]["absolute_tolerance"]  = 1e-12
    solver.parameters["newton_solver"]["relative_tolerance"]  = 1e-10
    solver.parameters["newton_solver"]["maximum_iterations"]  = 100
    solver.solve()

    # Extract velocity and pressure
    u_star, p_star = up.split()

    # Output solutions
    pr_sol.print_solution_steady(u_star, p_star)
    print('Solved Steady-State Problem')

    # Return mesh module, velocity-space, velocity solution, and BCs
    return rmsh, fsp.Q_v, u_star

def get_largest_eigenvalue_PL(V,Q, u_star, rmsh, R):

    # Parameter 
    #Re = Constant(0.0)
    Re = R
    # Trial-/Testfunctions
    V = fsp.Q_v.collapse()  # collapse to velocity subspace
    Q = fsp.Q_p.collapse()  # collapse to pressure subspace
    du, v = TrialFunction(V), TestFunction(V)
    p,  q = TrialFunction(Q), TestFunction(Q)
  

    # 1) Bilinear Form L
    a_L = 1/Re*(
        - Re * inner(dot(u_star, nabla_grad(du)), v)
        - Re * inner(dot(du, nabla_grad(u_star)), v)
        - inner(grad(du), grad(v))
    ) * rmsh.dx

    # 2) Mass‐Matrix 
    m_L = inner(du, v) * rmsh.dx

    # (2) D and DG for projector
    #Q = FunctionSpace(mesh, "Lagrange", 1)
    a_DG = -inner(grad(p), grad(q)) * rmsh.dx
    b_D  = + inner(q, div(du))  * rmsh.dx

    # Assemble matrices without BC
    A_L    = PETScMatrix()
    M_m    = PETScMatrix()
    DG_mat = PETScMatrix()
    D_mat  = PETScMatrix()
    assemble(a_L,   tensor=A_L, keep_diagonal = True)
    assemble(m_L, tensor=M_m, keep_diagonal = True)
    assemble(a_DG,  tensor=DG_mat, keep_diagonal = True)
    assemble(b_D,   tensor=D_mat, keep_diagonal = True)

    # ---------------------------------------------------------------
    # (3) BCs

    bc_u_in  = DirichletBC(V, Cnst((0.0,0.0)), rmsh.boundary_l)
    bc_u_w   = DirichletBC(V, Cnst((0.0,0.0)), rmsh.boundary_tb)
    bc_u_cyl = DirichletBC(V, Cnst((0.0,0.0)), rmsh.boundary_circle)
    bc_p_out = DirichletBC(Q, Cnst(0.0),      rmsh.boundary_r)
    bcs_u = [bc_u_in, bc_u_w, bc_u_cyl]
    bdofs_u = np.unique(np.hstack([list(bc.get_boundary_values().keys()) for bc in bcs_u])).astype(np.int32)
    bcs_p = [bc_p_out]
    bdofs_p = np.unique(np.hstack([list(bc.get_boundary_values().keys()) for bc in bcs_p])).astype(np.int32)
    # 3c) inner DOFs, eliminated Dirichlet-BCS
    all_u = np.arange(V.dim(), dtype=np.int32)
    all_p = np.arange(Q.dim(), dtype=np.int32)
    int_u = np.setdiff1d(all_u, bdofs_u, assume_unique=True)
    int_p = np.setdiff1d(all_p, bdofs_p, assume_unique=True)

    # 3d) PETSc Index Sets
    is_u = PETSc.IS().createGeneral(int_u, comm=PETSc.COMM_WORLD)
    is_p = PETSc.IS().createGeneral(int_p, comm=PETSc.COMM_WORLD)
    # ---------------------------------------------------------------

    # (4) Submatrices on Interior-DOFs
    A_L_int = A_L.mat().createSubMatrix(is_u, is_u) #L goes from V->V hence the BCS
    M_m_int = M_m.mat().createSubMatrix(is_u, is_u) # M goes from V->V hence the BCS
    DG_int  = DG_mat.mat().createSubMatrix(is_p, is_p) # DG goes from Q->Q hence the BCS
    D_int   = D_mat.mat().createSubMatrix(is_p, is_u) # D goes from V->Q hence the BCS
    G_full = D_mat.mat().transpose()
    G_int   = G_full.createSubMatrix(is_u, is_p)  # Gradient = Transpose from D_int

    ai_ptr, ai_idx, ai_data = A_L_int.getValuesCSR()
    di_ptr, di_idx, di_data = D_int.getValuesCSR()
    gi_ptr, gi_idx, gi_data = G_int.getValuesCSR()
    m_ptr, m_idx, m_data = M_m_int.getValuesCSR()
    n_u = len(int_u)
    n_p = len(int_p)
    L_sp = sp.csr_matrix((ai_data, ai_idx, ai_ptr), shape=(n_u, n_u))
    D_sp = sp.csr_matrix((di_data, di_idx, di_ptr), shape=(n_p, n_u))
    G_sp = sp.csr_matrix((gi_data, gi_idx, gi_ptr), shape=(n_u, n_p))
    M_mass = sp.csr_matrix((m_data, m_idx, m_ptr), shape=(n_u, n_u))
    K_sp = D_sp.dot(G_sp).tocsc()
    K_lu = splu(K_sp)
    DL_sp = D_sp.dot(L_sp).tocsc()
    DL_dense = DL_sp.toarray() 
    KinvDL = K_lu.solve(DL_dense)
    G_KinvDL = G_sp.dot(KinvDL) 
    M_sp = L_sp - sp.csr_matrix(G_KinvDL)

    eigvals, eigvecs = eigsh(M_sp, k= 1, M = M_mass, which='LA') # k is number of eigenvalues
    for i, lam in enumerate(eigvals,1):
        print(f"λ_{i} = {lam:.6e}")   

    v_max = eigvecs[:,-1]  
    full = np.zeros(V.dim())
    full[int_u] = v_max

    u_h = Function(V)
    u_h.vector().set_local(full)
    u_h.vector().apply("insert")
    return eigvals, u_h
def main():
    Rval = 43
    rarg.args.Re = Rval
    rmsh, Q_v, u_star = solve_steady()
    Q_p = fsp.Q_p
    lam_max = get_largest_eigenvalue_PL(Q_v, Q_p,u_star, rmsh, Rval)
    Rvals = np.linspace(1,60,60)
    lam_list = []    
    for Rval in Rvals:
        print(col.Fore.CYAN + f"\n--- Calculating for R = {Rval} ---" + col.Style.RESET_ALL)
        rarg.args.Re = Rval
        rmsh, Q_v, u_star = solve_steady()
        Q_p = fsp.Q_p
        eigvals, _ = get_largest_eigenvalue_PL(Q_v, Q_p, u_star, rmsh, Rval)
        lam_list.append(eigvals[-1])  # largest eigenvalue

    # Plotting
    plt.figure()
    plt.plot(Rvals, lam_list, marker='o')
    plt.xlabel("Reynoldsnumber Re")
    plt.ylabel("λ_max from P·L")
    plt.title("Stability analysis: λ_max vs. R")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("stability_analysis_test.png")
    plt.show()

if __name__ == "__main__":
    main()


