
"""
This script performs stability analysis of the Navier–Stokes steady-state solution by computing the largest eigenvalues and eigenvectors of the linearized operator L.

Steps:
 1. Solve the nonlinear steady-state Navier–Stokes problem for a given Reynolds number.
 2. Linearize around the computed steady solution and assemble the linear operator L.
 3. Compute the leading eigenvalues and eigenvectors of L using SLEPc.
 4. Visualize and save eigenvector fields and the dependence of λ_max on R.

Usage:
    python3 <scriptname> <problem-name> <mesh_path> <solution_path>

"""
import numpy as np
from dolfin import *
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc
import importlib
import switch_problem   as swi
import runtime_arguments as rarg
import matplotlib.pyplot as plt
from dolfin import XDMFFile
import print_out_solution as pr_sol 
import function_spaces_steady as fsp  # here Q_v is your velocity space


def get_largest_eigenvalue_L(V, u_star, mesh, R, nev = 2):
    V = fsp.Q_v.collapse()


    du, v = TrialFunction(V), TestFunction(V)

    # linearized NS operator on velocity only
    a_form = (
        - R * inner(grad(du) * u_star, v)
        - R * inner(grad(u_star) * du, v)
        - inner(grad(du), grad(v))
    ) * mesh.dx

    m_form = inner(du, v) * mesh.dx

    A = PETScMatrix()
    M = PETScMatrix()
    assemble(a_form, tensor=A) 
    assemble(m_form, tensor=M)

    # Collect DOFs of homogeneous Dirichlet on velocity
    bcs = [
        DirichletBC(V, Constant((0.0, 0.0)), mesh.boundary_l),
        DirichletBC(V, Constant((0.0, 0.0)), mesh.boundary_tb),
        DirichletBC(V, Constant((0.0, 0.0)), mesh.boundary_circle),
    ]
    bdofs = np.unique(
        np.hstack([list(bc.get_boundary_values().keys()) for bc in bcs])
    ).astype(np.int32)

    all_dofs = np.arange(V.dim(), dtype=np.int32)
    interior = np.setdiff1d(all_dofs, bdofs, assume_unique=True)
    is_int = PETSc.IS().createGeneral(interior, comm=PETSc.COMM_WORLD)

    A_int = A.mat().createSubMatrix(is_int, is_int)
    M_int = M.mat().createSubMatrix(is_int, is_int)

    # SLEPc solve
    eps = SLEPc.EPS().create(comm=PETSc.COMM_WORLD)
    eps.setOperators(A_int, M_int)
    eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    eps.setDimensions(nev)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
    eps.setTolerances(tol=1e-8, max_it=10000)
    st = eps.getST()
    st.getKSP().setType("preonly")
    st.getKSP().getPC().setType("lu")
    st.getKSP().getPC().setFactorSolverType("mumps")
    eps.solve()
    nconv = eps.getConverged()
    if nconv < 1:
        raise RuntimeError("No eigenvalue converged.")
    lams = []
    for i in range(min(nev, nconv)):
        lam = eps.getEigenpair(i, None, None).real
        lams.append(lam)
    if len(lams) > 1 and abs(lams[0] - lams[1]) < 1e-6:
        print(f"Leading eigenvalue λ₁={lams[0]:.6e} is (nearly) degenerate "
              f"with λ₂={lams[1]:.6e}")
    else:
        print(f"λ₁={lams[0]:.6e}  λ₂={lams[1] if len(lams)>1 else 'n/a'}")
    
    vr, _ = A_int.createVecs()
    lam = eps.getEigenpair(0, vr, None).real

    # 8)Reconstruct full vector
    full = PETSc.Vec().createWithArray(np.zeros(V.dim()), comm=PETSc.COMM_WORLD)
    full.setValues(interior, vr.getArray())
    full.assemble()
    u_h = Function(V)
    u_h.vector().set_local(full.getArray())
    u_h.vector().apply("insert")

    return lam, u_h



def solve_steady():
    # load your mesh module and VP
    rmsh = importlib.import_module(swi.rmsh)
    vp   = importlib.import_module(f"variational_problem_bc_{rarg.args.problem}_steady")

    # solve the full nonlinear steady state
    problem = NonlinearVariationalProblem(vp.F, vp.up, vp.bcs, vp.J)
    solver  = NonlinearVariationalSolver(problem)
    solver.parameters["newton_solver"]["absolute_tolerance"] = 1e-8
    solver.parameters["newton_solver"]["relative_tolerance"] = 1e-6
    solver.parameters["newton_solver"]["maximum_iterations"] = 100
    solver.solve()

    u_star, p_star = vp.up.split()   # u_star is in the pure velocity space
    pr_sol.print_solution_steady(u_star, p_star)
    print('Solved Steady-State Problem')    
    
    return rmsh, u_star


def main():
    Rval = 80
    rarg.args.Re = Rval

    mesh, u_star = solve_steady()
    lam, eigvec = get_largest_eigenvalue_L(fsp.Q_v, u_star, mesh, Rval)
    print(f"Largest eigenvalue of L: {lam:.6e}")
    p = plot(eigvec)
    plt.colorbar(p)
    plt.title("Eigenvector field of L (component-wise)")
    plt.savefig("eigvec_field_operator_L.png")
    plt.show()
    with XDMFFile("eigvec_field_operator_L.xdmf") as xdmf:
        xdmf.write(eigvec, 0.0)

    R_values = range(1, 11)
    lambda_values = []

    for R in R_values:
        print(f"Computing for R = {R} …")
        rarg.args.Re = R
        mesh, u_star = solve_steady()

        lam, _ = get_largest_eigenvalue_L(fsp.Q_v, u_star, mesh, R)
        lambda_values.append(lam)

    # --- Plot λ_max vs R ---
    plt.figure()
    plt.plot(list(R_values), lambda_values, marker='o', linestyle='-')
    plt.xlabel("Reynolds number R")
    plt.ylabel("λ_max")
    plt.title("λ_max vs R")
    plt.grid(True)
    plt.savefig("lambda_max_vs_R_L_operator.png")
    plt.show()
if __name__ == "__main__":
    main()
