"""
This script computes the first N eigenvalues and eigenfunctions of the Laplacian on a unit square with Dirichlet boundary conditions using finite element methods and SLEPc.

Usage:
    python3 <filename>
"""

import numpy as np
from mpi4py import MPI
from dolfin import *
from petsc4py import PETSc
from slepc4py import SLEPc
import matplotlib.pyplot as plt

def get_first_n_eigenpairs(mesh, alpha, nev=100):
    # 1) P2 finite element space
    V = FunctionSpace(mesh, "Lagrange", 2)
    u_trial = TrialFunction(V)
    v_test = TestFunction(V)

    # 2) Variational forms
    a_form = -alpha * dot(grad(u_trial), grad(v_test)) * dx
    m_form = u_trial * v_test * dx

    # 3) Assembly
    A_full = PETScMatrix()
    M_full = PETScMatrix()
    assemble(a_form, tensor=A_full)
    assemble(m_form, tensor=M_full)

   # 4) Dirichlet boundary conditions
    bcs = DirichletBC(V, Constant(0.0), "on_boundary")
    bdofs = np.array(list(bcs.get_boundary_values().keys()), dtype=np.int32)

    # 5) Index set for interior DOFs
    all_dofs = np.arange(V.dim(), dtype=np.int32)
    interior = np.setdiff1d(all_dofs, bdofs, assume_unique=True)
    is_interior = PETSc.IS().createGeneral(interior, comm=PETSc.COMM_WORLD)

    # 6) Extract interior submatrices
    A_int = A_full.mat().createSubMatrix(is_interior, is_interior)
    M_int = M_full.mat().createSubMatrix(is_interior, is_interior)

    # 7) SLEPc setup
    eps = SLEPc.EPS().create(comm=PETSc.COMM_WORLD)
    eps.setOperators(A_int, M_int)
    eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    eps.setDimensions(nev=nev)
    eps.setTolerances(tol=1e-10, max_it=1000)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
    # Direct solver with MUMPS
    st = eps.getST()
    ksp = st.getKSP()
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")

    eps.solve()
    nconv = eps.getConverged()
    if nconv < 1:
        raise RuntimeError("SLEPc: keine Eigenzahl konvergiert")

    # 8) Collect eigenvalues and eigenfunctions
    eigenvalues = []
    eigenfunctions = []
    for i in range(min(nconv, nev)):
        vr_int, vi_int = A_int.createVecs()
        lam = eps.getEigenpair(i, vr_int, vi_int).real
        eigenvalues.append(lam)
        full_vr = PETSc.Vec().createWithArray(np.zeros(V.dim()), comm=PETSc.COMM_WORLD)
        full_vr.setValues(interior, vr_int.getArray())
        full_vr.assemble()
        u_h = Function(V)
        u_h.vector().set_local(full_vr.getArray())
        u_h.vector().apply("insert")
        eigenfunctions.append(u_h)

    return eigenvalues, eigenfunctions


def main():
    # Mesh resolution per direction
    N = 50
    mesh = UnitSquareMesh(N, N)
    alpha = 3.0
    nev = 50  # Number of eigenvalues

    # Compute eigenpairs
    eigenvalues, eigenfunctions = get_first_n_eigenpairs(mesh, alpha, nev)
    lam1 = eigenvalues[0]
    u1 = eigenfunctions[0]

    # Analytic: generate and sort the first 100 eigenvalues
    m_vals = range(1, 11)
    n_vals = range(1, 11)
    analytic = sorted(
        [-alpha * np.pi**2 * (m**2 + n**2) for m in m_vals for n in n_vals],
        reverse=True
    )[:len(eigenvalues)]

    # Print the first eigenvalue
    true_val1 = -2 * alpha * np.pi**2
    print(f"Computed first eigenvalue: {lam1:.6f}")
    print(f"Expected first eigenvalue: {true_val1:.6f}")

    # Plot comparison of the first N eigenvalues
    indices = np.arange(1, len(eigenvalues) + 1)
    plt.figure()
    plt.plot(indices, eigenvalues, 'o', label='Computed')
    plt.plot(indices, analytic, '-', label='Analytic')
    plt.xlabel('$\lambda$ index')
    plt.ylabel('$\lambda$')
    plt.title(f'Comparison of first {len(eigenvalues)} eigenvalues')
    plt.legend()
    plt.savefig("eigenvalues_comparison.png")
    plt.show()

    # Optional: visualize the first eigenfunction
    plt.figure()
    c = plot(u1)
    plt.title("First Eigenfunction of the Laplacian")
    plt.xlabel("x")
    plt.ylabel("y")
    try:
        plt.colorbar(c)
    except Exception:
        pass  
    plt.savefig("eigenfunction_square_laplacian.png")
    plt.show()


if __name__ == "__main__":
    main()
