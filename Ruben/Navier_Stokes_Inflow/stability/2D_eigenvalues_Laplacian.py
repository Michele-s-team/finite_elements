"""
This script computes the smallest Laplace eigenvalue and eigenfunction on a custom 2D mesh (with cylindrical boundary) using finite element methods and SLEPc.

Usage:
    python3 2D_eigenvalues_Laplacian.py <problem_type> <mesh_location> <solution_path>
Example:
    python3 2D_eigenvalues_Laplacian.py square ./mesh/solution ./solution
"""



from mpi4py import MPI
from dolfin import *
import importlib
import switch_problem   as swi
from slepc4py import SLEPc
from petsc4py import PETSc
import runtime_arguments as rarg
import numpy as np
import function_spaces_steady as fsp
from dolfin import Mesh, XDMFFile, Measure
from dolfin import MeshValueCollection
import matplotlib.pyplot as plt

def compute_laplacian_eigen(rmsh, degree=2, nev=1, tol=1e-10):

    #1) Define Pdegree finite element space
    V = fsp.Q_p.collapse()
    u = TrialFunction(V)
    v = TestFunction(V)
    n = FacetNormal(rmsh.lmsh.mesh)
    h = CellDiameter(rmsh.lmsh.mesh)
    # 2) Set up variational forms (without minus sign enforcement)
    ymin = 0
    ymax = 1
    linear_bc_expr = Expression("5.0 * (x[1] - ymin) / (ymax - ymin)", degree=1,
                                ymin=ymin, ymax=ymax)
    g= linear_bc_expr
    alpha = Constant(20.0*degree**2)
    a_form = -dot(grad(u), grad(v)) * rmsh.dx
    m_form = u * v * rmsh.dx
    # 3) Assembly
    print(linear_bc_expr)
    A = PETScMatrix(); M = PETScMatrix()
    assemble(a_form, tensor=A)
    assemble(m_form, tensor=M)


    # 4) Dirichlet-BC
    bc_u_in   = DirichletBC(V, Constant(0.0), rmsh.boundary_l)
    bc_u_w    = DirichletBC(V, Constant(0.0), rmsh.boundary_tb)
    bc_u_cyl  = DirichletBC(V, Constant(0.0), rmsh.boundary_circle)
    bc_p_out  = DirichletBC(V, Constant(0.0), rmsh.boundary_r)
    #bc = [bc_u_in, bc_u_w, bc_u_cyl, bc_p_out]
    bc = [bc_p_out]
    #bc = DirichletBC(V, Constant(0.0), rmsh.boundary_l + rmsh.boundary_tb + rmsh.boundary_circle + rmsh.boundary_r)
    all_bdofs = np.hstack([list(b.get_boundary_values().keys()) for b in bc])
    bdofs = np.unique(all_bdofs).astype(np.int32)
    all_dofs = np.arange(V.dim(), dtype=np.int32)
    #interior = np.setdiff1d(all_dofs, bdofs)
    interior = np.setdiff1d(np.arange(V.dim(), dtype=np.int32), bdofs, assume_unique=True)

    # 5) Index-Sets 
    is_int = PETSc.IS().createGeneral(interior, comm=PETSc.COMM_WORLD)
    A_int = A.mat().createSubMatrix(is_int, is_int)
    M_int = M.mat().createSubMatrix(is_int, is_int)

    # 6) SLEPc-Setup
    eps = SLEPc.EPS().create(comm=PETSc.COMM_WORLD)
    eps.setOperators(A_int, M_int)
    eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    eps.setDimensions(nev=nev)
    eps.setTolerances(tol=tol, max_it=1000)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)

    # LU-Preconditioner (MUMPS)
    st = eps.getST()
    ksp = st.getKSP()
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    print("Matrix size:", A_int.getSize(), "  #interior DOFs:", len(interior))
    print("Min/Max interior index:", interior.min(), interior.max())
    eps.solve()
    if eps.getConverged() < 1:
        raise RuntimeError("Keine Eigenzahl konvergiert")

    # 7) Eigenvalue and vector
    vr, _ = A_int.createVecs()
    lam = eps.getEigenpair(0, vr, None).real

    # 8) Reconstruct full vector with the BC
    full = PETSc.Vec().createWithArray(np.zeros(V.dim()), comm=PETSc.COMM_WORLD)
    full.setValues(interior, vr.getArray())
    full.assemble()
    u_h = Function(V)
    u_h.vector().set_local(full.getArray())
    u_h.vector().apply("insert")

    return lam, u_h

def main():
    rmsh = importlib.import_module(swi.rmsh)
    lambda1, u1 = compute_laplacian_eigen(rmsh, degree=2)
    print(f"Kleinste Laplace-Eigenzahl: {lambda1:.6f}")

    plt.figure()
    mesh_plot = plot(u1)
    plt.colorbar(mesh_plot)
    plt.title("First Eigenfunction of Laplacian")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("Laplacian_on_square_circle.png")
    plt.show()

if __name__ == "__main__":
    main()
