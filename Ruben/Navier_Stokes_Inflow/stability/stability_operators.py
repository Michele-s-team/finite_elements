from dolfin import assemble, inner, grad, dot, div, TrialFunction, TestFunction, dx
from dolfin import as_backend_type
from dolfin import PETScMatrix
from mpi4py import MPI
from petsc4py import PETSc
try:
    from slepc4py import SLEPc
    have_slepc = True
except ImportError:
    have_slepc = False


def build_mass_matrix(V):
    """Assemble mass matrix R on velocity space V."""
    nu_trial = TrialFunction(V)
    nu_test = TestFunction(V)
    R = assemble(inner(nu_test, nu_trial)*dx)
    return as_backend_type(R).mat()


def build_L_matrix(u_star, V, bcs, Re):
    """Assemble linearized Navier–Stokes operator L using base flow u_star."""
    w = TrialFunction(V)
    phi = TestFunction(V)
    a1 = -inner(dot(u_star, grad(w)), phi)*dx
    a2 =  inner(dot(w,   grad(u_star)), phi)*dx
    a3 =  (1.0/Re)*inner(grad(w), grad(phi))*dx
    L = assemble(a1 + a2 + a3)
    # apply homogeneous BCs for perturbation
    for bc in bcs:
        bc.apply(L)
    return as_backend_type(L).mat()


def build_divergence_gradient(V, Q_p):
    """Assemble divergence D and gradient G operators."""
    w = TrialFunction(V)
    q = TestFunction(Q_p)
    D = assemble(div(w)*q*dx)
    p = TrialFunction(Q_p)
    phi = TestFunction(V)
    G = assemble(-inner(phi, grad(p))*dx)
    return as_backend_type(D).mat(), as_backend_type(G).mat()


def build_poisson(D_mat, G_mat, phi_bcs=None):
    from dolfin import TrialFunction, TestFunction, assemble, inner, grad, dx
    from dolfin import as_backend_type
    from petsc4py import PETSc
    import Ruben.Navier_Stokes_Inflow.function_spaces_steady as fsp

    # 1) Collapse the pressure subspace into a full FunctionSpace
    Q_p_full = fsp.Q_p.collapse()
    phi      = TrialFunction(Q_p_full)
    psi      = TestFunction(Q_p_full)

    # 2) Assemble Laplace operator with keep_diagonal=True
    A = assemble(inner(grad(phi), grad(psi)) * dx,
                 keep_diagonal=True)

    # 3) Convert to PETSc and attach constant nullspace
    A_petsc = as_backend_type(A).mat()
    ns      = PETSc.NullSpace().create(constant=True)
    A_petsc.setNullSpace(ns)

    # 4) Apply φ=0 Dirichlet BCs on outflow
    if phi_bcs:
        for bc in phi_bcs:
            bc.apply(A)       # note: apply to the DOLFIN matrix

    return A_petsc

def build_projector(D_mat, G_mat, poisson_mat, phi_bcs=None):
    """Construct function to apply Helmholtz–Hodge projector P = I - G (DG)^{-1}D,
    with Dirichlet BCs on phi."""
    from dolfin import PETScVector, as_backend_type
    # set up KSP for Poisson
    ksp = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
    ksp.setOperators(poisson_mat)
    ksp.setType('cg')
    ksp.getPC().setType('hypre')

    def projector(vec_in):
        # 1) build RHS = D * vec_in
        rhs_petsc = D_mat * vec_in
        rhs = PETScVector(rhs_petsc)
        # 2) apply phi Dirichlet BCs to RHS
        if phi_bcs:
            # PETSc Vec cannot directly bc.apply; use assembled form application hack:
            # project zero on Dirichlet dofs
            for bc in phi_bcs:
                bc.apply(rhs)
        rhs_petsc = as_backend_type(rhs).vec()
        # 3) solve Poisson: poisson_mat * phi = rhs
        phi_petsc = rhs_petsc.duplicate()
        ksp.solve(rhs_petsc, phi_petsc)
        phi = PETScVector(phi_petsc)
        # 4) enforce Dirichlet BC on solution phi
        if phi_bcs:
            for bc in phi_bcs:
                bc.apply(phi)
        # 5) compute correction: G * phi
        phi_petsc = as_backend_type(phi).vec()
        corr = G_mat * phi_petsc
        # 6) subtract
        result = vec_in.copy()
        result.axpy(-1.0, corr)
        return result
    return projector





def assemble_M_matrix(R_mat, L_mat, projector, Re):
    """Assemble full PETSc matrix M = (1/Re) * P * L column-wise."""
    n = L_mat.getSize()[0]
    M = PETSc.Mat().createAIJ(size=(n, n), comm=PETSc.COMM_WORLD)
    M.setUp()
    e = PETSc.Vec().createSeq(n)
    w = PETSc.Vec().createSeq(n)
    for i in range(n):
        e.setValue(i, 1.0)
        L_mat.mult(e, w)
        w = projector(w)
        #w.scale(1.0/Re)
        M.setValues(range(n), [i], w)
        e.setValue(i, 0.0)
    M.assemble()
    return M


def compute_eigenvalues(M, nev=6):
    """Compute nev eigenvalues of matrix M using SLEPc (largest magnitude)."""
    if not have_slepc:
        raise ImportError("SLEPc not available")
    eps = SLEPc.EPS().create(comm=PETSc.COMM_WORLD)
    eps.setOperators(M)
    eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
    eps.setDimensions(nev)
    eps.solve()
    nconv = eps.getConverged()
    ev = []
    for i in range(min(nconv, nev)):
        vr = PETSc.Vec().createSeq(M.getSize()[0])
        ki = eps.getEigenpair(i, vr)
        ev.append((ki.real, ki.imag))
    return ev