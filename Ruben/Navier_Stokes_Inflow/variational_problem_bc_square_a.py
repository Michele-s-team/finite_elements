# variational_problem_bc_square_a.py

from dolfin import *
import load_2d_mesh as lmsh        # your original mesh loader
import runtime_arguments as rarg   # we will override rarg.args.input_directory

# ─── 1) In‐line the same mesh + marker read as in read_mesh_square.py ────

mesh = lmsh.mesh  # already built when load_2d_mesh was imported

# triangle‐tags (for dx)
mvc_tri = MeshValueCollection("size_t", mesh, mesh.topology().dim())
with XDMFFile(rarg.args.input_directory + "/triangle_mesh.xdmf") as f:
    f.read(mvc_tri, "name_to_read")
cells = MeshFunction("size_t", mesh, mvc_tri)

# line‐tags (for ds)
mvc_line = MeshValueCollection("size_t", mesh, mesh.topology().dim() - 1)
with XDMFFile(rarg.args.input_directory + "/line_mesh.xdmf") as f:
    f.read(mvc_line, "name_to_read")
boundaries = MeshFunction("size_t", mesh, mvc_line)

# ─── 2) Taylor–Hood space ───────────────────────────────────────────────────

P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
W  = FunctionSpace(mesh, P2 * P1)

# ─── 3) Boundary conditions ────────────────────────────────────────────────

inflow_velocity = 1.0

class InflowProfile(UserExpression):
    def eval(self, values, x):
        values[0] = inflow_velocity*4.0*x[1]*(1.0-x[1])
        values[1] = 0.0
    def value_shape(self):
        return (2,)

g        = InflowProfile(degree=2)
bc_inlet = DirichletBC(W.sub(0), g,          boundaries, 1)  # marker=1: inlet
bc_walls = DirichletBC(W.sub(0), Constant((0.0,0.0)), boundaries, 3)  # marker=3: walls+circle
bc_pout  = DirichletBC(W.sub(1), Constant(0.0),    boundaries, 2)  # marker=2: outlet
bcs      = [bc_inlet, bc_walls, bc_pout]



R   = Constant(100.0)
psi = Function(W)
u, p = split(psi)
v, q = TestFunctions(W)

F = ( R*inner(dot(u, nabla_grad(u)), v)*dx
    + inner(grad(u), grad(v))*dx
    - p*div(v)*dx
    + q*div(u)*dx )
J = derivative(F, psi, TrialFunction(W))


