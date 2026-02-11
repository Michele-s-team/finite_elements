from fenics import *
import ufl as ufl

import differential_geometry.manifold.geometry as geo
import mesh.load as lmsh

alpha, beta = ufl.indices(2)

epsilon = ufl.PermutationSymbol(2)


# Global variables which will be set according to the gauge choice
dydtheta = None
sqrt_deth_circle = None
sqrt_deth_lr = None
sqrt_deth_tb = None
Nt_circle = None
Nn_circle = None
Nn_lr = None
Nn_tb = None
Nt_lr = None
Nt_tb = None
n_circle = None
n_lr = None
n_tb = None

# the facet normal vector, which cannot be plotted as a field. It is not a vector in the tangent bundle of \Omega
facet_normal = FacetNormal(lmsh.mesh)

if ("n_sub_meshes" in lmsh.parameters) and (lmsh.parameters["n_sub_meshes"] > 1):
    # lmsh loads multiple sub-meshes -> define the facet normal for each sub mesh
    sub_mesh_facet_normal = []
    for p in range(lmsh.parameters["n_sub_meshes"]):
        sub_mesh_facet_normal.append(FacetNormal(lmsh.sub_meshes[p]))

i, j, k, l = ufl.indices(4)


def calc_normal_cg2(mesh):
    n = FacetNormal(mesh)
    V = VectorFunctionSpace(mesh, "CG", 2)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(u, v) * ds
    l = inner(n, v) * ds
    A = assemble(a, keep_diagonal=True)
    L = assemble(l)

    A.ident_zeros()
    nh = Function(V)
    solve(A, nh.vector(), L)
    return nh


# the normal to the manifold pointing outwards the manifold and normalized according to the Euclidean metric, which can be plotted as a field
def facet_normal_smooth():
    u = calc_normal_cg2(lmsh.mesh)
    return as_tensor(u[k], (k))

'''
normal to a curve expressed n term of the reference and current configuration of a curve
Input values: 
    - 'ys': a two-dimensional vector for the reference curve configuration
    - 'u': a two-dimensional vector for the displacement field between current and reference configuration
Return values: 
    - 'n': unit normal to the curve in the current configuration (a two-dimensional vector with unit norm)
'''
def n_ale(ys, u):
    V = as_tensor(-epsilon[alpha, beta] * (ys.dx(0)[beta] + u.dx(0)[beta]), (alpha))
    return as_tensor(V[alpha] / geo.ufl_norm(ys.dx(0) + u.dx(0)), (alpha))
