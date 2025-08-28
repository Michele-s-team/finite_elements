from fenics import *
import ufl as ufl

import load_mesh as lmsh

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
