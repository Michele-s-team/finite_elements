from fenics import *
import ufl as ufl

import differential_geometry.manifold.geometry as geo
import mesh.load as lmsh

alpha, beta, gamma = ufl.indices(3)

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

'''
build the facet normals to all sub-meshes of a parent mesh
Input values: 
    - 'sub_meshes': the list of sub_meshes of the parent mesh
    - 'mesh_parameters': the dictionary of parameters of the parent mesh

Return values: 
    - 'sub_mesh_facet_normal': list of normals to each sub_mesh of the parent mesh
'''

def facet_normal_sub_meshes(sub_meshes, mesh_parameters):

    sub_mesh_facet_normal = []

    if ("n_sub_meshes" in mesh_parameters) and (mesh_parameters["n_sub_meshes"] > 1):

        # there are multiple sub-meshes in the parent mesh -> define the facet normal for each sub-mesh

        for p in range(mesh_parameters["n_sub_meshes"]):

            sub_mesh_facet_normal.append(FacetNormal(sub_meshes[p]))

    return sub_mesh_facet_normal

# here I define the facet normal vector, which cannot be plotted as a field. It is not a vector in the tangent bundle of \Omega

if "n_meshes" not in lmsh.parameters: 
    # 1 There is only one mesh

    facet_normal = FacetNormal(lmsh.mesh)

    sub_mesh_facet_normal = facet_normal_sub_meshes(lmsh.sub_meshes, lmsh.parameters)

    '''
    if ("n_sub_meshes" in lmsh.parameters) and (lmsh.parameters["n_sub_meshes"] > 1):

        # 1.1 there are multiple sub-meshes of the parent mesh -> define the facet normal for each sub mesh
        sub_mesh_facet_normal = []

        for p in range(lmsh.parameters["n_sub_meshes"]):
            sub_mesh_facet_normal.append(FacetNormal(lmsh.sub_meshes[p]))
    '''

else:
    # 2 There are multiple meshes

    facet_normal = [None] * lmsh.parameters['n_meshes']
    sub_mesh_facet_normal = [None] * lmsh.parameters['n_meshes']

    for i in range(lmsh.parameters["n_meshes"]):

        facet_normal[i] = FacetNormal(lmsh.mesh[i])

        sub_mesh_facet_normal[i] = facet_normal_sub_meshes(lmsh.sub_meshes[i], lmsh.mesh_parameters[i])

        '''
        sub_mesh_facet_normal[i] = []

        if ("n_sub_meshes" in lmsh.mesh_parameters[i]) and (lmsh.mesh_parameters[i]["n_sub_meshes"] > 1):

            # 2.1 there are multiple sub-meshes in the parent mesh[i] -> define the facet normal for each sub mesh

            for p in range(lmsh.mesh_parameters[i]["n_sub_meshes"]):
                sub_mesh_facet_normal[i].append(FacetNormal(lmsh.sub_meshes[i][p]))
        '''


i, j, k, l = ufl.indices(4)

'''
return the normal to a mesh as a smooth field
Input values: 
    - 'mesh': the mesh
Return values: 
    - the unit normal as a smooth field
'''
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
    - 'n_ale': unit normal to the curve in the current configuration (a two-dimensional vector with unit norm)
'''
def n_ale(ys, u):
    V = as_tensor(-epsilon[alpha, beta] * (ys.dx(0)[beta] + u.dx(0)[beta]), (alpha))
    return as_tensor(V[alpha] / geo.ufl_norm(ys.dx(0) + u.dx(0)), (alpha))

'''
variation of n_ale with respect to u
Input values: 
    - 'ys': a two-dimensional vector for the reference curve configuration
    - 'u': a two-dimensional vector for the displacement field between current and reference configuration
    - 'nu': the variation of u, nu = delta_u (two-dimensional vector field)
Return values: 
    - 'delta_n_ale': the variation od n_ale with respect to u (a two-dimensional vector with unit norm)
'''
def delta_n_ale(ys, u, nu):

    dxds = as_tensor((ys.dx(0)[alpha] + u.dx(0)[alpha]), (alpha))
    norm_dxds = geo.ufl_norm(dxds)

    return as_tensor(
        1.0/norm_dxds * (1.0/norm_dxds**2 * dxds[gamma] * nu.dx(0)[gamma] * epsilon[alpha, beta] * dxds[beta] - \
                         epsilon[alpha, beta] * nu.dx(0)[beta]), 
        (alpha))
