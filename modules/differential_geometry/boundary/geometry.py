from fenics import *
import numpy as np
import ufl as ufl

import constants.utils as const
import differential_geometry.manifold.geometry as geo
import mesh.load as lmsh
import mesh.utils as msh

alpha, beta, gamma, i, j, k, l = ufl.indices(7)

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
tangent to boundary facets of a two-dimensional mesh, normalized to unity. 
Note: be careful about the direction of this tangent, which is obtained by rotating FacetNormal by \pi/2 counterclockwise as seen by lookng at the xy plane from above (positive z)

Input values: 
    - 'mesh': the mesh
Return values:
    - 't': If the mesh is two-dimensional, [-n[1], n[0]] where n = FacetNormal(mesh). Otherwise [0] *[mesh dimension]
'''

def FacetTangent(mesh):

    n = FacetNormal(mesh)

    if mesh.topology().dim() == 2:
        return as_vector([-n[1], n[0]])
    else: 
        return as_vector([0] * mesh.topology().dim())



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


'''
build the facet tangents to all sub-meshes of a parent mesh
Input values: 
    - 'sub_meshes': the list of sub_meshes of the parent mesh
    - 'mesh_parameters': the dictionary of parameters of the parent mesh

Return values: 
    - 'sub_mesh_facet_tangent': list of tangents to each sub_mesh of the parent mesh
'''

def facet_tangent_sub_meshes(sub_meshes, mesh_parameters):

    sub_mesh_facet_tangent = []

    if ("n_sub_meshes" in mesh_parameters) and (mesh_parameters["n_sub_meshes"] > 1):

        # there are multiple sub-meshes in the parent mesh -> define the facet normal for each sub-mesh

        for p in range(mesh_parameters["n_sub_meshes"]):

            sub_mesh_facet_tangent.append(FacetTangent(sub_meshes[p]))

    return sub_mesh_facet_tangent

# here I define the facet normal vector, which cannot be plotted as a field. It is not a vector in the tangent bundle of \Omega

if "n_meshes" not in lmsh.parameters: 
    # 1 There is only one mesh

    facet_normal = FacetNormal(lmsh.mesh)
    facet_tangent = FacetTangent(lmsh.mesh)

    sub_mesh_facet_normal = facet_normal_sub_meshes(lmsh.sub_meshes, lmsh.parameters)
    sub_mesh_facet_tangent = facet_tangent_sub_meshes(lmsh.sub_meshes, lmsh.parameters)

else:
    # 2 There are multiple meshes

    facet_normal = [None] * lmsh.parameters['n_meshes']
    facet_tangent = [None] * lmsh.parameters['n_meshes']

    sub_mesh_facet_normal = [None] * lmsh.parameters['n_meshes']
    sub_mesh_facet_tangent = [None] * lmsh.parameters['n_meshes']

    for i in range(lmsh.parameters["n_meshes"]):

        facet_normal[i] = FacetNormal(lmsh.mesh[i])
        facet_tangent[i] = FacetTangent(lmsh.mesh[i])

        sub_mesh_facet_normal[i] = facet_normal_sub_meshes(lmsh.sub_meshes[i], lmsh.mesh_parameters[i])
        sub_mesh_facet_tangent[i] = facet_tangent_sub_meshes(lmsh.sub_meshes[i], lmsh.mesh_parameters[i])



'''
return the normal to a mesh as a smooth field
Note: the resulting vector field is not normalized to unity. 
Input values: 
    * Mandatory:
        - 'mesh': the mesh
        - 'n': the normal, e.g., FacetNormal(mesh) or FacetNormal(mesh)('+') ...
        - 'measure': the measure over which the normal will be smoothened, e.g., a ds or a dS of 'mesh'
    * Optional:
        - 'interior': (False by default): True whether 'measure' is of type 'dS', false if it is of type 'ds'
Return values: 
    - the normal as a smooth field
'''
def field_facet_normal(n, mesh, measure, interior=False):

    V = VectorFunctionSpace(mesh, "CG", 2)

    u = TrialFunction(V)
    v = TestFunction(V)

    if interior:
        # the provided measure is an interior measure in the mesh, of type 'dS' -> specify that the quantities in the variational problem are averages over '+' and '-' side

        a = inner(msh.average(u), msh.average(v)) * measure
        l = inner(n, msh.average(v)) * measure

    else:
        # the provided measure is at the mesh boundary, of type 'ds' -> no need to specify that the quantities in the variational problem  are  '+' or '-' 

        a = inner(u, v) * measure
        l = inner(n, v) * measure
    
    A = assemble(a, keep_diagonal=True)
    L = assemble(l)

    A.ident_zeros()
    nh = Function(V)

    solve(A, nh.vector(), L)

    return nh

'''
normal to the manifold pointing outwards the manifold and normalized according to the Euclidean metric, which can be plotted as a field
Input values: 
    * Mandatory:
        - 'mesh': the mesh of which the normal is to be computed
        - 'n': the normal, e.g., FacetNormal(mesh) or FacetNormal(mesh)('+') ...
        - 'measure': the measure of 'mesh' where the normal will be computed
    * Optional:
        - 'norm_threshold': the threshold for normalization of the normal. Entries of the normal whose norm is smaller than norm_threshold will be normalized by norm unity (these entries are irrelevant, because they live in the bulk of the mesh). 
Return values: 
    - 'n_smooth': the facet normal as a smooth field, norma
 '''

def field_facet_normal_normalized(mesh, n, measure,
                        interior=False,
                        norm_threshold = const.vector_norm_threshold):

    # obtain the non-normalized normal 
    n_smooth = field_facet_normal(n, mesh, measure, interior)

    '''
    n_vector contains the DOFs of the vector field n :
        n_vector = [nx_dof0, ny_dof0, nx_dof1, ny_dof1, ...]
    '''
    n_vector = n_smooth.vector().get_local()

    '''
    reshape n_vector in this way:
    Before reshape:
        n_vector = [nx_0, ny_0, nx_1, ny_1, nx_2, y_2, ...]     

    After reshape:
        n_vector = [
            [nx_0, ny_0],
            [nx_1, ny_1],
            [nx_2, ny_2],
            ...
        ]        
    
    '''
    n_vector = n_vector.reshape(-1, 2)  

    '''
    compute the norm of each entry of n_vector and store it into norm_n_vector
    '''
    norm_n_vector  = np.linalg.norm(n_vector, axis=1, keepdims=True)

    '''
    set the norm of vector values in the bulk of the mesh (which are zero), to unity to avoid dividing by zero
    '''
    norm_n_vector  = np.where(norm_n_vector < norm_threshold, 1.0, norm_n_vector)  

    # normalize n_vector by the norm written in norm_n_vector
    n_vector = n_vector / norm_n_vector

    '''
    reshape n_vector by flattenig it, so it has the correct format to be written back into n.vector()
    Before reshape:
        n_vector = [
            [nx_0, ny_0],
            [nx_1, ny_1],
            [nx_2, ny_2],
            ...
        ] 

    After reshape: 
            n_vector = [nx_0, ny_0, nx_1, ny_1, nx_2, y_2, ...]     

    '''

    n_vector = n_vector.reshape(-1)

    '''
    write n_vector into the DOF vector of n
    '''
    n_smooth.vector().set_local(n_vector.reshape(-1))

    n_smooth.vector().apply("insert")
    
    return n_smooth


'''
tangent to a manifold boundary, normalized according to the Euclidean metric, which can be plotted as a field
Input values: 
    * Mandatory:
        - 'mesh': the mesh of which the normal is to be computed
        - 'n': the normal, e.g., FacetNormal(mesh) or FacetNormal(mesh)('+') ...
        - 'measure': the measure of 'mesh' which defines the curve of which the tangent will be computed
    * Optional:
        - 'norm_threshold': the threshold for normalization of the normal. Entries of the normal whose norm is smaller than norm_threshold will be normalized by norm unity (these entries are irrelevant, because they live outside `measure`). 
Return values: 
    - 'n_smooth': the facet normal as a smooth field, norma
 '''
def field_facet_tangent_normalized(mesh, n, measure,
                        interior=False,
                        norm_threshold = const.vector_norm_threshold):
    
    # obtain the smooth, normalized normal
    n_smooth = field_facet_normal_normalized(mesh, n, measure, interior, norm_threshold)

    '''
    n_vector has the form
    [n_DOF_0_x, n_DOF_0_y, n_DOF_1_x, n_DOF_1_y, ...]
    '''
    n_vector = n_smooth.vector().get_local()

    '''
    after reshaping n_vector has the form
    [
        [n_DOF_0_x, n_DOF_0_y], 
        [n_DOF_1_x, n_DOF_1_y,],
    ...]
    '''

    n_vector = n_vector.reshape(-1, 2)


    '''
    take the 0-th and first column of n_vector, n_vector[:, 0] and n_vector[:, 1], respectively, change the sign of the 1th column and stack the two columns to obtain the tangent
    '''
    t_vector = np.stack([-n_vector[:, 1], n_vector[:, 0]], axis=1)


    # allocate a new vector for the smooth tangent
    t_smooth = n_smooth.copy()

    # write into `t_smooth` t_vector flattened
    t_smooth.vector().set_local(t_vector.reshape(-1))
    
    t_smooth.vector().apply("insert")

    return t_smooth


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



'''
return the tangent to mesh boundaries as a smooth field
Note: the resulting vector field is not normalized to unity. 
Input values: 
    - 'mesh': the mesh
Return values: 
    - the tangent as a smooth field
'''
def calc_tangent_cg2(mesh):

    t = FacetTangent(mesh)

    V = VectorFunctionSpace(mesh, "CG", 2)

    u = TrialFunction(V)
    v = TestFunction(V)
    
    a = inner(u, v) * ds
    l = inner(t, v) * ds
    
    A = assemble(a, keep_diagonal=True)
    L = assemble(l)

    A.ident_zeros()

    nh = Function(V)
    solve(A, nh.vector(), L)
    
    return nh