import colorama as col
import command as cmd
from fenics import *
import gmsh
import math
import meshio
import numpy as np
import os
import pygmsh
import shutil
import sys
import ufl

import calculus as cal
import constants.utils as const
import differential_geometry.manifold.geometry as geo
import function as fu
import geometry.utils as geo_u
import input_output as io

alpha = ufl.indices(1)


def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:, :2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(
        points=points, cells={cell_type: cells}, cell_data={"name_to_read": [cell_data]}
    )
    return out_mesh


'''
read the mesh from xdmf file
Input values: 
- 'file name': path and name of the xdmf file
Return values: 
- 'mesh': the mesh
'''


def read_mesh_xdmf(filename):
    mesh = Mesh()

    xdmf = XDMFFile(mesh.mpi_comm(), filename)
    xdmf.read(mesh)
    xdmf.close()

    return mesh


'''
read the mesh from h5 file
Input values: 
- 'file name': path and name of the h5 file
- 'mesh_name' [optional]: the name of the mesh in the file
Return values: 
- 'mesh': the mesh
'''


def read_mesh_h5(filename, mesh_name='mesh'):
    mesh = Mesh()
    with HDF5File(mesh.mpi_comm(), filename, "r") as infile:
        infile.read(mesh, mesh_name, False)
    return mesh


'''
Read a mesh from file
Input values: 
- 'file name': path and name of the file, which can be either an xdmf file or h5 file
Return values:
- 'mesh': the mesh
'''


def read_mesh(filename):
    # detect format from file extension
    if filename.endswith('.h5'):
        file_format = "h5"
    elif filename.endswith('.xdmf'):
        file_format = "xdmf"
    else:
        raise ValueError(f"File extension is invalid: {filename}")

    if file_format == "h5":
        return read_mesh_h5(filename)
    elif file_format == "xdmf":
        return read_mesh_xdmf(filename)
    else:
        print(f"File extension is invalid: {filename}")


'''
read the mesh  from  the .msh file 'infile' and write the mesh components (tetrahedra, triangles, lines, vertices) to 'outfile' (tetra_mesh.xdmf, triangle_mesh.xdmf ...)
the component type can be "tetra", "triangle", "line" or "vertex"
if 'prune_z' = true (false), the z component will be removed from the mesh
'''


def write_mesh_components(infile, outfile, component_type, prune_z):
    mesh_from_file = meshio.read(infile)
    # print(f'type of mesh_from_file  = {type(mesh_from_file)}')
    component_mesh = create_mesh(mesh_from_file, component_type, prune_z)
    # print(f'type of component _mesh  = {type(component_mesh)}')
    meshio.write(outfile, component_mesh)


'''
write to .h5 file the components of a mesh determined by a MeshFunction
Input values: 
- 'mesh': the mesh
- 'file_name': the .h5 file where the component will be written
- 'componand_function': the MeshFunction that specifies the component
- 'component_name': the name with which the component will be named in the output file
Example of usage:
    msh.write_mesh_components_h5(mesh_t, io.add_trailing_slash(rarg.args.output_directory) + "line_mesh.h5", cf_t, "cf")
'''


def write_mesh_components_h5(mesh, filename, component_function, component_name):
    with HDF5File(mesh.mpi_comm(), filename, "w") as outfile:
        outfile.write(mesh, "mesh")
        outfile.write(component_function, component_name)


'''
Given a mesh written in an xdmf file, read its components stored into the xdmf file and return the collection of components
Input values: 
- 'mesh': the mesh to read the components from
- 'dim': the dimension of the components to read: example: 1 for lines, 0 for vertices, etc. 
- 'filename': the name of the xdmf file where the components of the mesh are stored
Example: to read the lines of the mesh, call this method with 
    cf = msh.read_mesh_components_xdmf(mesh, 1, args.input_directory + "/line_mesh.xdmf")
'''


def read_mesh_components_xdmf(mesh, dim, filename):
    mesh_value_collection = MeshValueCollection("size_t", mesh, dim)
    with XDMFFile(filename) as infile:
        infile.read(mesh_value_collection, "name_to_read")
        infile.close()
    return cpp.mesh.MeshFunctionSizet(mesh, mesh_value_collection)


'''
Given a mesh written in an h5 file, read its components  stored in an h5 file and returns the collection of components
Input values: 
- 'mesh': the mesh to read the components from
- 'dim': the dimension of the components to read: example: 1 for lines, 0 for vertices, etc. 
- 'filename': the name of the h5 file where the components of the mesh are stored
'''


def read_mesh_components_h5(mesh, dim, filename, name_to_read):
    mesh_function = MeshFunction("size_t", mesh, dim)
    with HDF5File(mesh.mpi_comm(), filename, "r") as infile:
        infile.read(mesh_function, name_to_read)
    return mesh_function


'''
Given a mesh written in a file, read its components stored into the file and return the collection of components
Input values: 
- 'mesh': the mesh to read the components from
- 'dim': the dimension of the components to read: example: 1 for lines, 0 for vertices, etc. 
- 'filename': the name of the file (either .h5 or .xdmf) where the components of the mesh are stored
'''


def read_mesh_components(mesh, dim, filename, name_to_read="name_to_read"):
    # detect format from file extension
    if filename.endswith('.h5'):
        file_format = "h5"
    elif filename.endswith('.xdmf'):
        file_format = "xdmf"
    else:
        raise ValueError(f"File extension is invalid: {filename}")

    if file_format.lower() == "h5":

        return read_mesh_components_h5(mesh, dim, filename, name_to_read)


    elif file_format.lower() == "xdmf":
 
        return read_mesh_components_xdmf(mesh, dim, filename)

    else:
        raise ValueError(f"Unsupported file format: {file_format}")

'''
returns a mesh function that tags internal facets of a 2d mesh contained into two neighboring surfaces
Input values:   
    - 'mesh': the mesh
    - 'sf': the mesh function tagging mesh surfaces of the mesh
    - 'surface_a_id', 'surface_b_id': ID with which the neighboring surfaces have been tagged
    - 'boundary_ab_id': ID with which the 1d boundary between the two surfaces has been tagged

Return values"
    - 'mf': a mesh function where internal facets of surface_a are tagged with 'surface_a_id', the internal facets of surface_b are tagged with 'surface_b_id', and the facets at the interface between surface_a and surface_b are tagged with 'boundary_ab_id'
'''
def read_mesh_internal_components(mesh, sf, surface_a_id, surface_b_id, boundary_ab_id):

    # build a function mf_I that tags interior lines and allows for reading them
    mf = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)

    mesh.init(1, 2)   # build facet-to-cell connectivity

    for facet in facets(mesh):

        if facet.exterior() == False:
            # the facet under consideration does not to the exterior of the mesh -> it is an internal facet

            # print(f'facet {facet.index()} belongs to the interior of the mesh, vertices: {[v.index() for v in vertices(facet)]}')

            # consider the cells that have 'facet' as one of their boundary facets, and put their tag in the list 'cell_tags', which will contain two cells
            cell_tags = [sf[Cell(mesh, cell_id)] for cell_id in facet.entities(2)]

            if all(c == surface_a_id for c in cell_tags):
                # all cells that have facet as one of their boundary facets belong to l_surface -> the facet under consideration is an internal facet of l_surface -> tag this facet in mf_I with ID l_surface_id

                mf[facet] = surface_a_id

            elif all(c == surface_b_id for c in cell_tags):
                # all cells that have 'facet' as one of their boundary facets belong to r_surface -> the facet under consideration is an internal facet of r_surface -> tag this facet in mf_I with ID r_surface_id

                mf[facet] = surface_b_id

            else:
                # one of the two cells that have 'facet' as one of their boundary facets belongs to l_surface, and the other to r_surface -> the facet under consideration is an internal facet coinciding with 'm_line' -> tag this facet in mf_I with ID 'm_line_id'

                mf[facet] = boundary_ab_id

                # print(f'facet {facet.index()} belongs to both l_surface and and r_surface, vertices: {[v.index() for v in vertices(facet)]}')

    return mf
 
   


'''
compare the numerical value of the integral of a test function over a ds, dx, .... with the exact one and output the relative difference and prints out the difference
Input values: 
- 'exact_value': the exact value of the integral
- 'f_test': the function to integrate
- 'meashre': the integration measure
- 'label': the label to be printed out for the integral test
Return values: 
- the absolute value of the relative difference between the finite-element and the exact integral
'''


def test_mesh_integral(exact_value, f_test, measure, label):
    numerical_value = assemble(f_test * measure)

    result = abs((numerical_value - exact_value) / exact_value)
    print(
        f"{label} = {numerical_value:.{4}}, should be {exact_value:.{4}}, relative error =  {result:.{io.number_of_decimals}e}")

    return result


class BoundaryMarker(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

'''
print the coordinates of a vertex
Input values: 
    - 'vertex' <class 'dolfin.cpp.mesh.Vertex'>: the vertex
Return values: 
    - a list containing the coordinates of the vertex
'''
def vertex_coordinates(vertex):
        
    return [vertex.point().x(), vertex.point().y(), vertex.point().z() ]


'''
coumpute the points on a tagged boundary of a 2d mesh and returns them in an ordered way (in the order in which they are connected by edges: vertex[0], then the vertex to which vertex[0] is connected, ..)
Input values: 
    * Mandatory: 
        - 'mesh': the mesh
        - 'mesh_path': the path where 'triangle_mesh.xdmf' and 'line_mesh.xdmf' are located
        - 'id': a list of tags tag of the boundary whose vertices will be computed
    * Optional: 
        - 'outfile': path, name and extension of the csv file where the vertex coordinates will be printed 

'''
def sorted_boundary_points(mesh, mesh_path, id, outfile=None):
    
    mf = read_mesh_components(mesh, mesh.topology().dim()-1, os.path.join(mesh_path, "line_mesh.xdmf"))

    # build a list of facets which lie on the boundary of the mesh
    facet_list = []
    for facet in facets(mesh):  
        # run through all facets of the mesh  
        if mf[facet] in id:
            # the ID of the facet under consideration is equal to one of the IDs in 'id' -> add it to facet_list
            facet_list.append(facet)
                
    # print(f'\n\t facet list = {facet_list}')
      
    #initialize list of vertices   
    vertex_list = []
    
    # add the first vertex to exterior vertex and delete the corresponding edge in exterior_facets
    vertex_list.append(next(vertices(facet_list[0])))
    del facet_list[0]
    
    
    # loop through exterior_facets to append the vertices connected, through a facet, to the last added vertex in exterior_vertex
    while len(facet_list) > 0:

        # append the next vertex: loop through facets
        found = False
        for i in range(len(facet_list)):   
            
            if found:
                break
                          
            # loop through vertices in the facet under consideration
            for v in vertices(facet_list[i]): 
                
                # if the vertex is equal to the last vertex in exterior_vertices, append the *other* vertex in the facet to exterior_vertices and stop the loops and delete the facet under consideration from exterior_facets so it will not be reconsidered at next iterations
                if v.index() == vertex_list[-1].index():
                    # print(f'Added new vertex from facet = {exterior_facets[i]}')
                    
                    # Find the other vertex (the one that is not v)
                    # Get all vertices of this facet as a list
                    facet_vertices = list(vertices(facet_list[i]))
                    other_vertex = [vertex for vertex in facet_vertices if vertex.index() != v.index()][0]

                    # append other_vertex to exterior_vertices
                    vertex_list.append(other_vertex)
                    del facet_list[i]
                    
                    found = True
                    break

    # print(f'vertices:')
    # for v in vertex_list:
    #     print(f'\t{vertex_coordinates(v)}')

                   

    if outfile != None:
        
        csvfile = open(outfile, "w" )

        print(f"\":0\",\":1\",\":2\"", file=csvfile)

        for v in vertex_list:
            coordinates = vertex_coordinates(v)
            print( f"{coordinates[0]},{coordinates[1]},{coordinates[2]}", file=csvfile)
            
        csvfile.close()

'''
return the coordinates of the boundary points of a mesh
Input values: 
    * Mandatory: 
        - 'mesh': the mesh of which the boundary points
    * Optional: 
        - 'filename': path, name and extension of the csv file where the coordinates will be stored. If 'filename' is None,  coordinates will not be stored on file. 
'''
def boundary_points(mesh):
    # create a dummy function space of degree 1 which will be used only to extract the boundary points
    Q_dummy = FunctionSpace(mesh, 'CG', 1)

    # a map which takes as an input a vertex of Q_dummy.mesh and returns its corresponding degree of freedom
    vertex_to_degree_of_freedom_map = vertex_to_dof_map(Q_dummy)

    # a function which takes as argument the mesh vertices
    vertex_function = MeshFunction("size_t", mesh, 0)

    # set vertex_function -> 1 on the vertices which are part of the boundary (vertex_function is zero elsewhere)
    vertex_function.set_all(0)
    BoundaryMarker().mark(vertex_function, 1)

    # collect the vertices where the vertex_function = 1, i.e., the vertices on the boundary
    boundary_vertices = np.asarray(vertex_function.where_equal(1))

    degrees_of_freedom = vertex_to_degree_of_freedom_map[boundary_vertices]

    tab_degrees_of_freedom = Q_dummy.tabulate_dof_coordinates()
    coordinates = tab_degrees_of_freedom[degrees_of_freedom]
   

    return coordinates


# returns the bulk points of the mesh `mesh`
def bulk_points(mesh):
    # create a dummy function space of degree 1 which will be used only to extract the boundary points
    Q_dummy = FunctionSpace(mesh, 'CG', 1)

    # a map which takes as an input a vertex of Q_dummy.mesh and returns its corresponding degree of freedom
    vertex_to_degree_of_freedom_map = vertex_to_dof_map(Q_dummy)

    # a function which takes as argument the mesh vertices
    vertex_function = MeshFunction("size_t", mesh, 0)

    # set vertex_function -> 1 on the vertices which are part of the boundary (vertex_function is zero elsewhere)
    vertex_function.set_all(0)
    BoundaryMarker().mark(vertex_function, 1)

    # collect the vertices where the vertex_function = 0, i.e., the vertices in the bulk
    boundary_vertices = np.asarray(vertex_function.where_equal(0))

    degrees_of_freedom = vertex_to_degree_of_freedom_map[boundary_vertices]

    x = Q_dummy.tabulate_dof_coordinates()
    x = x[degrees_of_freedom]

    # csvfile = open( "test_bulk_points.csv", "w" )
    # for p in x:
    #     print( f"{p[0]},{p[1]}", file=csvfile )
    # csvfile.close()

    # print("Degrees of freedom on the boundary:")
    # for degree_of_freedom in degrees_of_freedom:
    # print(f"\t{x[degree_of_freedom]}, {geo.np.linalg.norm( x[degree_of_freedom])}")

    return x


# return the set of boundary points whose distance from the point c lies between r and R
def boundary_points_circle(mesh, r, R, c):
    points = boundary_points(mesh)

    x = []
    for point in points:
        if ((geo.np.linalg.norm(point - c) > r) and (geo.np.linalg.norm(point - c) < R)):
            x.append(point)

    # csvfile = open( "test_boundary_points_circle.csv", "w" )
    # for p in x:
    #     print( f"{p[0]},{p[1]}", file=csvfile )
    # csvfile.close()

    return x


# compute the lowest and largest x and y values of points in the mesh and return them as a vector in the format [[x_min, x_max], [y_min, y_max]]
def extremal_coordinates(mesh):
    points = boundary_points(mesh)

    if mesh.topology().dim() == 2:

        x_min = points[0][0]
        x_max = x_min
        y_min = points[0][1]
        y_max = y_min

        for point in points:
            if point[0] < x_min:
                x_min = point[0]

            if point[0] > x_max:
                x_max = point[0]

            if point[1] < y_min:
                y_min = point[1]

            if point[1] > y_max:
                y_max = point[1]

        # print(f"\textremal coordinates: {x_min}, {x_max}, {y_min}, {y_max}")

        return [[x_min, x_max], [y_min, y_max]]

    elif mesh.topology().dim() == 1:

        x_min = points[0][0]
        x_max = x_min

        for point in points:
            if point[0] < x_min:
                x_min = point[0]

            if point[0] > x_max:
                x_max = point[0]


        return [x_min, x_max]
    
    
'''
compute the size of the mesh in each dimension
Input values: 
    - 'mesh': the mesh
Return values:
    - [size_x, size_y, ...]: the difference between the lasgest and the smallest coordinate in each dimension
'''

def compute_size(mesh):
    
    if mesh.topology().dim() == 1:
        [x_min, x_max] = extremal_coordinates(mesh)
        
        return x_max - x_min
        
    elif mesh.topology().dim() == 2:
        [[x_min, x_max], [y_min, y_max]] = extremal_coordinates(mesh)
        
        return [x_max - x_min, y_max - y_min]
    
'''
compute the difference between functions f and g on the boundary of the mesh on which f and g are defined, returning 
sqrt(\sum_{i \in {vertices in the boundary of the mesh} [f(x_i) - g(x_i)]^2/ (number of vertices in the boundary of the mesh})
'''


def difference_on_boundary(f, g):
    mesh = f.function_space().mesh()
    boundary_points_mesh = boundary_points(mesh)

    # print("\n\nx\tf(x)-g(x)")
    diff = 0.0
    for x in boundary_points_mesh:
        delta = f(x) - g(x)
        diff += (delta ** 2)

    diff = np.sqrt(diff / len(boundary_points_mesh))

    return diff


'''
compute the difference between functions f and g in the bulk of the mesh on which f and g are defined, returning 
sqrt(\sum_{i \in {vertices in the bulk of the mesh} [f(x_i) - g(x_i)]^2/ (number of vertices in the bulk of the mesh})
'''


def difference_in_bulk(f, g):
    mesh = f.function_space().mesh()
    bulk_points_mesh = bulk_points(mesh)

    diff = 0.0
    for x in bulk_points_mesh:
        delta = f(x) - g(x)
        diff += (delta ** 2)

    diff = np.sqrt(diff / len(bulk_points_mesh))

    return diff


# return sqrt(<(f-g)^2>_measure / <measure>), where measure can be dx, ds_...
def difference_wrt_measure(f, g, measure):

    return sqrt(assemble(((f - g) ** 2 * measure)) / assemble(Constant(1.0) * measure))


# return sqrt(<f^2>_measure / <measure>), where measure can be dx, ds_...
def abs_wrt_measure(f, measure):
    
    return difference_wrt_measure(f, Constant(0), measure)



'''
returns the average of a field `f` (scalar, vector or tensor) with respect to a measure
Input values: 
    - `f`: the field 
    - `measure`: the measure, e.g., `dx`

Return values: 
    - (int f * measure )/ (int measure)
'''
def average_wrt_measure(f, measure):

    return assemble(f * measure ) / assemble(Constant(1.0) * measure)


'''
compute the difference between functions f and g on the boundary of the mesh, boundary_c, given by the boundary points whose distance from point c lies between r and R, returning 
sqrt(\sum_{i \in {vertices in boundary_c} [f(x_i) - g(x_i)]^2/ (number of vertices in boundary_c})
'''


def difference_on_boundary_circle(f, g, r, R, c):
    mesh = f.function_space().mesh()
    boundary_c_points = boundary_points_circle(mesh, r, R, c)

    diff = 0.0
    for x in boundary_c_points:
        delta = f(x) - g(x)
        diff += (delta ** 2)

    diff = np.sqrt(diff / len(boundary_c_points))

    return diff


'''
NOTE: This method is different from input_output.print_mesh_vertices_to_csv. 

print the mesh vertices tags and coordinates to csv file

Input values: 
    - `infile`: full path of the input msh file
    - `outfile`: full path of the output csv file

    Return values: 
    The output csv file is
    tag,:0,:1,:2
    tag_vertex_0,vertex_0_x_coord,vertex_0_y_coord,vertex_0_z_coord
    tag_vertex_1,vertex_1_x_coord,vertex_1_y_coord,vertex_1_z_coord
    ...

    The tag convention is the same used in mesh.utils.print_mesh_edges_to_csv
'''

def print_mesh_vertices_to_csv(infile, outfile):

    # initialize gmsh
    gmsh.initialize()

    # open the .msh file
    gmsh.open(infile)

    # create the path for the csv file if it does not exist
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    # construct a map which, given the tag of a node, gives its coordinates
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    nodes = [[node_tags[i], list(node_coords[3 * i: 3 * (i + 1)])] for i in range(len(node_tags))]

    # sort nodes in increasing order of `tag`
    nodes.sort(key=lambda n: n[0])      


    csvfile = open(outfile, "w")
    print(f"tag,:0,:1,:2", file=csvfile)

    for node in nodes:

        print(f"{node[0]},{node[1][0]},{node[1][1]},{node[1][2]}", file=csvfile)

    csvfile.close()

    # finalize gmsh
    gmsh.finalize()




'''
write to csv file 'outfile' the coordinates of the start and end vertices which define the lines (edges) of a mesh

Input values: 
    - 'infile': the .msh file where the mesh is stored
    - 'outfile': the .csv file where the edges will be stored, in the format

    p_1,p_2
    id_p_1_edge_0,id_p_2_edge_0
    id_p_1_edge_1,id_p_2_edge_1
    ...
'''

def print_mesh_edges_to_csv(infile, outfile):

    # initialize gmsh
    gmsh.initialize()

    # open the .msh file
    gmsh.open(infile)

    mesh_dimension = gmsh.model.getDimension()   # highest entity dimension present

    '''
    get the list of components with dimension equal to the mesh dimension: 
        - for a 1d mesh `components` are edges
        - for a 2d mesh `components` are triangles
        - for a 3d mesh `components` are tetrahedre
    ''' 
    components = gmsh.model.mesh.getElements(dim=mesh_dimension)

    # construct a map which, given the tag of a node, gives its coordinates
    # node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    # node_map = {node_tags[i]: node_coords[3 * i: 3 * (i + 1)] for i in range(len(node_tags))}
    # print( "node map = ", node_map )

    # Store unique edges from the triangle elements
    # initialize a 'list' of unique elements, this sets the list to empty
    edges = set()

    if mesh_dimension == 1: 

        # check that mesh components are segments
        if list(components[0]) != [1]:
            print(f"{col.Fore.RED}Error: expected linear lines (type 1), got {list(components[0])}{col.Style.RESET_ALL}")

        '''
        loop over all edge nodes

        here `component_nodes` is, for example [array([10, 11, 12, 11, 13, 12])], where 
        
            [10,   11,    11,     13]
            └ edge 1 ┘ └ edge 2 ┘
        '''
        component_nodes = components[2][0] if len(components[2]) > 0 else []

        for i in range(0, len(component_nodes), 2):
            
            # store into pair_12 = [ID_1, ID_2] the IDs of the vertices which lie at the extremities of the line in the triangle, and similarly for pair_23, pair_31
            pair_12 = tuple(sorted([component_nodes[i], component_nodes[i + 1]]))
  
            # this pushes back the elements pair_12 to edges
            edges.update([pair_12])

    elif mesh_dimension == 2:

        # check that the mesh components are  triangles 
        if list(components[0]) != [2]:
            print(f"{col.Fore.RED}Error: expected linear triangles (type 2), got {list(components[0])}{col.Style.RESET_ALL}")
            
        '''
        loop over all triangle nodes

        here `component_nodes` is, for example [array([10, 11, 12, 11, 13, 12])], where 
        
            [10,11,12, 11,13,12]
            └ tri 1 ┘ └ tri 2 ┘
        '''
        component_nodes = components[2][0] if len(components[2]) > 0 else []

        for i in range(0, len(component_nodes), 3):
            # store into pair_12 = [ID_1, ID_2] the IDs of the vertices which lie at the extremities of the line in the triangle, and similarly for pair_23, pair_31
            pair_12 = tuple(sorted([component_nodes[i], component_nodes[i + 1]]))
            pair_23 = tuple(sorted([component_nodes[i + 1], component_nodes[i + 2]]))
            pair_31 = tuple(sorted([component_nodes[i + 2], component_nodes[i]]))

            # this pushes back the elements pair_12, pair_23, pair_31 to edges
            edges.update([pair_12, pair_23, pair_31])

    elif mesh_dimension == 3:

        # check that the mesh copmonents are tetrahedra
        if list(components[0]) != [4]:
            print(f"{col.Fore.RED}Error: expected linear tetrahedra (type 4), got {list(components[0])}{col.Style.RESET_ALL}")


        '''
        loop over all tetrahedron nodes

        here `component_nodes` is, for example [array([10, 11, 12, 14, 11, 13, 12, 10])], where 
        
            [10,11,12,14, 11,13,12,10]
            └ tetra 1 ┘ └ tetra 2 ┘
        '''


        component_nodes = components[2][0] if len(components[2]) > 0 else []

        for i in range(0, len(component_nodes), 4):

            pair_12 = tuple(sorted([component_nodes[i], component_nodes[i + 1]]))
            pair_23 = tuple(sorted([component_nodes[i + 1], component_nodes[i + 2]]))
            pair_31 = tuple(sorted([component_nodes[i + 2], component_nodes[i]]))

            pair_41 = tuple(sorted([component_nodes[i + 3], component_nodes[i]]))
            pair_42 = tuple(sorted([component_nodes[i + 3], component_nodes[i + 1]]))
            pair_43 = tuple(sorted([component_nodes[i + 3], component_nodes[i + 2]]))


            # this pushes back the elements pair_12, pair_23, pair_31 to edges
            edges.update([pair_12, pair_23, pair_31, pair_41, pair_42, pair_43])


    # loop through the edges added before and write the endoints of their lines to file
    csvfile = open(outfile, "w")
    print(f"p_1,p_2", file=csvfile)
    for edge in edges:

        print(f"{edge[0]},{edge[1]}", file=csvfile)

    csvfile.close()

    # finalize gmsh
    gmsh.finalize()

'''
print the mesh triangles to csv file

Input values: 
    - 'infile': the .msh file from which the mesh will be read
    - 'outfile': the path, name and extension of the csv file where the triangles will be stored 

    the .csv file where the edges will be stored, in the format

    p_1,p_2,p_3
    id_p_1_triangle_0,id_p_2_triangle_0,id_p_3_triangle_0
    id_p_1_triangle_1,id_p_2_triangle_1,id_p_3_triangle_1
    ...
    
'''
def print_mesh_triangles_to_csv(infile, outfile):

    # initialize gmsh
    gmsh.initialize()

    # open the .msh file
    gmsh.open(infile)

    # get the list of components with dimension 2 from the mesh (triangles)
    triangles = gmsh.model.mesh.getElements(dim=2)

    # construct a map which, given the tag of a node, gives its coordinates
    # node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    # node_map = {node_tags[i]: node_coords[3 * i: 3 * (i + 1)] for i in range(len(node_tags))}

    # Store unique edges from the triangle elements
    # initialize a 'list' of unique elements, this sets the list to empty
    triplets = set()


    '''
    loop over all triangle nodes

    here `triangle_nodes` is, for example [array([10, 11, 12, 11, 13, 12])], where 
    
        [10,11,12, 11,13,12]
        └ tri 1 ┘ └ tri 2 ┘
    '''
    triangle_nodes = triangles[2][0] if len(triangles[2]) > 0 else []
    

    for i in range(0, len(triangle_nodes), 3):

        # store into triplet = [ID_1, ID_2, ID_3] the IDs of the vertices which form the triangle
        triplet = tuple(sorted([triangle_nodes[i], triangle_nodes[i + 1], triangle_nodes[i+2]]))
        
        # this pushes back the triplet to triplets
        triplets.update([triplet])
        
    
    # loop through the triplets added before and write the vertices of each triplet to file
    csvfile = open(outfile, "w")
    print(f"p_1,p_2,p_3", file=csvfile)
    for triplet in triplets:

        print(f"{triplet[0]},{triplet[1]},{triplet[2]}", file=csvfile)

    csvfile.close()
    
    # finalize gmsh
    gmsh.finalize()


'''
print the mesh tetrahedra to csv file

Input values: 
    - 'infile': the .msh file from which the mesh will be read
    - 'outfile': the path, name and extension of the csv file where the triangles will be stored 

    the .csv file where the tetrahedra will be stored, in the format

    p_1,p_2,p_3,p_4
    id_p_1_tetrahedron_0,id_p_2_tetrahedron_0,id_p_3_tetrahedron_0,id_p_4_tetrahedron_0
    id_p_1_tetrahedron_1,id_p_2_tetrahedron_1,id_p_3_tetrahedron_1,id_p_4_tetrahedron_1
    ...
    
'''
def print_mesh_tetrahedra_to_csv(infile, outfile):

    # initialize gmsh
    gmsh.initialize()

    # open the .msh file
    gmsh.open(infile)

    # get the list of components with dimension 2 from the mesh (triangles)
    tetrahedra = gmsh.model.mesh.getElements(dim=3)

    # Store unique edges from the triangle elements
    # initialize a 'list' of unique elements, this sets the list to empty
    quartets = set()

    '''
    loop over all tetrahedra nodes

    here `tetrahedra_nodes` is, for example [array([10, 11, 12, 13, 11, 13, 12, 10])], where 
    
        [10,11,12,13, 11,13,12,10]
        └ tetra 1  ┘   └ tetra 2 ┘
    '''
    tetrahedra_nodes = tetrahedra[2][0] if len(tetrahedra[2]) > 0 else []

    
    for i in range(0, len(tetrahedra_nodes), 4):

        # store into quartet = [ID_1, ID_2, ID_3, ID_4] the IDs of the vertices which form the triangle
        quartet = tuple(sorted([tetrahedra_nodes[i], tetrahedra_nodes[i + 1], tetrahedra_nodes[i + 2], tetrahedra_nodes[i + 3]]))
        
        # this pushes back the triplet to triplets
        quartets.update([quartet])
        
    
    # loop through the triplets added before and write the vertices of each triplet to file
    csvfile = open(outfile, "w")
    print(f"p_1,p_2,p_3,p_4", file=csvfile)
    for quartet in quartets:

        print(f"{quartet[0]},{quartet[1]},{quartet[2]},{quartet[3]}", file=csvfile)

    csvfile.close()

    # finalize gmsh
    gmsh.finalize()
    



'''
print the coordinates of start and end points of line 'line'
'''


def print_line_info(line, label):
    # Get the start and end points of the specific line
    start_point, end_point = get_line_extrema(line)

    print(f"\t{label}:\n\t\ttag = {line}")
    print_point_info(start_point, 'start_point')
    print_point_info(end_point, 'end_point')


# print the coordiantes of point 'point'
def print_point_info(point, label):
    r = get_point_coordinates(point)
    print(f"\t{label}:\n\t\ttag = {point},\n\t\tcoordinates =  {r}")

    return r


# print the info of all points in list 'list', which has label 'label'
def print_point_list_info(list, label):
    print(f'{label}: length = {len(list)}\ncontent:')
    for i in range(len(list)):
        print_point_info(list[i], f'point #{i}')


'''
add a line given by n-1 segments  separated by n points, between a point and a coordinate
- 'p_start' : ID of the starting point of the line
- 'r_end' :  coordinate of end point of the line
- 'n': number of points

Returns 
- 'points': a list of IDs of the points added as part of the line
- 'segments': a list of IDs of segments added as part of the line 
'''


def add_line_p_start_r_end_n(p_start, r_end, n, model):
    # print("Generating line ... ")

    points = [p_start]
    segments = []

    # coordinates of the start point
    r_start = get_point_coordinates(p_start)

    if n > 1:

        for i in range(1, n):
            dr = np.subtract(r_end, r_start)
            dr *= i / (n - 1)
            points.append(add_point(np.add(r_start, dr), model))

            segments.append((add_line_p_start_p_end(points[i - 1], points[i], model))[1])

            # print_point_info(points[-1], 'last added point')
            # print_line_info(segments[-1], 'last added segment')

        # print("... done.")

    else:
        print("Cannot add points!! ")

    return points, segments


'''
add a line given by n-1 segments  separated by n points, between two points
- 'p_start' : ID of the starting point of the line
- 'r_end' :  coordinate of end point of the line
- 'n': number of points

Returns 
- 'points': a list of IDs of the points added as part of the line
- 'segments': a list of IDs of segments added as part of the line 
'''


def add_line_p_start_p_end_n(p_start, p_end, n, model):
    # print("Generating line ... ")

    points = [p_start]
    segments = []

    # coordinates of the start point
    r_start = get_point_coordinates(p_start)
    r_end = get_point_coordinates(p_end)

    if n > 1:

        for i in range(1, n - 1):
            dr = np.subtract(r_end, r_start)
            dr *= i / (n - 1)
            points.append(add_point(np.add(r_start, dr), model))

            segments.append(add_line_p_start_p_end(points[i - 1], points[i], model)[1])

            # print_point_info(points[-1], 'last added point')
            # print_line_info(segments[-1], 'last added segment')

        # print("... done.")

        points.append(p_end)
        model.synchronize()

        segments.append(add_line_p_start_p_end(points[n - 2], p_end, model)[1])

        # print_point_info(points[-1], 'last added point')
        # print_line_info(segments[-1], 'last added segment')

    else:
        print("Cannot add points!! ")

    return points, segments


'''
add point with coordinates 'r' to model 'model' and return the result
'''


def add_point(r, model):
    point = model.add_point(r[0], r[1], r[2])
    model.synchronize()

    return point


'''
add a line between points 'p_start' and 'p_end' in model 'model' and return the line
'''


def add_line_p_start_p_end(p_start, p_end, model):
    line = model.add_line(p_start, p_end)
    model.synchronize()

    return [p_start, p_end], line


'''
add a line betweeen point 'p_start' and a new point with coordiantes r_end, which will be created, and return the line 

'''


def add_line_p_start_r_end(p_start, r_end, model):
    p_end = add_point(r_end, model)
    points_start_end, line = add_line_p_start_p_end(p_start, p_end, model)

    return points_start_end, line


'''
add a line between two points by setting the point coordinates
- 'r_start' : coordinates of the start point
- 'r_end' : coordinates of the end point
- 'model' : meshing model

return values:
- a list with the start and end point
- the line
'''


def add_line_r_start_r_end(r_start, r_end, model):
    p_start = add_point(r_start, model)
    p_end = add_point(r_end, model)
    points_start_end, line = add_line_p_start_p_end(p_start, p_end, model)

    return points_start_end, line


# get the coordinates of the vertex 'vertex', where vertex[0] is the dimension of the vertex (0) an vertex[1] the vertex tag (id)
def get_point_coordinates(point):
    return gmsh.model.getValue(0, point, [])  # 0 = vertex dimension


'''
return extermal points of line 'line'
'''


def get_line_extrema(line):
    start_point, end_point = gmsh.model.getAdjacencies(1, line)[1]  # [1] gives point tags

    return start_point, end_point


'''
return the coordinates of the center of mass of line 'line'
'''


def get_line_center_of_mass_coordinates(line):
    start_point, end_point = get_line_extrema(line)

    start_r = get_point_coordinates(start_point)
    end_r = get_point_coordinates(end_point)

    return (np.add(start_r, end_r) / 2)


'''
sort a list of vertices
- 'vertex_list': a list of vertices: vertex_list[i] = [ vertex_dimension (=0), vertex_id ]
- 'direction_id': the ID of the coordinate according to which the list will be sorted: 
    * to sort according to the x coordinate set direction_id = 0, 
    * to sort according to the y coordinate set direction_id = 1, 
    * to sort according to the z coordinate set direction_id = 2, 
- 'reverse': if True, the list will be sorted with respect to increasing order of the coordinate 'coordinate_id', and in reverse order otherwise
Return values:
- the sorted list of vertices
'''


def sort_vertex_list(vertex_list, direction_id, reverse):
    point_coordinates = []

    for vertex in vertex_list:
        coordinates = get_point_coordinates(vertex[1])
        point_coordinates.append([vertex, coordinates])

    point_coordinates.sort(key=lambda x: x[1][direction_id], reverse=reverse)
    print(f'sorted list = {point_coordinates}')

    return point_coordinates


'''
create a circle composed of four arcs
- 'c_r' : coordinates of the center of the circle
- 'r' : circle radius
- 'model' the meshing model used

return values:
- the circle lines (the four arcs)
- the circle points
'''


def add_circle_with_arcs(c_r, r, model):
    # add the center of the circle
    p_c = add_point(c_r, gmsh.model.geo)

    # add the point on the left, 'p_l', on the right 'p_r', on the top 'p_t' and on the bottom 'p_b'
    p_l = add_point(np.subtract(c_r, [r, 0, 0]), model)
    p_r = add_point(np.add(c_r, [r, 0, 0]), model)
    p_t = add_point(np.add(c_r, [0, r, 0]), model)
    p_b = add_point(np.subtract(c_r, [0, r, 0]), model)

    # add four arcs which will make the circle: add the arc from p_r to p_t , and similarly for the other arcs
    arc_rt = model.add_circle_arc(p_r, p_c, p_t)
    model.synchronize()

    arc_tl = model.add_circle_arc(p_t, p_c, p_l)
    model.synchronize()

    arc_lb = model.add_circle_arc(p_l, p_c, p_b)
    model.synchronize()

    arc_br = model.add_circle_arc(p_b, p_c, p_r)
    model.synchronize()

    circle_lines = [arc_rt, arc_tl, arc_lb, arc_br]

    # add the circle loop
    circle_loop = model.add_curve_loop(circle_lines)
    model.synchronize()

    return circle_lines, circle_loop


'''
create a circle composed of multiple segments
- 'c_r' : coordinates of the center of the circle
- 'r' : circle radius
- 'n_segments': the number of segments
- 'model' the meshing model used

return values:
- the circle points
- the circle segments
'''


def add_circle_with_lines(c_r, r, n_segments, model):
    points_circle = []
    segments_circle = []

    coord = np.add(c_r, [r, 0, 0])
    points_circle.append(add_point(coord, model))

    for i in range(1, n_segments - 1):
        coord = np.add(c_r, np.dot(cal.R_z(i / (n_segments - 1) * 2.0 * np.pi), [r, 0, 0]))
        points_circle.append(add_point(coord, model))
        segments_circle.append((add_line_p_start_p_end(points_circle[i - 1], points_circle[i], model))[1])

    segments_circle.append(add_line_p_start_p_end(points_circle[-1], points_circle[0], model)[1])

    return points_circle, segments_circle


'''
tag as physical entities the objects with a given dimension in a mesh
Input values:
- 'list_of_objects': an array containing the objects to be tagged
- 'dimension': the dimension of the objects that one wants to tag
- 'tag' : the tag which one wants to give to the objects
- 'labal' : the lable which one wants to give to the objects
'''


def tag_group(list_of_objects, dimension, tag, label):
    gmsh.model.addPhysicalGroup(dimension, list_of_objects, tag)
    gmsh.model.setPhysicalName(dimension, tag, label)


'''
Print the information on a triangle in a mesh
Input values:
- 'triangle': the triangle, an element of mesh.cells[i].data
- 'mesh': the mesh
'''


def print_mesh_triangle(triangle, mesh):
    # vertex_1 = tuple(sorted([triangle[0], triangle[1]]))
    # vertex_2 = tuple(sorted([triangle[1], triangle[2]]))
    # vertex_3 = tuple(sorted([triangle[2], triangle[0]]))
    coordinates_vertex_1 = mesh.points[triangle[0]]
    coordinates_vertex_2 = mesh.points[triangle[1]]
    coordinates_vertex_3 = mesh.points[triangle[2]]

    print(f'\tTriangle {np.sort(triangle)}')
    print(f'\t\t{coordinates_vertex_1}\n\t\t{coordinates_vertex_2}\n\t\t{coordinates_vertex_3}')


'''
Print all triangles of a mesh
Input values 
- 'mesh': the mesh, a <meshio mesh object>
'''


def print_mesh_triangles(mesh):
    print('Cell triangles: ')
    for cell_block in mesh.cells:
        if cell_block.type == "triangle":
            for triangle in cell_block.data:
                print_mesh_triangle(triangle, mesh)


'''
Print all mesh vertices
Input values: 
- 'mesh': the mesh, a <meshio mesh object>
'''


def print_mesh_vertices(mesh):
    for i, point in enumerate(mesh.points):
        print(f"Vertex ID: {i}, Coordinates: {point}")


'''
Print all element types of a mesh (such as triangles, tetrahedra, lines ...)
Input values: 
- 'mesh': the mesh, a <meshio mesh object>
'''


def print_mesh_element_types(mesh):
    print("Cell types in the mesh:")
    for cell_block in mesh.cells:
        print(f"\t{cell_block.type}")


'''
print information (element types, triangles, vertices) on a mesh
Input values: 
- 'mesh': the mesh, a <meshio mesh object>
- 'title' : a title for the printout
'''


def print_mesh_info(mesh, title):
    print(f'{title}')
    print_mesh_element_types(mesh)
    print_mesh_triangles(mesh)
    print_mesh_vertices(mesh)


'''
assign a tag to lines in a cell which satisfy a given condition
Input values:
- 'line_condition': a function of the line which tells whether the line satifies the condition to be tagged
- 'tag' : the tag which one wants to assign to the lines
- 'mesh': the mesh, a <meshio mesh object>
'''


def asssign_tag_to_lines(line_condition, tag, mesh):
    # assign to the l edge the id 'lower_edge_id'
    for j in range(len(mesh.cells)):
        # loop through  blocks of lines

        if mesh.cells[j].type == "line":
            # print(f'\tI am on line block {mesh.cells[j].data}')

            # loop through the lines in  block  mesh.cells[j].data
            for i in range(len(mesh.cells[j].data)):

                if line_condition(mesh.cells[j].data[i]):
                    # the extremal points lie on the axis x[1] = 0 -> the line mesh.cells[j].data[i] belongs to the b edge of the rectangle
                    # print(f"\t\tLine: {i} -> Point 1: {point1}, Point 2: {point2}")
                    # tag the line under consideration with ID target_id
                    mesh.cell_data['gmsh:physical'][j][i] = tag


'''
This function mirrors the points in a rectangular mesh: 
Input values: 
- 'mirror_function': the function which performs the mirroring of each point
- 'points' : Array of points to be duplicated
- 'point_data' : Data that contains dimensional tag of the points (must be duplicated as well to avoid issues during the reading of the mesh)
Return values: 
- 'new_points' : the old and the new points
- 'non_mirrored_new_points_indices' : the indices of the old points which have not been mirrored, and of the 
newly mirrored points in the new array 
(they are not just the indices of the old points traslated by some constant since the points on the x axis has not been duplicated and they were not ordered in the old list)
- 'mirrored_point_data ': array of the points which have been mirrored 

Example of usage: 
'''


def mirror_points(axis_of_symmetry_condition, mirror_function, points, point_data):
    offset = 0
    non_mirrored_plus_new_points_indices = []
    mirrored_points = []
    mirrored_point_data = []

    print('Called mirror_points. Looping through points to mirror them ...')

    for i in range(len(points)):
        # if np.isclose(points[i, 1], axis_of_symmetry_condition, rtol=cal.small_number):
        if axis_of_symmetry_condition(points[i]):
            # I ran into a point with x[1] = y_coordinate_axis_of_symmetry -> do not mirror it and append to old_plus_new_points the same index 'i' as the original point
            offset += 1
            non_mirrored_plus_new_points_indices.append(i)

            # print(f'\tNot mirroring points with label {i}')

        else:
            #  I ran into a point with x[1] != y_coordinate_axis_of_symmetry -> mirror it
            non_mirrored_plus_new_points_indices.append(i - offset + len(points))
            l = list(point_data['gmsh:dim_tags'][i, :])

            # append two points with indexes:
            # 1) the original point
            mirrored_point_data.append(l)
            # 2) the mirror of the original point
            # mirrored_points.append([points[i, 0], h - points[i, 1], points[i, 2]])
            mirrored_points.append(mirror_function(points[i]))

    print('... done.')

    mirrored_points = np.array(mirrored_points)
    old_plus_new_points = np.vstack((points, mirrored_points))

    return old_plus_new_points, non_mirrored_plus_new_points_indices, mirrored_point_data


'''
mirrors lines in a mesh according to an axis of symmetry
Input values:
- 'mesh': the mesh, a <meshio mesh object>
- 'gamma_axis_of_symmetry': the curve which defines the axis of symmetry
- 'non_mirrored_plus_new_points_indices': the indices of the old points which have not been mirrored, and of the new points, as returned from 'mirror_points'

Example of usage: 
old_plus_new_points, non_mirrored_plus_new_points_indices, mirrored_point_data = msh.mirror_points(point_on_axis_of_symmetry, mirror_function, mesh.points,
 msh.mirror_lines(mesh, gamma_axis_of_symmetry, non_mirrored_plus_new_points_indices)                                                                                                  mesh.point_data)
'''


def mirror_lines(mesh, gamma_axis_of_symmetry, non_mirrored_plus_new_points_indices):
    print('Duplicating cell lines ... ')

    for j in range(len(mesh.cells)):
        # print(f'\tj = {j}', flush=True)

        if mesh.cells[j].type == 'line':
            lines = np.copy(mesh.cells[j].data)
            filtered_lines = []

            # print(f'\t\tlines = {lines}')

            for i in range(np.shape(lines)[0]):

                # print(f'\t\t\tlines[i] = {lines[i]}')

                if (not cal.line_on_axis(lines[i], gamma_axis_of_symmetry, mesh)):
                    filtered_lines.append([non_mirrored_plus_new_points_indices[lines[i, 0]],
                                           non_mirrored_plus_new_points_indices[lines[i, 1]]])

                    # print('\t\t\t\tLine has been mirrored')

                # else:
                # print('\t\t\t\tLine has not been mirrored')

            filtered_lines = np.array(filtered_lines)

            # print(f'\t\tfiltered_lines = {filtered_lines}', flush=True)

            if filtered_lines != []:
                lines_plus_filtered_lines = np.vstack((lines, filtered_lines))
            else:
                lines_plus_filtered_lines = lines

            # print(f'\t\tlines + filetered lines = {lines_plus_filtered_lines}', flush=True)

            mesh.cells[j] = meshio.CellBlock("line", lines_plus_filtered_lines)

            N = np.shape(mesh.cells[j].data)[0]

            # print(f'\t\tN = {N}', flush=True)
            # print(f'\t\tcell_data["gmsh:physical"][{j}] = {mesh.cell_data["gmsh:physical"][j]}', flush=True)

            mesh.cell_data['gmsh:physical'][j] = np.array([mesh.cell_data['gmsh:physical'][j][0]] * N)
            mesh.cell_data['gmsh:geometrical'][j] = np.array([mesh.cell_data['gmsh:geometrical'][j][0]] * N)

    print('... done.')


'''
mirror the triangles in a cell
- 'mesh': the mesh, a <meshio mesh object>
- 'old_plus_new_points' : the set of old and new (mirrored) points, as returned from 'mirror_points'
- 'non_mirrored_plus_new_points_indices': the indices of the non-mirrored and new points, as returned from 'mirror_points'
- 'mirrored_point_data': data of the mirrored poitns, as returned from 'mirror_points'

'''


def mirror_triangles(mesh, old_plus_new_points, non_mirrored_plus_new_points_indices, mirrored_point_data):
    old_triangles = mesh.cells_dict['triangle']

    # duplicate cell blocks of type 'triangle'
    new_triangles = np.copy(old_triangles)

    # run through the old triangles
    for i in range(np.shape(new_triangles)[0]):
        # for each old triangle, run through each of its three vertices
        for j in range(3):
            '''
            assign to the new triangle the vertex tag of the old triangle, mapped towards the vertex tags of the mirrored vertices
            In this way, one reconstructs the same pattern as the old triangles, for the flipped part of the mesh
            '''
            new_triangles[i, j] = non_mirrored_plus_new_points_indices[old_triangles[i, j]]

    mesh.points = old_plus_new_points
    mesh.point_data['gmsh:dim_tags'] = np.vstack((mesh.point_data['gmsh:dim_tags'], mirrored_point_data))
    mesh.cells[-1] = meshio.CellBlock("triangle", np.vstack((old_triangles, new_triangles)))
    N = np.shape(mesh.cells[-1].data)[0]
    mesh.cell_data['gmsh:physical'][-1] = np.array([mesh.cell_data['gmsh:physical'][-1][0]] * N)
    mesh.cell_data['gmsh:geometrical'][-1] = np.array([mesh.cell_data['gmsh:geometrical'][-1][0]] * N)


'''
mirror a mesh with respect to an axis of symmetry
Input values: 
- 'mesh': the mesh, a <meshio mesh object>
- 'gamma_axis_of_symmetry': the curve which defines the axis of symmetry

Example of usage:
gamma_axis_of_symmetry = lambda t: cal.line(r_1, r_4, t)
msh.mirror_mesh(mesh, gamma_axis_of_symmetry)
'''


def mirror_mesh(mesh, gamma_axis_of_symmetry):
    # define the function which tells whether a point is on the axis of symmetry
    f_on_axis_of_symmetry = lambda point: cal.point_on_line(point, gamma_axis_of_symmetry)

    # define the function which mirrors the coordinates of a point with respect to the axis of symmetry
    f_mirror = lambda point: cal.mirror_point_line(point, gamma_axis_of_symmetry)

    # mirror  mesh points and return the relative data
    old_plus_new_points, non_mirrored_plus_new_points_indices, mirrored_point_data = mirror_points(f_on_axis_of_symmetry, f_mirror, mesh.points,
                                                                                                   mesh.point_data)
    # mirror  mesh triangles
    mirror_triangles(mesh, old_plus_new_points, non_mirrored_plus_new_points_indices, mirrored_point_data)

    # mirror mesh lines
    mirror_lines(mesh, gamma_axis_of_symmetry, non_mirrored_plus_new_points_indices)


'''
check the l <-> symmetry of a square mesh
Input values :
- 'mesh': the mesh, a <meshio mesh object>
- 'center': the center with respect to which symmetry will be assessed. This method will assess the symmetry with respect to the lines
  x[0] = center[0] (line parallel to the x[1] axis) and with respect to the line x[1] = center[1] (line parallel to the x[0] axis)

Example of usage:
    msh.check_lr_symmetry_square_mesh(mesh, c)
'''


def check_mesh_symmetry(mesh, center):
    Q = FunctionSpace(mesh, 'CG', 1)
    coordinates = Q.tabulate_dof_coordinates()

    print(f'Number of vertices = {Q.dim()}')

    average_lr = 0
    n_vertices_average_lr = 0

    average_tb = 0
    n_vertices_average_tb = 0

    for i in range(Q.dim()):

        if ((not np.isclose(coordinates[i][0], center[0]))):
            average_lr += coordinates[i][0]
            n_vertices_average_lr += 1

        if ((not np.isclose(coordinates[i][1], center[1]))):
            average_tb += coordinates[i][1]
            n_vertices_average_tb += 1

    average_lr /= n_vertices_average_lr
    average_tb /= n_vertices_average_tb

    print(f'Check l <-> r symmetry: <x - center_x> = {col.Fore.BLUE}{(average_lr - center[0]):.{io.number_of_decimals}e}{col.Fore.RESET}')
    print(f'Check t <-> b symmetry: <y - center_y> = {col.Fore.BLUE}{(average_tb - center[1]):.{io.number_of_decimals}e}{col.Fore.RESET}')


'''
Generate a mesh given by a ring slice
Input values: 
- 'r', 'R': the inner and outer radii of the circles delimiting the ring
- 'c_r', 'c_R' the centers of the rings
- 'theta': the angular width of the slice, in radians
- 'resolution': the mesh resolution
- 'output_file': the .msh file where the mesh will be stored. The mesh lines will be written in the same folder in edges.csv file

Example of usage:
    msh.generate_mesh_ring_slice(r, R, c_r, c_R, theta, resolution, mesh_slice_file)
'''


def generate_mesh_ring_slice(r, R, c_r, c_R, theta, resolution, output_file):
    output_directory = io.add_trailing_slash(os.path.dirname(output_file))

    # create the path for the csv file if it does not exist
    os.makedirs(output_directory, exist_ok=True)

    surface_id = 1
    circle_r_id = 2
    circle_R_id = 3
    line_t_id = 4
    line_b_id = 5
    ids = [1, line_b_id, circle_R_id, circle_r_id, line_t_id]

    #  mesh is generated used pygmsh and it's saved in slice_mesh_msh_file
    geometry = pygmsh.geo.Geometry()
    model = geometry.__enter__()

    print(f'r = {r}\nr = {R}\nc_r = {c_r}\nc_R = {c_R}\nresolution = {resolution}\noutput directory = {output_file}')

    # center points, used to define the arcs
    p_c_r = model.add_point((c_r[0], c_r[1], 0))
    p_c_R = model.add_point((c_R[0], c_R[1], 0))

    # extremal points of the ring slice
    r_1 = np.array([r, 0])
    r_2 = cal.R(theta).dot(r_1)
    r_4 = np.array([R, 0])
    r_3 = cal.R(theta).dot(r_4)

    p_1 = model.add_point((r_1[0], r_1[1], 0), mesh_size=resolution)
    p_2 = model.add_point((r_2[0], r_2[1], 0), mesh_size=resolution)
    p_3 = model.add_point((r_3[0], r_3[1], 0), mesh_size=resolution)
    p_4 = model.add_point((r_4[0], r_4[1], 0), mesh_size=resolution)
    model.synchronize()

    arc_12 = model.add_circle_arc(p_1, p_c_r, p_2)
    model.synchronize()

    line_23 = model.add_line(p_2, p_3)
    model.synchronize()

    arc_34 = model.add_circle_arc(p_3, p_c_r, p_4)
    model.synchronize()

    line_41 = model.add_line(p_4, p_1)
    model.synchronize()

    slice_lines = [arc_12, line_23, arc_34, line_41]
    slice_loop = model.add_curve_loop(slice_lines)
    model.synchronize()

    slice_surface = model.add_plane_surface(slice_loop)
    model.synchronize()

    model.add_physical([slice_surface], "Volume")
    model.add_physical([slice_lines[0]], "r")
    model.add_physical([slice_lines[2]], "R")
    model.add_physical([slice_lines[1]], "top")
    model.add_physical(slice_lines[3], "bottom")

    geometry.generate_mesh(dim=2)
    gmsh.write(output_file)

    print_mesh_edges_to_csv(output_file, output_directory + 'edges.csv')

    gmsh.clear()
    geometry.__exit__()


"""
Translates the coordinates of each point in the mesh by the displacement field u.
This function returns a new mesh with the translated coordinates.

Parameters:
- 'mesh': the original Mesh
- 'u': the displacement field, a Function in a VectorFunctionSpace defined over the mesh

Returns:
- a Mesh object with deformed coordinates, and same ids and mesh structure
"""


def deform_mesh(mesh, u):
    # Copy the mesh to avoid modifying the original
    deformed_mesh = Mesh(mesh)

    # Create a coordinate map for modifying vertex coordinates
    new_mesh_coordinates = deformed_mesh.coordinates()

    # Loop over all vertex coordinates and apply displacement
    for i in range(len(new_mesh_coordinates)):
        
        new_mesh_coordinate = new_mesh_coordinates[i]
        value_u = u(new_mesh_coordinates[i])  # Evaluate displacement at this point
        new_mesh_coordinates[i] = new_mesh_coordinate + value_u

    return deformed_mesh


'''
full write of mesh data to file
Input values: 
    - 'mesh_file': the .msh file where the mesh is stored
    - 'components': a list of the components of the mesh to be written, e.g., ['tetra', 'triangle', 'line', 'vertex']. 
        They must be inserted in decreasing order of dimension of the component: for example 'triangle' before 'vertex'
    - 'parameters': a dictionary of mesh parameters
    - 'output_directory': the path where the mesh info will be written
    - 'prune_z': whether the z component should be pruned (true) or not (false)

Example of usage:
    msh.full_write(mesh_file, ['triangle', 'line', 'vertex'], rpam.parameters, output_directory, True)
'''


def full_write(mesh_file, components, parameters, output_directory, prune_z):
    output_directory_slash = io.add_trailing_slash(output_directory)

    for component in components:
        write_mesh_components(mesh_file, os.path.join(output_directory_slash, component + "_mesh.xdmf"), component, prune_z)

    # print  mesh vertices to csv file
    mesh = read_mesh(output_directory_slash + components[0] + "_mesh.xdmf")


    # print the mesh vertices to csv fie
    print_mesh_vertices_to_csv(mesh_file, os.path.join(output_directory_slash, "vertices.csv"))

    # print the mesh edges to csv fie
    print_mesh_edges_to_csv(mesh_file, os.path.join(output_directory_slash, "edges.csv"))

    if mesh.topology().dim() > 1:
        
        # the mesh has dimension > 1 -> print the mesh triangles to csv
        print_mesh_triangles_to_csv(mesh_file, os.path.join(output_directory_slash, "triangles.csv"))

        if mesh.topology().dim() > 2: 

            # the mesh has dimension > 2 -> print the mesh tetrahedra to csv
            print_mesh_tetrahedra_to_csv(mesh_file, os.path.join(output_directory_slash, "tetrahedra.csv"))

    # print mesh metadata
    io.write_parameters_to_csv_file(os.path.join(output_directory_slash, "mesh_metadata.csv"), parameters)


'''
Given a parent mesh and a submesh of it, and function mf_parent which identifies facets on the parent mesh, 
this method returns the function which identifies the facet markers on the  sub_mesh, with the same ids as in the parent mesh
Input values: 
- 'parent': the parent mesh
- 'submesh': the submesh of the parent mesh
- 'mf_parent': the function which identifies facets on the parent mesh
Return values
- 'mf_submesh': the function which identifies facets on a submesh of the parent mesh

Example of usage: 
    mf = msh.read_mesh_components(lmsh.mesh, 1, rarg.args.input_directory + "/line_mesh.xdmf")
    submesh_out = SubMesh(lmsh.mesh, sf, parameters["surface_out_id"])
    mf_submesh_out = transfer_facet_tags_to_sub_mesh(lmsh.mesh, submesh_out, mf)
    
Then you can create a ds on the submesh with 
    ds_l_submesh_out = Measure("ds", domain=submesh_out, subdomain_data=mf_submesh_out, subdomain_id=parameters["line_sub_mesh_1_l_id"])
'''


def transfer_facet_tags_to_sub_mesh(parent_mesh, sub_mesh, mf_parent):
    # Create facet marker on submesh
    mf_sub = MeshFunction('size_t', sub_mesh, 1, 0)

    vertex_map = sub_mesh.data().array("parent_vertex_indices", 0)

    # run through all the facets of the sub_mesh
    for sub_mesh_facet in facets(sub_mesh):
        # extract the vertices of the facet under considerationn
        sub_mesh_facet_vertices = sub_mesh_facet.entities(0)

        # consider the relative vertices in the parent mesh
        parent_vertices = [vertex_map[v] for v in sub_mesh_facet_vertices]
        #  search for a facet in the parent mesh that shares these
        for facet in facets(parent_mesh):
            if sorted(facet.entities(0)) == sorted(parent_vertices):
                # a corresponding facet in the parent mehs has been found
                mf_sub[sub_mesh_facet.index()] = mf_parent[facet.index()]
                break

    return mf_sub


'''
map the tags of  boundary lines of a parent mesh to a boudnary mesh derived from the parent mesh
Input values: 
- 'boundary_mesh': the boundary mesh obtained from the parent mesh
- 'mf_parent_mesh' : the map which tags the lines in the parent mesh
Return values: 
- 'mf_boundary_mesh': the map which tags the lines in the boundary mesh, with the same ids which they had in the parent mesh

Example of usage: 
    submesh_out = SubMesh(parent_mesh, sf, rpam.parameters["surface_out_id"])
    boundary_mesh = BoundaryMesh(submesh_out, "exterior", order=True)
    mf_submesh_out = msh.transfer_facet_tags_to_sub_mesh(parent_mesh, submesh_out, mf)
    mf_boundary_mesh = msh.transfer_facet_tags_to_bounday_mesh(boundary_mesh, mf_submesh_out)
'''


def transfer_facet_tags_to_bounday_mesh(boundary_mesh, mf_parent_mesh):
    # entity_map(1) maps boundary mesh facets to sub_mesh facets
    boundary_to_parent_facet_map = boundary_mesh.entity_map(1)

    # construct a map function which tags all vertices (dimension = 1), with id 0
    mf_boundary_mesh = MeshFunction("size_t", boundary_mesh, 1, 0)  # facets in 1D mesh are edges

    # run on all facets of boundary_mesh
    for i, b_facet in enumerate(facets(boundary_mesh)):
        # obtain the id with whuch the facet under consideration  was tagged in boundary mesh
        submesh_facet_id = boundary_to_parent_facet_map[i]
        # impose that the function  mf_boundary_mesh evaluated on the facet under consideration must be equal to the id that the facet had in the submesh
        mf_boundary_mesh[b_facet] = mf_parent_mesh[submesh_facet_id]

    return mf_boundary_mesh


'''
Given a parent mesh and a submesh of it, and function sf_parent which identifies cells on the parent mesh, 
this method returns the function which identifies the cells on the sub_mesh, with the same ids as in the parent mesh
Input values: 
- 'sub_mesh': the sub_mesh of the parent mesh
- 'sf_parent': the function which identifies cells on the parent mesh
Return values
- 'sf_submesh': the function which identifies cells on the sub_mesh of the parent mesh

Example of usage: 
    sf = msh.read_mesh_components(lmsh.mesh, 2, rarg.args.input_directory + "/triangle_mesh.xdmf")
    submesh_out = SubMesh(lmsh.mesh, sf, parameters["surface_out_id"])
    sf_submesh_out = msh.transfer_cell_tags_to_sub_mesh(submesh_out, sf)

Then you can create a ds on the submesh with 
 

'''


def transfer_cell_tags_to_sub_mesh(sub_mesh, sf_parent):
    sf_submesh_out = MeshFunction('size_t', sub_mesh, 2)
    parent_cell_map = sub_mesh.data().array('parent_cell_indices', 2)

    # run over all cells of the sub_mesh
    for sub_cell in range(sub_mesh.num_entities(2)):
        # map the cell of the sub_mesh into the corresponding mesh of the parent cell
        parent_cell = parent_cell_map[sub_cell]
        # assign the correct id of the function sf_submesh_out calculated on the mesh of the sub_mesh under consideration, setting it to the same id it has in the parent_mesh
        sf_submesh_out[sub_cell] = sf_parent[parent_cell]

    return sf_submesh_out


'''
read a 1,2 or 3d mesh stored into an xdmf file
Input values: 
- 'input_path': the path where 'tetra_mesh.xdmf', 'triangle_mesh.xdmf', or 'line_mesh.xdmf' are located
Return values: 
- 'mesh': the mesh, or [] if the mesh could not be read
- 'sf': the mesh function for the components of the mesh with the largest dimension 
'''


def read_from_xdmf_file(mesh_path):
    mesh_path_with_slash = io.add_trailing_slash(mesh_path)

    if cmd.check_if_file_exists(mesh_path_with_slash + "tetra_mesh.xdmf"):
        mesh = read_mesh(mesh_path_with_slash + "tetra_mesh.xdmf")
        sf = read_mesh_components(mesh, mesh.topology().dim(), mesh_path_with_slash + "tetra_mesh.xdmf")
        print('3d mesh')

        result = mesh, sf

    else:
        if cmd.check_if_file_exists(mesh_path_with_slash + "triangle_mesh.xdmf"):
            mesh = read_mesh(mesh_path_with_slash + "triangle_mesh.xdmf")
            sf = read_mesh_components(mesh, mesh.topology().dim(), mesh_path_with_slash + "triangle_mesh.xdmf")

            result = mesh, sf

        else:
            if cmd.check_if_file_exists(mesh_path_with_slash + "line_mesh.xdmf"):
                mesh = read_mesh(mesh_path_with_slash + "line_mesh.xdmf")
                sf = read_mesh_components(mesh, mesh.topology().dim(), mesh_path_with_slash + "line_mesh.xdmf")

                result = mesh, sf
            else:

                print(f"{col.Fore.RED}No mesh could be loaded!{col.Style.RESET_ALL}")

                result = []

    return result


'''
read a 1d mesh stored into an h5 file
Input values: 
    - 'mesh_path': the path where 'line_mesh.h5' is located
Return values: 
    - 'mesh': the mesh, or [] if the mesh could not be read
    - 'cf': the mesh function for the components of the mesh with the largest dimension (lines)
'''


def read_from_h5_file(mesh_path):
    mesh_path_with_slash = io.add_trailing_slash(mesh_path)

    if cmd.check_if_file_exists(mesh_path_with_slash + "line_mesh.h5"):
        mesh = read_mesh(mesh_path_with_slash + "line_mesh.h5")
        cf = read_mesh_components(mesh, mesh.topology().dim(), mesh_path_with_slash + "line_mesh.h5", "cf")

        print('1d mesh')

        result = mesh, cf

    else:
        print(f"{col.Fore.RED}No mesh could be loaded!{col.Style.RESET_ALL}")
        result = []

    return result


def read_from_file(mesh_path, file_format='xdmf'):
    if file_format == 'xdmf':
        return read_from_xdmf_file(mesh_path)
    elif file_format == 'h5':
        return read_from_h5_file(mesh_path)
    else:
        print(f"{col.Fore.RED}No mesh could be loaded!{col.Style.RESET_ALL}")


'''
write a mesh to xdmf file
Input values:
- 'mesh': the mesh
- 'map': the map containing the tags of the mesh elements (triangles, lines, etc) 
- 'output_file': path + name of the xdmf file where the mesh will be written 
'''


def write_mesh(mesh, output_file, map=None):
    with XDMFFile(output_file) as xdmf:
        xdmf.write(mesh)
        xdmf.write(map)
        xdmf.close()


'''
write a mesh to a `.msh` file

Input values: 
    * Mandatory: 
        - `mesh`: the mesh
        - `filename`: path and filename with `.msh` extension where the mesh will be written
    * Optinal:
        - 'vertex_function`: a mesh function that tags mesh vertices
        - `cell_function`: a mesh function taht tags mesh cells

'''
def write_line_mesh_to_msh(mesh, filename,
                 vertex_function=None,
                 cell_function=None):

    '''
    vertex coordinates: dolfin gives (N, 1) for a 1D mesh; meshio wants (N, 3)
        coordinates = [[x_vertex_0], [x_vertex_1], .... ]
    '''
    coordinates = mesh.coordinates()                      # shape (N, 1)

    # print(f'coordinates = {coordinates}')
    '''
    create an array `points` filled with zero three-dimensional coordinates, and then fill their x component with the x- values in `vertex_coordinates`
    points = [
        [x_vertex_0, 0, 0],
        [x_vertex_1, 0, 0],
        ...
    ]
    '''
    points = np.zeros((coordinates.shape[0], 3))
    # print(f'points bef = {points}')
    points[:, 0] = coordinates[:, 0]                       # x in first column, y=z=0
    # print(f'points aft = {points}')

    '''
    connectivity: each interval cell is a pair of vertex indices
    cells = [
        [0, 1], 
        [1, 2], 
        ...
    ]
    '''
    cells = mesh.cells().astype(np.uint64)                             # shape (n_intervals, 2)

    # print(f'cells = {cells}')
    cell_blocks = [("line", cells)]

    '''
    `physical entities` stores the entities to which physical tags are assigned
    For example:
        physical_entities = [array([1, 1, 1 ..., 1], dtype=uint64), array([2, 4, 3], dtype=uint64)]
    where the first array are cells (tagged with `1`) and the second are three vertices, tagged with `2`, `4` and `3`
    '''
    physical_entities = []


    if cell_function is not None:
        # a `cell_function has been provided -> write its tags into `physical_entities`

        physical_entities.append(cell_function.array().astype(np.uint64))

    else:
        # no `cell_function` has been provided -> sett `physical_entities` to an array with length equal to the length of `cells` and all filled with 0

        physical_entities.append(np.zeros((len(cells)), dtype=np.uint64))

    '''
    vertex_tags = [tag_vertex_0, tag_vertex_1, ...]
    '''
    vertex_tags = vertex_function.array().astype(np.uint64) 

    # print(f'vertex_tags = {vertex_tags}')

    '''
    tagged_ids contains the positions of the vertices in `vertex_tags` that have a nonzero tag. For example, if `vertex_tags` = [1, 0, 0, 3], then `taggd_ids` = [0, 3]
    '''
    tagged_ids = np.nonzero(vertex_tags)[0]
    # print(f'tagged_ids = {tagged_ids}')

    if len(tagged_ids > 0):

        '''
        reshape tagged_ids in column format and append it to cell_blocks: they are thus interpreted as a cell block with a single vertex
        '''
        vertex_cells = tagged_ids.reshape(-1, 1).astype(np.uint64)
        # print(f'vertex_cells = {vertex_cells}')

        cell_blocks.append(('vertex', vertex_cells))
        '''
        append the corresponding tags of the tagged vertices to `physical_entities`
        '''
        physical_entities.append(vertex_tags[tagged_ids])

    # print("coords of tagged =", points[tagged_ids, 0])
    # print("tags of tagged   =", vertex_tags[tagged_ids])
    
    '''
    write the mesh to .msh file by using the tag lists  built above
    '''
    meshio.write_points_cells(
        filename,
        points,
        cell_blocks,
        cell_data={
            "gmsh:physical": physical_entities,
            "gmsh:geometrical": [np.zeros_like(a) for a in physical_entities]
        },
        file_format="gmsh22",
        binary=False
    )

    # m = meshio.read(filename)
    # print("data = ", m.cell_data_dict.get("gmsh:physical", "NO PHYSICAL DATA"))
    # print("vertex physical:", m.cell_data_dict["gmsh:physical"].get("vertex", "NONE"))

'''
this method generates a submesh from a parent mesh
Input values:
- 'parent_mesh_path': the path where the field triangle_mesh.xdmf and line_mesh.xdmf are stored
- 'sub_mesh_path': the path where triangle_mesh.xdmf and line_mesh.xdmf of the submesh whill be stored
- 'sub_mesh_id' : the id with which the triangles of the submesh are tagged in the parent mesh

Return values:
- 'sub_mesh', 'boundary_sub_mesh': the sub_mesh and the mesh given by the boundary of the sub_mesh
'''


def generate_sub_mesh(parent_mesh_path, sub_mesh_path, sub_mesh_id):
    parent_mesh_path_slash = io.add_trailing_slash(parent_mesh_path)
    submesh_path_slash = io.add_trailing_slash(sub_mesh_path)

    parent_mesh = read_mesh(parent_mesh_path_slash + 'triangle_mesh.xdmf')

    # create entity maps fo the parent mesh
    sf_parent_mesh = read_mesh_components(parent_mesh, parent_mesh.topology().dim(), parent_mesh_path_slash + "triangle_mesh.xdmf")
    mf_parent_mesh = read_mesh_components(parent_mesh, parent_mesh.topology().dim() - 1, parent_mesh_path_slash + "line_mesh.xdmf")

    # extract the outer sub_mesh from the parent mesh, by picking only the triangles with submesh_id
    sub_mesh = SubMesh(parent_mesh, sf_parent_mesh, sub_mesh_id)
    # create the boundary mesh of sub_mesh
    sub_mesh_boundary = BoundaryMesh(sub_mesh, "exterior", order=True)

    # print(f'type of sub_mesh: {type(sub_mesh)}')

    # create entity maps of sub_mesh for triangles and lines
    sf_sub_mesh = transfer_cell_tags_to_sub_mesh(sub_mesh, sf_parent_mesh)
    mf_sub_mesh = transfer_facet_tags_to_sub_mesh(parent_mesh, sub_mesh, mf_parent_mesh)

    # create entity map for boundary mesh for lines
    mf_boundary_sub_mesh = transfer_facet_tags_to_bounday_mesh(sub_mesh_boundary, mf_sub_mesh)

    # write the triangles for sub_mesh to xdmf file
    write_mesh(sub_mesh, submesh_path_slash + "triangle_mesh.xdmf", sf_sub_mesh)
    # write the lines of the boundary mesh to xdmf file
    write_mesh(sub_mesh_boundary, submesh_path_slash + "line_mesh.xdmf", mf_boundary_sub_mesh)
    
    # print  submesh vertices to csv file
    io.print_mesh_vertices_to_csv(sub_mesh, submesh_path_slash + "vertices.csv")
    
    if sub_mesh.topology().dim() == 2:
        #  sub_mesh is two-dimensional -> print its coordinates 
        # print  submesh triangles to csv file
        io.print_mesh_triangles_to_csv(sub_mesh, submesh_path_slash + "triangles.csv")
    
    # print sub mesh metadata
    # io.write_parameters_to_csv_file(submesh_path_slash + "mesh_metadata.csv", submesh_parameters)

    return sub_mesh, sub_mesh_boundary

'''
generate a line mesh whose vertex coordinates are provided as input
Input values: 
    - 'coordinates': the coordinates of the vertices [v_0_x, v_1_x, ...]
Return values: 
    - 'mesh': the mesh
'''
def IntervalMeshCoordinates(coordinates):

    n_vertices = len(coordinates)
    n_cells = n_vertices - 1
  
    mesh = Mesh()
    editor = MeshEditor()
  
    sorted_coordinates = sorted(coordinates)
        
    editor.open(mesh, 'interval', 1, 1)  # cell type, topological dim, geometric dim

    editor.init_vertices(n_vertices)
    editor.init_cells(n_cells)

    for i, x in enumerate(sorted_coordinates):
        editor.add_vertex(i, np.array([x]))

    for i in range(n_cells):
        editor.add_cell(i, np.array([i, i + 1], dtype=np.uintp))

    editor.close()

    return mesh

'''
generate a one-dimensional mesh as an IntervalMesh given its geometric parameters and tags
Input values: 
    * Mandatory:
        - 'x_l', 'x_r': the left and right x coordinate of the extremal points of the line mesh
        - 'n_intervals': the number of intervals into which the line mesh is divided
        - 'line_id': the id of the line mesh: all lien intervals will be tagged with this id
        - 'vertex_l_id', 'vertex_r_id': the id of the extermal left and right vertices, respectively
        - 'x_m_id' [optional]: the coordinate of the middle vertex in the mesh: this coordinate must match with one of the coordinates of the mesh vertices
        - 'vertex_m_id': the id of the middle vertex in the mesh
    * Optional:
        - 'metadata': the mesh metadata to write in the output directory
        - 'coordinates': a set of coordinates [x_0, x_1, ...]. If provided, the line mesh will have vertices sitting at these coordinates only
        - 'output_directory': the path where the mesh will be written. In that path this method will write the mesh component, vertices and, if metadata != None, the mesh metadata

Return values: 
    - 'mesh': the one-dimensional mesh
    - 'cell_function_temp': the mesh funciton tagging cells (line intervals) in the mesh
    - 'vertex_function_temp': the mesh function tagging vertices in the mesh


Example of usage: 
          mesh_1d, cf_mesh_1d, vf_mesh_1d = msh.genereate_line_mesh(0, parameters['L'], len(x_coordinates) - 1,  parameters[f'sub_mesh_{p}_id'], parameters['vertex_sub_mesh_1_l_id'], parameters['vertex_sub_mesh_1_r_id'])
'''


def genereate_line_mesh(x_l, x_r, n_intervals, line_id, vertex_l_id, vertex_r_id, x_m=None, vertex_m_id=None, output_directory=None, metadata=None, coordinates=None):
    
    if coordinates == None:
        # this method has been called with out coordinates, only with n_intervals -> generate a line mesh with uniform spacing 

       mesh = IntervalMesh(n_intervals, x_l, x_r)

    else:
        # this method has been called with 'coordinates' != Null -> build a mesh with those specific coordinates

        mesh = IntervalMeshCoordinates(coordinates)


    # create a function for the lines
    cell_function = MeshFunction("size_t", mesh, mesh.topology().dim())
    cell_function.set_all(line_id)  # Tag entire line as region parameters['line_id']

    # creat a function for the vertices
    vertex_function = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    
    if (x_m is not None) and (vertex_m_id is not None):
        # I am generating a mesh with a middle vertex -> create the boolean variable vertex_m_exists to check whether x_m matches one of the coordinates of the mesh vertices
        vertex_m_exists = False
        
    for vertex in vertices(mesh):
        x = vertex.point().x()  # Get x-coordinate

        if math.isclose(x, x_l):
            vertex_function[vertex] = vertex_l_id

        if math.isclose(x, x_r):
            vertex_function[vertex] = vertex_r_id
            
        # if there is a middle vertex, tag id with vertex_m_id
        if (x_m is not None) and (vertex_m_id is not None):
            # I am generating a mesh with a middle vertex -> check if the mesh vertex coordinate under consideration matches x_m
             if math.isclose(x, x_m):
                vertex_function[vertex] = vertex_m_id
                vertex_m_exists = True

    if (x_m is not None) and (vertex_m_id is not None):
        # I am generating a mesh with a middle vertex -> if no vertex coordinate matches x_m, print an error message
        if vertex_m_exists is not True:
            print(f"{col.Fore.RED}{'Error: middle vertex is not one of the mesh vertices!'}{col.Style.RESET_ALL}")

    if output_directory is not None:

        '''
        write the mesh lines and vertices to .h5 files: 
        one needs to write them to .h5 file rather than to .xdmf file because only .h5 file can be properly read later on
        '''

        mesh_file = os.path.join(output_directory, 'mesh.msh')

        write_mesh_components_h5(mesh, output_directory + "line_mesh.h5", cell_function, "cf")
        write_mesh_components_h5(mesh, output_directory + "vertex_mesh.h5", vertex_function, "vf")

        # write the mesh to .msh file
        write_line_mesh_to_msh(mesh, mesh_file,
                                vertex_function=vertex_function,
                                cell_function=cell_function
                            )

        # print the mesh vertices to csv fie
        print_mesh_vertices_to_csv(mesh_file, os.path.join(output_directory, "vertices.csv"))
        
        # print the mesh edges to csv fie
        print_mesh_edges_to_csv(mesh_file, os.path.join(output_directory, "edges.csv"))
        

        # print mesh metadata
        if metadata is not None:
            io.write_parameters_to_csv_file(output_directory + "mesh_metadata.csv", metadata)

    return mesh, cell_function, vertex_function


'''
generate a mesh given by a square with a polygon hole
Input values: 
    * Mandatory:
        - 'polygon coordinates': a list of coordinates [[p0_x, p0_y], [p1_x, p1_y], ...] of the points defining the polygon
        - 'mesh_parameters_directory': the path of the file 'mesh_parameters.csv' where the mesh parameters are located
        - 'output_directory': the path where the mesh will be stored 
    * Optional: 
        - 'additional_metadata': some additional data that will be written appended to mesh_metadata.csv. It is None by default. 
'''

def generate_square_polygon_mesh(polygon_coordinates, mesh_parameters_directory, output_directory,
                                additional_metadata=None):
    
    # remove the output directory it it already exists, and create it from scratch
    shutil.rmtree(output_directory, ignore_errors=True)
    os.makedirs(output_directory)

    geometry = pygmsh.occ.Geometry()
    model = geometry.__enter__()

    # reset gmsh state from any previous call, AFTER pygmsh has initialized it
    gmsh.clear()
    gmsh.model.add("model")  # need a model after clear()

    parameters_file_path = os.path.join(mesh_parameters_directory, 'mesh_parameters.csv')
    parameters = io.read_parameters_from_csv_file(parameters_file_path)

    mesh_file = os.path.join(output_directory,  "mesh.msh")
  
    # write into metadata the file format wich which the mesh will be written
    metadata = parameters.copy()
    metadata['file_format'] = 'xdmf'

    if additional_metadata is not None:
        metadata.update(additional_metadata)

    # generate the mesh

    # add square
    square_points = [gmsh.model.geo.addPoint(0, 0, 0),
                    gmsh.model.geo.addPoint(parameters["L"], 0, 0),
                    gmsh.model.geo.addPoint(parameters["L"], parameters["h"], 0),
                    gmsh.model.geo.addPoint(0, parameters["h"], 0)]

    square_lines = [gmsh.model.geo.addLine(square_points[0], square_points[1]),
                    gmsh.model.geo.addLine(square_points[1], square_points[2]),
                    gmsh.model.geo.addLine(square_points[2], square_points[3]),
                    gmsh.model.geo.addLine(square_points[3], square_points[0]),
                    ]

    square_loop = gmsh.model.geo.addCurveLoop(square_lines)


    # add polygon
    polygon_points = [gmsh.model.geo.addPoint(polygon_coordinates[0][0], polygon_coordinates[0][1], 0)]
    gmsh.model.geo.synchronize()

    polygon_lines = []

    for i in range(1, len(polygon_coordinates)):

        polygon_points.append(gmsh.model.geo.addPoint(polygon_coordinates[i][0], polygon_coordinates[i][1], 0))
        gmsh.model.geo.synchronize()

        polygon_lines.append(gmsh.model.geo.addLine(polygon_points[i-1], polygon_points[i]))
        gmsh.model.geo.synchronize()

    polygon_lines.append(gmsh.model.geo.addLine(polygon_points[-1], polygon_points[0]))
    gmsh.model.geo.synchronize()

    polygon_loop = gmsh.model.geo.addCurveLoop(polygon_lines)
    gmsh.model.geo.synchronize()

    gmsh.model.geo.addPlaneSurface([square_loop, polygon_loop])
    gmsh.model.geo.synchronize()



    # tag physical objects

    # tag 1-dimensional objects
    lines = gmsh.model.getEntities(dim=1)

    # square lines
    tag_physical_object(lines[0], parameters['line_b_id'], gmsh.model, 'line_b')
    tag_physical_object(lines[1], parameters['line_r_id'], gmsh.model, 'line_r')
    tag_physical_object(lines[2], parameters['line_t_id'], gmsh.model, 'line_t')
    tag_physical_object(lines[3], parameters['line_l_id'], gmsh.model, 'line_l')

    # polygon lines
    tag_physical_object([lines[i] for i in range(4, len(lines))], parameters['polygon_id'], gmsh.model, 'polygon_line')


    # tag 2-dimensional objects
    surfaces = gmsh.model.getEntities(dim=2)

    tag_physical_object(surfaces[0], parameters['surface_id'], gmsh.model, 'surface')


    # set the mesh resolution
    distance = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance, "FacesList", [polygon_loop])

    threshold = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold, "IField", distance)
    gmsh.model.mesh.field.setNumber(threshold, "LcMin", parameters["resolution"])
    gmsh.model.mesh.field.setNumber(threshold, "LcMax", parameters["resolution"])
    gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0)
    gmsh.model.mesh.field.setNumber(threshold, "DistMax", max(parameters["L"], parameters["h"]))

    minimum = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", [threshold])
    gmsh.model.mesh.field.setAsBackgroundMesh(minimum)
    gmsh.model.geo.synchronize()


    geometry.generate_mesh(dim=2)
    gmsh.write(mesh_file)

    full_write(mesh_file, ['triangle', 'line'], metadata, output_directory, True)

    # print the boundary points of the boundary given by the polygon
    sorted_boundary_points(
        read_mesh(os.path.join(output_directory, 'triangle_mesh.xdmf')), 
        output_directory, 
        [parameters['polygon_id']],
        os.path.join(output_directory, 'boundary_points_id_' + str(parameters['polygon_id']) + '.csv'))


    model.__exit__()


'''
generate a mesh given by a square with a shape inside, where the shape is meshed inside. The shape is laid flat to obtain a 1d mesh, which is also generated. 
Input values: 
    * Mandatory:
        - 'shape coordinates': a list of coordinates [[p0_x, p0_y], [p1_x, p1_y], ...] of the points defining the shape
            Note: if the meshing algorithm inserts additional vertices in between 'shape_coordinates', this method will insert these vertices into 'shape_coordinates' and call again itself to generate a mesh with these vertices
        - 'mesh_parameters_directory': the path of the file 'mesh_parameters.csv' where the mesh parameters are located
        - 'output_directory': the path where the mesh will be stored 
    * Optional: 
        - 'epsilon': the tolerance with which distances are evaluated
'''
def generate_square_shape_line_mesh(shape_coordinates, mesh_parameters_directory, output_directory,
                                    epsilon = const.epsilon):
    

    # remove the output directory it it already exists, and create it from scratch
    shutil.rmtree(output_directory, ignore_errors=True)
    os.makedirs(output_directory)

    geometry = pygmsh.occ.Geometry()
    model = geometry.__enter__()

    # reset gmsh state from any previous call, AFTER pygmsh has initialized it
    gmsh.clear()
    gmsh.model.add("model")  # need a model after clear()

    '''
    # This allows for setting space-dependent mesh resolution 
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    '''


    parameters_file_path = os.path.join(mesh_parameters_directory, 'mesh_parameters.csv')
    parameters = io.read_parameters_from_csv_file(parameters_file_path)

    for coordinate in shape_coordinates:
        if cal.point_in_box(coordinate, [[0, parameters['L']], [0, parameters['h']]]) == False:

            print(f"{col.Fore.RED}{'Error: the shape is not included into the square!'}{col.Style.RESET_ALL}")
            sys.exit()


    # mesh A will be stored in output_directory_square_mesh
    output_directory_mesh_0 = io.add_trailing_slash(os.path.join(output_directory, 'mesh_0'))
    os.mkdir(output_directory_mesh_0)
    # mesh B will be stored in output_directory_line_mesh
    output_directory_mesh_1 = io.add_trailing_slash(os.path.join(output_directory, 'mesh_1'))
    os.mkdir(output_directory_mesh_1)

    mesh_0_file = os.path.join(output_directory_mesh_0, "mesh.msh")

    # total length of the shape boundary
    shape_length = cal.polygon_length(shape_coordinates)


    #write metadata for ensemble mesh
    mesh_metadata = parameters.copy()

    # remove spurious entities in mesh_metadata
    if parameters['shape_format'] == 'parametric':

        if 'shape_coordinates' in parameters:
            del mesh_metadata['shape_coordinates']
            
    elif parameters['shape_format'] == 'coordinates':

        mesh_metadata['shape_coordinates'] = shape_coordinates

        if 'shape_parametric_form' in parameters:
            del mesh_metadata['shape_parametric_form']
            
        if 'N' in parameters:
            del mesh_metadata['N']

    # comphte the center of mass of the shape with respect to `shape_coordinates`and write it into mesh_metadata
    c = np.mean(shape_coordinates, axis=0).tolist()

    mesh_metadata['c'] = c


    # write metadata for mesh 0
    mesh_0_metadata = {}
    mesh_0_metadata['L'] = parameters['L']
    mesh_0_metadata['h'] = parameters['h']
    mesh_0_metadata['resolution'] = parameters['resolution']
    mesh_0_metadata['n_sub_meshes'] = parameters['n_sub_meshes_0']
    mesh_0_metadata['shape_format'] = parameters['shape_format']
    mesh_0_metadata['c'] = c

    # if the shape derives from a parametric form, write N and the parametric function
    if parameters['shape_format'] == 'parametric':
        mesh_0_metadata['shape_parametric_form'] = parameters['shape_parametric_form']
        mesh_0_metadata['N'] = parameters['N']
        
    
    # if the shape comes both from a parametric form or from raw coordinates, write the raw coordinates of the shape
    mesh_0_metadata['shape_coordinates'] = shape_coordinates

    mesh_0_metadata['sub_mesh_0_dim'] = parameters['sub_mesh_0_0_dim']
    mesh_0_metadata['sub_mesh_1_dim'] = parameters['sub_mesh_0_1_dim']

    mesh_0_metadata['sub_mesh_0_id'] = parameters['sub_mesh_0_0_id']
    mesh_0_metadata['sub_mesh_1_id'] = parameters['sub_mesh_0_1_id']

    mesh_0_metadata['line_l_id'] = parameters['line_l_id']
    mesh_0_metadata['line_r_id'] = parameters['line_r_id']
    mesh_0_metadata['line_t_id'] = parameters['line_t_id']
    mesh_0_metadata['line_b_id'] = parameters['line_b_id']
    mesh_0_metadata['shape_id'] = parameters['shape_id']

    mesh_0_metadata['file_format'] = 'xdmf'


    # write metadata for mesh 1
    mesh_1_metadata = {}

    mesh_1_metadata['L'] = shape_length
    mesh_1_metadata['x_l'] = 0
    mesh_1_metadata['x_r'] = mesh_1_metadata['L']

    mesh_1_metadata['vertex_l_id'] = parameters['vertex_l_id']
    mesh_1_metadata['vertex_r_id'] = parameters['vertex_r_id']
    mesh_1_metadata['line_id'] = parameters['shape_id']

    mesh_1_metadata['file_format'] = 'h5'




    # A) generate mesh A (square with circle)

    #1. add  square


    square_p_bl = gmsh.model.geo.addPoint(0, 0, 0)
    square_p_br = gmsh.model.geo.addPoint(parameters["L"], 0, 0)
    square_p_tr = gmsh.model.geo.addPoint(parameters["L"], parameters["h"], 0)
    square_p_tl = gmsh.model.geo.addPoint(0, parameters["h"], 0)
    gmsh.model.geo.synchronize()

    square_line_b = gmsh.model.geo.addLine(square_p_bl, square_p_br)
    square_line_r = gmsh.model.geo.addLine(square_p_br, square_p_tr)
    square_line_t = gmsh.model.geo.addLine(square_p_tr, square_p_tl)
    square_line_l = gmsh.model.geo.addLine(square_p_tl, square_p_bl)
    gmsh.model.geo.synchronize()

    square_loop = gmsh.model.geo.addCurveLoop([square_line_b, square_line_r, square_line_t, square_line_l])
    gmsh.model.geo.synchronize()


    #2. add shape


    shape_points = [gmsh.model.geo.addPoint(shape_coordinates[0][0], shape_coordinates[0][1], 0)]
    gmsh.model.geo.synchronize()

    shape_lines = []



    for i in range(1, len(shape_coordinates)):

        shape_points.append(gmsh.model.geo.addPoint(shape_coordinates[i][0], shape_coordinates[i][1], 0))
        gmsh.model.geo.synchronize()

        shape_lines.append(gmsh.model.geo.addLine(shape_points[-2], shape_points[-1]))
        gmsh.model.geo.synchronize()


    shape_lines.append(gmsh.model.geo.addLine(shape_points[-1], shape_points[0]))
    gmsh.model.geo.synchronize()



    shape_loop = gmsh.model.geo.addCurveLoop(shape_lines)
    gmsh.model.geo.synchronize()

    square_minus_shape_surface = gmsh.model.geo.addPlaneSurface([square_loop, shape_loop])
    gmsh.model.geo.synchronize()

    gmsh.model.mesh.embed(1, shape_lines, 2, square_minus_shape_surface)
    gmsh.model.geo.synchronize()

    shape_surface = gmsh.model.geo.addPlaneSurface([shape_loop])
    gmsh.model.geo.synchronize()



    #3. add 1-dimensional objects
    lines = gmsh.model.getEntities(dim=1)

    # add square lines
    tag_physical_object(lines[0], parameters['line_b_id'], gmsh.model, 'line_b')
    tag_physical_object(lines[1], parameters['line_r_id'], gmsh.model, 'line_r')
    tag_physical_object(lines[2], parameters['line_t_id'], gmsh.model, 'line_t')
    tag_physical_object(lines[3], parameters['line_l_id'], gmsh.model, 'line_l')

    #add shape lines
    tag_physical_object([lines[i] for i in range(4, 4 + len(shape_coordinates))], parameters['shape_id'], gmsh.model, 'shape_loop')



    #4. add 2-dimensional objects
    surfaces = gmsh.model.getEntities(dim=2)

    tag_physical_object(surfaces[0], parameters['sub_mesh_0_1_id'], gmsh.model, 'square_minus_shape_surface')
    tag_physical_object(surfaces[1], parameters['sub_mesh_0_0_id'], gmsh.model, 'shape_surface')


    #5. set the resolution
    # se resolution equal to parameters["resolution"] at a distance 0 from surface_in, and  at distance max(parameters["L"],parameters["h"]) from sub_mesh_0_1_id
    distance = gmsh.model.mesh.field.add("Distance")

    gmsh.model.mesh.field.setNumbers(distance, "FacesList", [shape_loop])

    threshold = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold, "IField", distance)
    gmsh.model.mesh.field.setNumber(threshold, "LcMin", parameters["resolution"])
    gmsh.model.mesh.field.setNumber(threshold, "LcMax", parameters["resolution"])
    gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0)
    gmsh.model.mesh.field.setNumber(threshold, "DistMax", max(parameters["L"], parameters["h"]))


    minimum = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", [threshold])
    gmsh.model.mesh.field.setAsBackgroundMesh(minimum)

    gmsh.model.geo.synchronize()

    geometry.generate_mesh(dim=2)
    gmsh.write(mesh_0_file)

    full_write(mesh_0_file, ['triangle', 'line'], mesh_0_metadata, output_directory_mesh_0, True)

    generate_sub_mesh(output_directory_mesh_0, os.path.join(output_directory_mesh_0, 'sub_meshes', 'sub_mesh_0'), parameters["sub_mesh_0_0_id"])
    generate_sub_mesh(output_directory_mesh_0, os.path.join(output_directory_mesh_0, 'sub_meshes', 'sub_mesh_1'), parameters["sub_mesh_0_1_id"])



    #6. check that the number of mesh vertices on the circle matches N and if it does not, insert these poitns into shape_coordinates and call again generate_square_shape_line_mesh. 
    mesh_0 = read_mesh(os.path.join(output_directory_mesh_0, 'triangle_mesh.xdmf'))
    mf_mesh_0 = read_mesh_components(mesh_0, mesh_0.topology().dim() - 1, os.path.join(output_directory_mesh_0, 'line_mesh.xdmf'))

    # collect unique vertex indices and coordinates touched by facets tagged with shape_id
    shape_vertex_ids = []
    shape_vertex_coordinates = []

    for facet in facets(mesh_0):
        #run through all facets of mesh_0 

        if mf_mesh_0[facet] == parameters['shape_id']:
            # the facet under consideration belongs to the shape

            for v in vertices(facet):
                # run through the vertices of the facet under consideration, and ad them to shape_vertex_ids

                if v.index() not in shape_vertex_ids:

                    shape_vertex_ids.append(v.index())
                    shape_vertex_coordinates.append(v.point().array().tolist())

    n_vertices_on_shape = len(shape_vertex_ids)
    n_vertices_on_line = len(shape_coordinates) 

    if n_vertices_on_shape != n_vertices_on_line:
        # the meshing algorithm has added additional vertices on the shape, while I want the number of vertices on the shape to match N, and thus the number of vertices in the line mesh -> print an error message

        print(f"{col.Fore.YELLOW}{'Warning: The number of vertices on shape does not match the number of vertices of the 1d mesh. Recalculating shape_coordinates ...'}{col.Style.RESET_ALL}")
        print(f'\tNumber of vertices on shape = {n_vertices_on_shape}\n\tNumber of vertices on line = {n_vertices_on_line}')

        for i in range(len(shape_vertex_coordinates)):
            # run through the mesh vertices which belong to a facet on the shape

            for j in range(len(shape_coordinates)):
                # find two subsequent vertices in shape_coordinates, such that shape_vertex_coordinates[i] lies on the line between them

                p = shape_vertex_coordinates[i][:2]
                p_start = shape_coordinates[j]
                p_end = shape_coordinates[(j+1) % len(shape_coordinates)]

                lies_in_between = geo_u.between_points(p, p_start, p_end) and \
                (np.linalg.norm(np.subtract(p, p_start)) > epsilon) and \
                (np.linalg.norm(np.subtract(p, p_end)) > epsilon)

                if lies_in_between:
                    # shape_vertex_coordinates[i] lies between two subsequent vertices in shape_coordinates, and it is not one of the vertices in shape_coordinates -> insert it into shape_coordinates and break the inner loop

                    shape_coordinates.insert(j+1, p)

                    break


        print(f"{col.Fore.YELLOW}{'... done.'}{col.Style.RESET_ALL}")

     
        model.__exit__()


        # now shape_coordinates includes the additional vertices introduced by the meshing algorithm -> call again generate_square_shape_line_mesh with this new shape_coordinates -> this will generate a 2d mesh and a line mesh, in which the number of vertices on the 2d mesh boundary shape coincides with the number of vertices on the line mesh

        generate_square_shape_line_mesh(shape_coordinates, mesh_parameters_directory, output_directory, epsilon)

    else:
        # the meshing algorithm introduced no additional vertices in between the vertices of shape_coordinates -> proceed

        _, cumulative_arc_length = shape_tool(mesh_0, mf_mesh_0, mesh_0_metadata['shape_coordinates'], parameters['shape_id'])
        mesh_1_metadata['coordinates'] = cumulative_arc_length

        # print the boundary points of the boundary given by the shape
        sorted_boundary_points(
            read_mesh(os.path.join(output_directory_mesh_0, 'triangle_mesh.xdmf')), 
            output_directory_mesh_0, 
            [parameters['shape_id']],
            os.path.join(output_directory_mesh_0, 'boundary_points_id_' + str(parameters['shape_id']) + '.csv'))


        # B) mesh B (line)


        # generate the line mesh corresponding to the shape

        genereate_line_mesh(0, shape_length, None,
                                parameters['shape_id'], parameters['vertex_l_id'], parameters['vertex_r_id'],
                                x_m=None,
                                vertex_m_id=None,
                                output_directory=output_directory_mesh_1, 
                                metadata=mesh_1_metadata,
                                coordinates=cumulative_arc_length)


        #print overall mesh metadata
        io.write_parameters_to_csv_file(os.path.join(output_directory, 'mesh_metadata.csv'), mesh_metadata)

        model.__exit__()


'''
return the geometrical shape of an element for a mesh with different dimensions
Input values: 
- 'mesh': the mesh
Return values: 
- the geometry: 'tetrahedron' for a 3d mesh, 'triangle' for a 2d mesh, 'interval' for a 1d mesh

Example of usage:
    P_u = FiniteElement('P', msh.element_geometry(lmsh.mesh), rpam.parameters['function_space_degree'])
'''


def element_geometry(mesh):
    d = mesh.topology().dim()

    if d == 3:
        return tetrahedron
    elif d == 2:
        return triangle
    elif d == 1:
        return interval

'''
read the sub-meshes of a mesh. This only works if the parent mesh is two-dimensional. 
Input values: 
    - 'mesh': the mesh of which the sub-meshes will be read
    - 'sf': the MeshFunctionSizet for the geometrical components with the largest dimension in 'mesh'. For example, if 'mesh' is 3d, this will be a function for tetrahedra, if 'mesh' is 2d this will be a function for triangles, etc. 
    - 'mesh_metadata': a dictionary containing the mesh metadata for 'mesh'
    - 'input_directory': the directory where 'mesh' is stored
Return values:
    - 'sub_meshes': a list containing the sub_meshes
    - 'sf_sub_meshes':  a list containing a function to tag the sub_mesh cells. sf_sub_meshes[i] contains a function to tag cells of the i-th sub_mesh if sub_mesh[i] is one-dimensional, and it is None otherwise
    - 'mf_sub_meshes': a list containing a function to tag the sub_mesh vertices. mf_sub_meshes[i] contains a function to tag vertices of the i-th sub_mesh  of the i-th sub_mesh if sub_mesh[i] is one-dimensional, and it is None otherwise
'''
def read_sub_meshes(mesh, sf, mesh_medatada, input_directory):

    if "n_sub_meshes" in mesh_medatada:
        # found sub-meshes

        print(f'Found sub_meshes')

        # read the functions that tag elements of the parent mesh
        sf_mesh = read_mesh_components(mesh, mesh.topology().dim(), os.path.join(input_directory, "triangle_mesh.xdmf"))
        mf_mesh = read_mesh_components(mesh, mesh.topology().dim() - 1, os.path.join(input_directory, "line_mesh.xdmf"))


        # mesh_parameters contain the field n_sub_meshes -> generate sub_meshes

        # the list of sub_meshes
        sub_meshes = []

        # the list of functions to tag objects in sub_meshes
        sf_sub_meshes = []
        mf_sub_meshes = []

        if mesh_medatada["n_sub_meshes"] > 1:
            #  'mesh' contains multiple sub_meshes: run through them and generate each sub_mesh from the parent mesh
            
            print('Generating sub_meshes ... ')
            
            for p in range(mesh_medatada["n_sub_meshes"]):

                if mesh_medatada[f'sub_mesh_{p}_dim'] > 1:
                    # the sub_mesh under consideration has dimension > 1: generate it in the ordinary way  with 'SubMesh'

                    sub_meshes.append(SubMesh(mesh, sf, mesh_medatada[f'sub_mesh_{p}_id']))

                    # the functions that tag cells and vertices on the sub-mesh are obtained by transferring the respective functiosn on the parent mesh 
                    sf_sub_meshes.append(transfer_cell_tags_to_sub_mesh(sub_meshes[p], sf_mesh))
                    mf_sub_meshes.append(transfer_facet_tags_to_sub_mesh(mesh, sub_meshes[p], mf_mesh))

                elif mesh_medatada[f'sub_mesh_{p}_dim'] == 1:
                    '''
                    the sub_mesh under consideration has dimension 1 -> it is a line: if I generated it with 'sub_meshes.append(SubMesh(mesh, sf, parameters[f'sub_mesh_{p}_id']))' 
                    I would obtain a one-dimensional mesh embedded in two-dimensional space, thus in fact a two-dimensional mesh, which is not what I want : I want a truly one-dimensional mesh. 
                    -> I create an IntervalMesh and assign to it the coordinates of the submesh, and append to sub_meshes the IntervalMesh
                    '''

                    # create the one-dimensional submesh from the facet function 'mf_mesh' and the id which identifies the sub_mesh under consideration: first extract the coordinates of the points in the one-dimensional submesh and store them into x_coordinates
                    x_coordinates = []
                    for facet in facets(mesh):
                        if mf_mesh[facet] == mesh_medatada[f'sub_mesh_{p}_id']:
                            for vertex in vertices(facet):
                                x_coordinates.append(vertex.point().x())

                    # then remove duplicates from x_coordinates and sort it 
                    x_coordinates = sorted(list(set(x_coordinates)))  

                    # generate the one-dimensional submesh and return its cell mesh function and vertex mesh function
                    sub_mesh_1d, cf_sub_mesh_1d, vf_sub_mesh_1d = genereate_line_mesh(0, mesh_medatada['L'], None,
                                                                                        mesh_medatada[f'sub_mesh_{p}_id'], mesh_medatada[f'vertex_sub_mesh_{p}_l_id'], mesh_medatada[f'vertex_sub_mesh_{p}_r_id'],
                                                                                        coordinates=x_coordinates)
                    
                    sub_meshes.append(sub_mesh_1d)
                    sf_sub_meshes.append(cf_sub_mesh_1d)
                    mf_sub_meshes.append(vf_sub_mesh_1d)

                print(f'Sub_mesh {p} has dimension {sub_meshes[p].topology().dim()}')

            print('... done.')

        else:
            # mesh does not contain multiple sub_meshes -> return None for all fields
    
            sub_meshes = None
            sf_sub_meshes = None 
            mf_sub_meshes = None

    else:
        # did not find sub_meshes -> return None for all fields

        print(f'Did not find sub_meshes')

        sub_meshes = None
        sf_sub_meshes = None 
        mf_sub_meshes = None


    return sub_meshes, sf_sub_meshes, mf_sub_meshes


'''
Given 2d mesh given by a recangle with a meshed shape in it, and a line mesh obtained by lying the shape boundary on a line, this method transfers a field (scalar, vector or tensor) defined on the 2d mesh, on the line mesh. 

Input values: 
    * Mandatory:
        - 'f_2d': the field on the 2d mesh
        - 'f_1d': the field on the 2d mesh
        - 'mesh_2d': the 2d mesh is stored
        - 'mf_mesh_2d': a function on 'mesh_2d' that tags its facets
        - 'shape_coordinates' : [[p_0_x, p_0_y], [p_1_x, p_1_y], ... ] the coordinates of the vertices of the shape in 'mesh_2d'
        - 'shape_id': the ID with which the shape is tagged in the 2d mesh 
'''

def transfer_2d_to_1d(f_2d, f_1d, mesh_2d, mf_mesh_2d, shape_coordinates, shape_id):

    # 1. initialize 
    # mesh_2d = read_mesh(os.path.join(mesh_2d_path, 'triangle_mesh.xdmf'))
    # mf_mesh_2d = read_mesh_components(mesh_2d, mesh_2d.topology().dim() - 1, os.path.join(mesh_2d_path, 'line_mesh.xdmf'))
    # mesh_2d_parameters = io.read_parameters_from_csv_file(os.path.join(mesh_2d_path, "mesh_metadata.csv"))

    coordinates_mesh_2d = mesh_2d.coordinates()

    Q_1d = f_1d.function_space()
    value_shape_1d = Q_1d.ufl_element().value_shape()
    value_size_1d = int(np.prod(value_shape_1d))
    dim_1d = Q_1d.mesh().geometry().dim()
    dof_indices_1d = Q_1d.dofmap().dofs()

    coordinates_all_1d = Q_1d.tabulate_dof_coordinates().reshape(-1, dim_1d)
    dof_coordinates_1d = coordinates_all_1d[::value_size_1d]


    # 2. read the parametric form of the shape in the 2d mesh
    indices_vertices_on_shape, cumulative_arc_length = shape_tool(mesh_2d, mf_mesh_2d, shape_coordinates, shape_id)



    #7. write the values of f_2d into f_1d

    for i in range(len(dof_coordinates_1d)):
        # run through all unique DOF coordinates of 1d mesh

        # return j in such a way that cumulative_arc_length[j] <= dof_coordinates_1d[i][0] < cumulative_arc_length[j+1]
        j = np.searchsorted(cumulative_arc_length, dof_coordinates_1d[i][0], side='right') - 1
        j = np.clip(j, 0, len(indices_vertices_on_shape) - 2)

        p_start = coordinates_mesh_2d[indices_vertices_on_shape[j]]
        p_end = coordinates_mesh_2d[indices_vertices_on_shape[j+1]]

        # p is the point in between p_start and p_end whose arc length along the shape corresponds to  dof_coordinates_1d[i][0] (the arc length of the DOF on the 1d mesh)
        p = np.add(p_start, 
                    np.multiply(
                            np.subtract(p_end, p_start), 
                            (dof_coordinates_1d[i][0] - cumulative_arc_length[j])/(cumulative_arc_length[j+1] - cumulative_arc_length[j])
                    )
                    )
    
        # print(f'to 1d vertex {dof_coordinates_1d[i][0]} corresponds 2d vertex {p}')
        # print(f'  f_2d(p)   = {np.atleast_1d(f_2d(p))[0]}')
        # print(f'  expected  = {p[0] + 2*p[1]}')

        # set the DOF of f_1d according to the value of f_2d computed on p
        for k in range(value_size_1d):
            # run through all components of the field and write them into f_1d

            f_1d.vector()[dof_indices_1d[value_size_1d * i + k]] = np.atleast_1d(f_2d(p))[k]
            

        # print(f'  f_1d(p)   = {f_1d(dof_coordinates_1d[i][0])}')



'''
given a 2d mesh with a shape boundary which is laid flat on a 1d mesh, it maps a point on the 1d mesh onto the 2d mesh
Input values: 
    - 'x': the coordinate of the point on the 1d mesh
    - 'mesh': the 2d mesh
    - 'mf_mesh': a function on the 2d mesh that tags mesh facets
    - 'shape_coordinates': [[p_0_x, p_0_y], [p_1_x, p_1_y], ...], the list of coordinates of the mesh vertices lying on the shape
    - 'shape_id': the id with which the shape is tagged on the 2d mesh
Return values: 
    - 'p': [p_x, p_y] the coordinates of the point corresponding to 'x' on the 2d mesh
'''
def map_1d_to_2d(x, mesh, mf_mesh, shape_coordinates, shape_id):

    # 1. initialize 
    # mesh = read_mesh(os.path.join(mesh_path, 'triangle_mesh.xdmf'))
    # mf_mesh = read_mesh_components(mesh, mesh.topology().dim() - 1, os.path.join(mesh_path, 'line_mesh.xdmf'))
    # mesh_parameters = io.read_parameters_from_csv_file(os.path.join(mesh_path, "mesh_metadata.csv"))
    mesh_coordinates = mesh.coordinates()

    # 2. read the parametric form of the shape in the 2d mesh
    indices_vertices_on_shape, cumulative_arc_length = shape_tool(mesh, mf_mesh,  shape_coordinates, shape_id)
  
    # return j in such a way that cumulative_arc_length[j] <= dof_coordinates_1d[i][0] < cumulative_arc_length[j+1]
    j = np.searchsorted(cumulative_arc_length, x, side='right') - 1
    j = np.clip(j, 0, len(indices_vertices_on_shape) - 2)

    p_start = mesh_coordinates[indices_vertices_on_shape[j]]
    p_end = mesh_coordinates[indices_vertices_on_shape[j+1]]

    # p is the point in between p_start and p_end whose arc length along the shape corresponds to  dof_coordinates_1d[i][0] (the arc length of the DOF on the 1d mesh)
    p = np.add(p_start, 
                    np.multiply(
                            np.subtract(p_end, p_start), 
                            (x - cumulative_arc_length[j])/(cumulative_arc_length[j+1] - cumulative_arc_length[j])
                    )
            )

    return p


 


'''
compute quantities related a to a shape (a one-dimensional manifold, a curve) embedded in a 2d mesh
Input values: 
    * Mandatory: 
        - 'mesh': the mesh
        - 'mf_mesh': a funciton defined on the mesh that tags its facets
        - 'shape_coordinates': [[p_0_x, p_0_y], [p_1_x, p_1_y], ...], the list of coordinates of the mesh vertices lying on the shape
        - 'shape_id': the ID withi which the shape is tagged in the 2d mesh

Return values: 
    - 'indices_vertices_on_shape': the indices (defined as in facet_vertex.index()) of the vertices on the 2d mesh, ordered in increasing order of the parameter t by which the shape is parameterized, 'indices_vertices_on_shape' = [index_v_t_0, index_v_t_1, ... ]
    - 'cumulative_arc_length': cumulative_arc_length[i] is the cumulated arc length from the beginning of the curve up to vertex with index indices_vertices_on_shape[i] included
'''
def shape_tool(mesh, mf_mesh, shape_coordinates, shape_id):

    # mesh = read_mesh(os.path.join(mesh_path, 'triangle_mesh.xdmf'))
    # mesh_parameters = io.read_parameters_from_csv_file(os.path.join(mesh_path, "mesh_metadata.csv"))
    # mf_mesh = read_mesh_components(mesh, mesh.topology().dim() - 1, os.path.join(mesh_path, 'line_mesh.xdmf'))
    mesh_coordinates = mesh.coordinates()

    # shape_coordinates = mesh_parameters['shape_coordinates']


    # 2. read the parametric form of the shape in the 2d mesh
    # shape_parametric_form = io.read_function_expresssion(parameters_mesh_2d['shape_parametric_form'])


    # 3. compute the facets of the 2d mesh that lie on shape: facets_on_shape contains the facets of the mesh of f_2d that have been tagged with ID 'shape_id'
    facets_on_shape = []

    for facet in facets(mesh):
        #run through all facets of mesh_0 

        if mf_mesh[facet] == shape_id:
            # the facet under consideration belongs to the shape

            facets_on_shape.append(facet)


    # 4. compute the vertices of the 2d mesh that lie on the shape
    # 4.1 initialize vertices_on_shape = [[v_0_x, v_0_y]] with the coordinates of the vertex on shape corresponding to the curvilinear coordinate t = 0
    indices_vertices_on_shape = []

    # 1. Add the first vertex
    for facet in facets_on_shape:
        #run through all facets_on_shape

        # find the facet that contains the first two vertices of the parametric curve of shape

        v_list = list(vertices(facet))
    
        if (np.isclose(v_list[0].point().array()[:2], shape_coordinates[0]).all()) and (np.isclose(v_list[1].point().array()[:2], shape_coordinates[1]).all()):
            # add the vertex under consideration if it is equal to coordinates_vertices_on_shape[0]

            indices_vertices_on_shape.append(v_list[0].index())

            break

        if (np.isclose(v_list[1].point().array()[:2], shape_coordinates[0]).all()) and (np.isclose(v_list[0].point().array()[:2], shape_coordinates[1]).all()):
            # add the vertex under consideration if it is equal to coordinates_vertices_on_shape[0]

            indices_vertices_on_shape.append(v_list[1].index())

            break

    # print(f'The vertex corresponding to t=0 is {coordinates_vertices_on_shape}, index = {indices_vertices_on_shape}')

    # 4.2 Add subsequent vertices by running on the edges in a sequential way
    used_facet_indices = set()

    while len(indices_vertices_on_shape) < len(shape_coordinates):
        # stop when you addedd N vertices

        for facet in facets_on_shape:
            # run through all facets on shape

            if facet.index() not in used_facet_indices:
                # if the facet under consideration has not been used already, proceed

                # build a list of vertices on the facet under consideration
                v_list = list(vertices(facet))

                # if the facet under consideration has one of its endpoints equal to the last added vertex to indices_vertices_on_shape, add it to indices_vertices_on_shape, update indices_vertices_on_shape and break
                if (v_list[0].index() == indices_vertices_on_shape[-1]):
                
                    used_facet_indices.add(facet.index())
                    indices_vertices_on_shape.append(v_list[1].index())

                    break

                if (v_list[1].index() == indices_vertices_on_shape[-1]):
                
                    used_facet_indices.add(facet.index())
                    indices_vertices_on_shape.append(v_list[0].index())

                    break


    # 
    # print(f'finished, indices_vertices_on_shape = {indices_vertices_on_shape}')


    # 5. compute the arc length along the shape in the 2d mesh
    l = 0.0
    cumulative_arc_length = [l]

    for i in range(1, len(indices_vertices_on_shape)):

        delta_l =  np.linalg.norm(np.subtract(mesh_coordinates[indices_vertices_on_shape[i]], mesh_coordinates[indices_vertices_on_shape[i-1]]))

        l += delta_l
        cumulative_arc_length.append(l)

    delta_l = np.linalg.norm(np.subtract(mesh_coordinates[indices_vertices_on_shape[-1]], mesh_coordinates[indices_vertices_on_shape[0]]))

    l += delta_l
    cumulative_arc_length.append(l)

    # append last vertex index to account for periodicity of the shape
    indices_vertices_on_shape.append(indices_vertices_on_shape[0])


    return indices_vertices_on_shape, cumulative_arc_length


'''
Given 2d mesh given by a recangle with a meshed shape in it, and a line mesh obtained by lying the shape boundary on a line, this method transfers a field (scalar, vector or tensor) defined on the 1d mesh, on the 2d mesh. 

Input values: 
    * Mandatory:
        - 'f_1d': the field on the 2d mesh
        - 'f_2d': the field on the 2d mesh
        - 'mesh_2d': the 2d mesh is stored
        - 'mf_mesh_2d': a function on 'mesh_2d' that tags its facets
        - 'shape_coordinates' : [[p_0_x, p_0_y], [p_1_x, p_1_y], ... ] the coordinates of the vertices of the shape in 'mesh_2d'
        - 'shape_id': the ID with which the shape is tagged in the 2d mesh 
    * Optional:
        - 'epislon': the accuracy threshold to identify to which a vertex belongs to a segment in the 2d mesh 
'''

def transfer_1d_to_2d(f_1d, f_2d, mesh_2d, mf_mesh_2d, shape_coordinates, shape_id,
                      epsilon = const.epsilon):

    # 1. initialize 
    # mesh_2d = read_mesh(os.path.join(mesh_2d_path, 'triangle_mesh.xdmf'))
    # mf_mesh_2d = read_mesh_components(mesh_2d, mesh_2d.topology().dim() - 1, os.path.join(mesh_2d_path, 'line_mesh.xdmf'))
    # mesh_2d_parameters = io.read_parameters_from_csv_file(os.path.join(mesh_2d_path, "mesh_metadata.csv"))

    coordinates_mesh_2d = mesh_2d.coordinates()

    Q_2d = f_2d.function_space()
    value_shape_2d = Q_2d.ufl_element().value_shape()
    value_size_2d = int(np.prod(value_shape_2d))
    dim_2d = Q_2d.mesh().geometry().dim()
    dof_indices_2d = Q_2d.dofmap().dofs()

    coordinates_all_2d = Q_2d.tabulate_dof_coordinates().reshape(-1, dim_2d)
    dof_coordinates_2d = coordinates_all_2d[::value_size_2d]

    indices_vertices_on_shape, cumulative_arc_length = shape_tool(mesh_2d, mf_mesh_2d, shape_coordinates, shape_id)


    #7. write the values of f_1d into f_2d

    for i in range(len(dof_coordinates_2d)):
        # coordinates of the i-th node in the 2d mesh

        coordinate_2d = dof_coordinates_2d[i][:2]

        for j in range(len(indices_vertices_on_shape) - 1):

            # segment j: from vertex j to vertex j+1 on the shape
            shape_vertex_start = coordinates_mesh_2d[indices_vertices_on_shape[j]][:2]
            shape_vertex_end = coordinates_mesh_2d[indices_vertices_on_shape[j + 1]][:2]

            shape_edge_dr = shape_vertex_end - shape_vertex_start
            shape_edge_dr_length = np.linalg.norm(shape_edge_dr)
            delta = coordinate_2d - shape_vertex_start

            # l is the length of the projection of delta along the line going through shape_vertex_start to and shape_vertex_end        
            l = np.dot(delta, shape_edge_dr) / shape_edge_dr_length

            # projection is the orthogonal projection of the DOF coordinate under consideration on the line going through shape_vertex_start and shape_vertex_end
            projection = shape_vertex_start + l * shape_edge_dr / shape_edge_dr_length
            residual = np.linalg.norm(projection - np.array(coordinate_2d))


            if residual < epsilon and (- epsilon < l < shape_edge_dr_length + epsilon):
                # DOF lies on segment j — compute its arc length

                arc_length = cumulative_arc_length[j] + np.clip(l, 0.0, shape_edge_dr_length)

                # write into f_2d for each component
                for component in range(value_size_2d):

                    (f_2d.vector())[dof_indices_2d[i * value_size_2d + component]] = (np.atleast_1d(f_1d(arc_length)))[component]

                break  # no need to check other segments in the j loop



'''
tag a physical object, or a list of objects, in a mesh
Input values: 
    * Mandatory:    
        - 'object': the object to be tagged, e.g. a line or a list of lines
        - 'id': an integer, the tag that will be given to 'object'
        - 'model': the model used to generate the mesh, e.g., gmsh.model
    * Optional:
        - 'label': the label to be given to the object

'''
def tag_physical_object(object, id, model, 
                        label=''):

    if isinstance(object, list):
        # 'object' is a list -> take as dimension the dimension of its first entry
        dim = object[0][0]
        object_to_tag = [object[i][1] for i in range(len(object))]

    else: 
        # 'object' is not a list -> take as dimension the dimension of 'object'
        dim = object[0]
        object_to_tag = [object[1]]

    model.addPhysicalGroup(dim, object_to_tag, id)
    model.setPhysicalName(dim, id, label)


'''
given a field f (scalar, vector, or tensor) on  mesh A, and a deformation field that trasnforms mesh A into mesh B, and a field g (same type as f) on mesh B, set g equal to f
Input values: 
    - 'f': function on mesh A
    - 'g': function on mesh B
    - 'u': displacement field, defined on mesh A
'''
def transfer(f, g, u):

    f_def = fu.deform_function(f, u)
    f_def.set_allow_extrapolation(True)

    Q_g = g.function_space()

    g_value_shape = Q_g.ufl_element().value_shape()
    g_value_size = int(np.prod(g_value_shape))

    g_dim = Q_g.mesh().geometry().dim()

    g_dof_coordinates_all = Q_g.tabulate_dof_coordinates().reshape(-1, g_dim)

    '''
    subsample coordinates by skipping repeats (one physical point per value_size DOFs)
    Run through g_dof_coordinates_all by taking every g_value_size entry in it, and writes the result into g_dof_coordinates
    '''
    g_dof_coordinates = g_dof_coordinates_all[::g_value_size]

    # write the values of f into g
    for i in range(len(g_dof_coordinates)):
        # run through all unique DOF coordinates 
       
       for j in range(g_value_size):
        # run through all components of the field f and write them into g

        g.vector()[g_value_size * i + j] = np.atleast_1d(f_def(g_dof_coordinates[i]))[j]

'''
given a DG field f (scalar, vector, or tensor) on  mesh A, and a deformation field that trasnforms mesh A into mesh B, and a DG field g (same type as f) on mesh B, set g equal to f
This method is extends `transfter` to DG fields. 

Input values:
    - 'f': function on mesh A
    - 'g': function on mesh B
    - 'u': displacement field, defined on mesh A
    - 'sf_f': cell MeshFunction ("size_t") marking the triangles of mesh A 
    - 'sf_g': cell MeshFunction ("size_t") marking the triangles of mesh B
'''
def transfer_dg(f, g, u, sf_f, sf_g):

    f_def = fu.deform_function(f, u)
    f_def.set_allow_extrapolation(True)

    mesh_f = f_def.function_space().mesh()
    tree_f = mesh_f.bounding_box_tree()

    Q_g = g.function_space()

    g_value_shape = Q_g.ufl_element().value_shape()
    g_value_size = int(np.prod(g_value_shape))

    g_mesh = Q_g.mesh()
    g_dim = g_mesh.geometry().dim()


    '''
    g_dof_coordinates_all stores the coordinates of the points where DOFs sit. Because the field 'g' defined on each DOF has value_size components, dof_coordinates is composed of blocks, where each block has 'value_size' entries, and blocks are all identical
    For example, dof_coordinates is of the form ->
        row 0:  [x0, y0]   ← this corresponds to g[0] at DOF point 0
        row 1:  [x0, y0]   ← this corresponds to g[1] at DOF point 0
        ...
        row value_size  [x1, y1]   ← this corresponds to g[0] at DOF point 1
        row value_size+1 [x1, y1]   ← this corresponds to g[1] at DOF point 1
        ...
        
    '''
    g_dof_coordinates_all = Q_g.tabulate_dof_coordinates().reshape(-1, g_dim)

    '''
    subsample coordinates by skipping repeats (one physical point per value_size DOFs)
    '''
    g_dof_coordinates = g_dof_coordinates_all[::g_value_size]

    # print(f'g DOF coordinates = {np.array2string(g_dof_coordinates, threshold=np.inf)}', flush=True)

    '''
    map each DOF of g to the cell of g's mesh that owns it: for a DG space
    every DOF belongs to exactly one cell

    Here

    g_dofmap.cell_dofs(cell.index()) = 
        [ID of 1st DOF sitting on cell `cell`, 
        ID of 2nd DOF sitting on cell `cell`, 
        ... ]

    dof_to_cell[ID of DOF] = index of the cell to which DOF belongs
    '''
    g_dofmap = Q_g.dofmap()
    dof_to_cell = np.empty(Q_g.dim(), dtype=np.int64)
    for cell in cells(g_mesh):
        dof_to_cell[g_dofmap.cell_dofs(cell.index())] = cell.index()

    # print(f'DOF to cell = {np.array2string(dof_to_cell, threshold=np.inf)}', flush=True)

    '''
    sf_f_values[cell_id] = value of the MeshFunction `sf_f` on cell `cell_id`
    sf_g_values[cell_id] = value of the MeshFunction `sf_g` on cell `cell_id`
    '''
    sf_f_values = sf_f.array()
    sf_g_values = sf_g.array()

    '''
    values is an array whose length is the number of components of the `f` 
    '''
    values = np.empty(g_value_size)

    # write the values of f into g
    for i in range(len(g_dof_coordinates)):
        # run through all unique DOF coordinates of `g`

        x = g_dof_coordinates[i]

        '''
        given the DOF under consideration, store into `DOF_id` the value of the MeshFunction `sf_g` on the cell to which the DOF belongs
        '''
        DOF_region_tag = sf_g_values[dof_to_cell[g_value_size * i]]


        # all cells of f_def's mesh whose closure contains x
        candidate_f_cells = tree_f.compute_entity_collisions(Point(*x))

        # print(f'candidate_f_cells = {candidate_f_cells}')

        '''
        run through `candidate_f_cells` and stop as soon as one finds a cell whose region tag is the same as   `DOF_region_tag`
        '''
        cell_index = -1
        for c in candidate_f_cells:
            if sf_f_values[c] == DOF_region_tag:
                cell_index = c
                break

        if cell_index >= 0:
            '''
            a cell whose region tag is the same as   `DOF_region_tag` has been found -> write into `values` the value of `f_def` corresponding to cell `cell_index`
            f_def may be discontinuous across cells, and here its value relative to cell `cell_index` is used
            '''

            f_def.eval_cell(values, x, Cell(mesh_f, cell_index))

            # print(f'wrote into values = {values}')

        else:

            # sign

            # x is contained in no cell of the matching region: the deformed old mesh and the new mesh mismatch by a tiny amount at the boundaries (solver tolerance / coordinate round-trip). Extrapolate from the nearest cell OF THE MATCHING REGION, so the evaluation stays on the correct side of the interface

            distance_min = np.inf

            for c in cells(mesh_f):
                # run through all cells in `mesh_f`

                if sf_f_values[c.index()] == DOF_region_tag:
                    # the cell under consideration has the same region tag as `DOF_region_tag`

                    distance = c.distance(Point(*x))

                    if distance < distance_min:

                        distance_min = distance
                        cell_index_nearest = c.index()

            '''
            at the end of the loop `distance_min` is the distance to the closes cell to the DOF under consideration which has tag `DOF_region_tag`, and `cell_index_nearest` is the index of such cell

            write into `values` the DOF of `f_def` computed on `x`: f_def may be discontinuous across cells, and here its value relative to cell `cell_index_nearest` is used
            '''

            f_def.eval_cell(values, x, Cell(mesh_f, cell_index_nearest))

            # print(f'Warning: transfer_dg fallback at x = {x}, tag = {DOF_region_tag}, extrapolated from cell at distance {distance_min:e}', flush=True)

        for j in range(g_value_size):
            # run through all components of the field f and write `value`, i.e., the value of `f` on the DOF under consideration, into g

            g.vector()[g_value_size * i + j] = values[j]
            
'''
given a fiels (scalar, vector, tensor) f defined on a 1d mesh and a function g (same type as f) defnied on another 1d mesh which has the same length as the 1d mesh of g, transfer the profile of f into g

Input values: 
    - 'f': the field to be read. Note that this method will do f.set_allow_extrapolation(True)
    - 'g': the field to be written in

'''

def transfer_1d(f, g):

    f.set_allow_extrapolation(True)

    Q_g = g.function_space()

    value_shape = Q_g.ufl_element().value_shape()
    value_size  = int(np.prod(value_shape)) if value_shape else 1


    # unique DOF coordinates: tabulate_dof_coordinates repeats each position
    # value_size times, so stride by value_size to get unique positions
    dof_coords  = (Q_g.tabulate_dof_coordinates())[::value_size]

    dof_map = Q_g.dofmap().dofs()
    dof_values = g.vector().get_local()


    for i in range(len(dof_coords)):
        # run through all coordinates in the 1d mesh

        s  = dof_coords[i][0]

        value = np.atleast_1d(np.array(f(s)))

        for k in range(value_size):

            dof_values[dof_map[value_size * i + k]] = value[k]

    g.vector().set_local(dof_values)
    g.vector().apply("insert")

'''
compute the mesh quality, defined as the minimal value of d r_in / r_out across all mesh cells
Input values; 
    - 'mesh': the mesh
Return values; 
    - 'result': the mesh quality
'''
def custom_mesh_quality(mesh):

    result, _ = MeshQuality.radius_ratio_min_max(mesh)

    return result

'''
return the jump in a field with respect to a facet normal for discontinuous function spaces
Input values: 
    - 'u': the field
    -  'n': the facet normal
Return values: 
    - 'u("+") * n("+")[alpha] + u("-") * n("-")[alpha]': the jump

'''
def jump(u, n): 

    return as_tensor(u("+") * (n("+"))[alpha] + u("-") * (n("-"))[alpha], (alpha))


'''
Return the average of a field across facets in a discontinuous function space
Input values: 
    - 'u': the field
Return values: 
    -  the average (u('+')+u('-'))/2
'''
def average(u):

    return (u("+")+u("-"))/2

'''
set field defined on a DG space equal to a profile in a mesh region 
Input values: 
    * Mandatory:
        - 'f': the field defined on a DG space
        - 'g': the profile to which 'f' will be set
    * Optional:
        - 'sf': 'None' by default, the mesh function that tags mesh surfaces. 
        - 'region_id': 'None' by default, the id of the mesh region (surface) on which 'f' will be set equal to 'g'. If 'id' is 'None' then this method will run through all cells in the mesh, 'f' to 'g' on the cell DOFs
'''
def interpolate_dg(f, g, sf=None, region_id=None):

    Q = f.function_space()
    element = Q.ufl_element()

    if (element.family() != 'Discontinuous Lagrange'):
        # the method has been called on a field defined on a continuous function space -> throw an error and exit

        print(f'{col.Fore.RED}Error: interpolate_dg has been called on a field with a continuous function space!! Stopping now.{col.Fore.RESET}')

        sys.exit(1)

    if (element.value_shape() != g.value_shape()):
        # the value shape of Q and that of g differ -> check whether this is due to a 'convention' issue where scalars have been given a value shape of (1,) vs. ()

        if ((((element.value_shape() == ()) and (g.value_shape() == (1,))) or ((element.value_shape() == (1,)) and  (g.value_shape() == ()))) == False):
            # the discrepancy was not due to a convention issue -> throw an error and exit

            print(f'{col.Fore.RED}Error: value shapes are different!!\n\telement value shape = {element.value_shape()}\n\tg value shape= {g.value_shape()}\nStopping now.{col.Fore.RESET}')

            sys.exit(1)
   
    value_size  = int(np.prod(element.value_shape())) if element.value_shape() else 1

    mesh = Q.mesh()

    '''
    dof_coordinates stores the coordinates of the points where DOFs sit. Because the field 'f' defined on each DOF has value_size components, dof_coordinates is composed of blocks, where each block has 'value_size' entries, and blocks are all identical
    For example, dof_coordinates is of the form ->
        row 0:  [x0, y0]   ← this corresponds to f[0] at DOF point 0
        row 1:  [x0, y0]   ← this corresponds to f[1] at DOF point 0
        ...
        row value_size  [x1, y1]   ← this corresponds to f[0] at DOF point 1
        row value_size+1 [x1, y1]   ← this corresponds to f[1] at DOF point 1
        ...
        
    '''
    dof_coordinates = Q.tabulate_dof_coordinates()


    '''
    f_values contains the value of 'f' on the DOFs, and it has the same structure as 'dof_coordinates'
    f_values = 
            entry 0: f[0] at DOF point 0
            entry 1: f[1] at DOF point 0
            ...
            entry value_size  f[0] at DOF point 1
            entry value_size + 1 f[1] at DOF point 1
            ...
        
    '''
    f_values = f.vector().get_local()   # get a copy of field values
    
    for cell in cells(mesh):
        # run on all mesh cells

        if region_id != None:
            #region_id has been given when calling this method -> evaluate sf on the cell to obtain the tag of cell `cell` 

            # compute 'sf' on the cell; under consideration
            cell_tag = sf[cell]

        '''
        cell_dofs contains the IDs of the DOFs that are contained into 'cell', it has the structure
        [
            id_f_0_on_DOF_0, 
            id_f_0_on_DOF_1,
            ...,
            id_f_0_on_DOF_{n_nodes-1},

            id_f_1_on_DOF_0, 
            id_f_1_on_DOF_1,
            ...,
            id_f_1_on_DOF_{n_nodes-1},

            ...
        ]
        where the pattern is repeated value_size times, i.e., one for each component of 'f', and n_nodes = [number of DOFs in the cell] / [value_size]. In other words

        cell_dofs[j * n_nodes + i] = [index in f.values().get_local() corresponding to the j-th component of the field 'f' sitting on ith DOF in the cell 'cell']
        '''
        cell_dofs = Q.dofmap().cell_dofs(cell.index())

        n_nodes = len(cell_dofs) // value_size


        '''
        remove the redundancy in cell_dofs and store the result in 
        cell_dofs_unique = 
            [
                id_DOF_0, 
                id_DOF_1,
                ...,
                id_DOF_{n_nodes-1}
            ]
        
        '''
        cell_dofs_unique = cell_dofs[:n_nodes]


        if (region_id == None) or (cell_tag == region_id):
            # if 'cell_tag' == 'id', then the cell under consideration belongs to the surface tagged with 'id' -> set the DOFs of 'f' relative to this cell according to 'g'

            for i in range(len(cell_dofs_unique)):
                # run over physical DOFs contained to 'cell' and print out the value of 'f' by specifying that those DOFs belong to region tagged with 'cell_tag' in a separate column of the csv output file. Note that, because the space of 'f' is discontinuous, here DOFs in 'cell' may belong to different mesh regions, and thus have different tags

                # pad 'x' to three dimensions
                dof_coordinate = dof_coordinates[cell_dofs_unique[i]]

                for j in range(value_size):
                    f_values[cell_dofs[j * n_nodes + i]] = np.atleast_1d(g(dof_coordinate[:2]))[j]


    f.vector().set_local(f_values) 
    f.vector().apply("insert")


'''
recognize to which mesh subregions + and - parts of an internal boundary correspond in a mesh
Input values:
    - 'mesh': the mesh
    - 'sf': the mesh function that tags mesh facets
    - 'region_a_id', 'region_b_id': the tags with which regions a and b are tagged in 'sf'
    - 'dS_ab': the measure corresponding to the facets between region a and b
Return values: 
    - [symbol corresponding to region a, symbol corresponding to region b]. For example, if region a corresponds to '-' and region b to '+', this method returns ['-', '+']

'''
def plus_minus(mesh, sf, region_a_id, region_b_id, dS_ab):

    Q = FunctionSpace(mesh, 'DG', 0)

    class a_expression(UserExpression):
        def eval(self, values, x):

            values[0] = 1.0

        def value_shape(self):
            return (1,)
        
    class b_expression(UserExpression):
        def eval(self, values, x):

            values[0] = -1.0

        def value_shape(self):
            return (1,)

    u = Function(Q)

    interpolate_dg(u, a_expression(), sf, region_a_id)
    interpolate_dg(u, b_expression(), sf, region_b_id)

    mean_p = assemble(u("+") * dS_ab) / assemble(Constant(1.0) * dS_ab)
    mean_m = assemble(u("-") * dS_ab) / assemble(Constant(1.0) * dS_ab)

    if (np.isclose(mean_p, 1)) and (np.isclose(mean_m, -1)):

        return ['+', '-']
    
    elif (np.isclose(mean_p, -1)) and (np.isclose(mean_m, 1)):

        return ['-', '+']

    else:

        print(f"{col.Fore.RED}Error: plus_minus failed!{col.Style.RESET_ALL}")
        sys.exit(1)



def ufl_conditional_form(mesh, sf, form_a, form_b, tag_a, tag_b):

    Q = FunctionSpace(mesh, 'DG', 0)
    cell_tag = Function(Q)

    cell_tag.vector()[:] = sf.array()   # DG0 dofs are ordered by cell index

    # build f as a pure UFL expression
    result = conditional(
                ufl.eq(cell_tag, tag_a),
                        form_a,
                        conditional(ufl.eq(cell_tag, tag_b), 
                                    form_b,
                                    form_b * 0
                                    )
                )
    
    return result


'''
Return key identifying the DOF at coordinate x on an interface facet:
Input values: 
    * Mandatory:
        - 'x': the DOF coordinates
        - 'facet_vertex_ids': a list containing the IDs of the facet vertices
        - 'facet_id': the ID of the facet
        - 'coordinates': mesh.coordinates()
        - 'degree': the degree of the polynomial space of the field under consideration
    * Optional: 
        - 'tol': the ditance tolerance used to tell whether `x` lies on the edge segment 

Return values: 
    - ('v', vertex_id)          if x coincides with a facet vertex
    - ('e', facet_id, t_int)    if x is an interior edge DOF. Here t_int = round(t * degree) = [1, 2, 3, ...] identifies the canonical number of `x` along the edge. Here `facet_id` is the ID of the facet under consideration 
'''

def get_key(x, facet_vertex_ids, facet_id, coordinates, degree, tol=const.epsilon):

    #1. check if x is at one of the facet's extremal vertices
    for v_id in facet_vertex_ids:
        # run through IDs of extremal vertices of the facet under consideration

        if np.allclose(x, coordinates[v_id], atol=tol):
            # `x` coincides with the coordinates of one of these extremal vertices -> return ('v', [ID of the extremal vertex that coincides with `x`])

            return ('v', int(v_id))

    #2. check if `x` is an interior edge DOF: compute t along canonical direction

    v0_id, v1_id = sorted(int(v) for v in facet_vertex_ids)

    # coordinates of the extremal vertices of the facet
    x_0 = coordinates[v0_id]
    x_1 = coordinates[v1_id]

    # vector going from `x_0` to `x_1`
    d = x_1 - x_0

    # 0<t<1 locates `x` along the facet: t=0 means that `x` is `x_0` and t=1 means that `x` is `x_1`
    t = np.dot(x - x_0, d) / np.dot(d, d)

    '''
    For a Lagrange element of degree k, the interior edge nodes are placed at equally spaced positions t = 1/k, 2/k, ..., (k-1)/k. Multiplying by degree maps these to integers 1, 2, ..., k-1. round handles floating point imprecision, a
    '''
    t_int = int(round(t * degree))

    return ('e', int(facet_id), t_int)


'''
Consider a DG field `f` (scalar, vector or tensor) defined on a mesh divided into two surfaces which are delimited by a shape. Here `f` may be discontinuous at the shape. This method overwrites the DOFs of `f` at the shape by setting them equal to the DOFs of surface_1 -> DOFs belonging to surface_0 are overwritten
Input values; 
    * Mandatory: 
        - `f`: the field
        - `sf`: the mesh function tagging surfaces
        - `mf_I`: the mesh function tagging interior facets
        - `shape_id`: tag of the shape
        - `surface_0_id`, `surface_1_id`: tags of surface_0 and surface_0

    * Optional:
        - `tol`: tolerance used for spatial distances

'''
def overwrite_interface_dofs(f, sf, mf_I, shape_id, surface_0_id, surface_1_id, tol=const.epsilon):

    Q = f.function_space()
    mesh = Q.mesh()
    mesh.init(1, 2)

    degree = Q.ufl_element().degree()
    value_shape = Q.ufl_element().value_shape()
    value_size = int(np.prod(value_shape)) if value_shape else 1

    dof_coordinates = Q.tabulate_dof_coordinates()
    f_values = f.vector().get_local()
    coordinates = mesh.coordinates()
    dofmap = Q.dofmap()



    '''
        Step 1: build surface_1-side interface map 
        This step would build a list, interface_map, which is 
        interface_map = {
        ([key of DOF 0 lying on shape], [values of f on DOF 0 lying on shape],
        ([key of DOF 1 lying on shape], [values of f on DOF 1 lying on shape],
        ...
        )
        
        }
    '''
    #  interface_vertex_ids  is the set of IDs of all vertices that lie on the shape
    spahe_vertex_ids = set()

    fluid_interface_map = {}

    for facet in facets(mesh):
        # run through all mesh facets

        if mf_I[facet] == shape_id:
            # `facet` belongs to the shape interface

            # `facet_vertex_ids` is a list of ids of vertices belonging to `facet`
            facet_vertex_ids = facet.entities(0)
            facet_vertex_coords = coordinates[facet_vertex_ids]

            for v_id in facet_vertex_ids:
                spahe_vertex_ids.add(int(v_id))

            for cell_id in facet.entities(2):
                # run through all cells that neighbor `facet`

                cell = Cell(mesh, cell_id)

                if sf[cell] == surface_1_id:
                    # `cell` belongs to surface_1

                    '''
                    cell_dofs contains the IDs of the DOFs that are contained into 'cell', it has the structure
                    [
                        id_f_0_on_DOF_0, 
                        id_f_0_on_DOF_1,
                        ...,
                        id_f_0_on_DOF_{n_nodes-1},

                        id_f_1_on_DOF_0, 
                        id_f_1_on_DOF_1,
                        ...,
                        id_f_1_on_DOF_{n_nodes-1},

                        ...
                    ]
                    where the pattern is repeated value_size times, i.e., one for each component of 'f', and n_nodes = [number of DOFs in the cell] / [value_size]. In other words

                    cell_dofs[j * n_nodes + i] = [index in f.values().get_local() corresponding to the j-th component of the field 'f' sitting on ith DOF in the cell 'cell']
                    '''

                    cell_dofs = dofmap.cell_dofs(cell.index())
                    n_nodes = len(cell_dofs) // value_size

                    for i in range(n_nodes):
                        # run through all physical DOFs in `cell`

                        x = dof_coordinates[cell_dofs[i]][:2]

                        if cal.point_on_segment(x, facet_vertex_coords[0], facet_vertex_coords[1]):

                            # `x` lies on `facet`: get its key as defined in `get_key`
                            key = get_key(x, facet_vertex_ids, facet.index(), coordinates, degree, tol)

                            if key not in fluid_interface_map:
                                ''' 
                                append to fluid_interface map
                                    ([key of the DOF corresponding to `x`], value of `f` on that DOF)
                                '''
                                fluid_interface_map[key] = [f_values[cell_dofs[j * n_nodes + i]] for j in range(value_size)]

    '''
        Step 2: patch all shape (surface_0) DOFs at interface locations 
        This step will write into DOFs of `f` belongin to `surface_0` and lying on the shape (which are stored in `fluid_interface_map`) the values of `f` in `surface_1`
    '''  
    for cell in cells(mesh):
        # run through all cells in the mesh

        if sf[cell] == surface_0_id:
            # `cell belongs to surface_0 -> proceed with overwriting

            '''
                cell_dofs contains the IDs of the DOFs that are contained into 'cell', it has the structure
                [
                    id_f_0_on_DOF_0, 
                    id_f_0_on_DOF_1,
                    ...,
                    id_f_0_on_DOF_{n_nodes-1},

                    id_f_1_on_DOF_0, 
                    id_f_1_on_DOF_1,
                    ...,
                    id_f_1_on_DOF_{n_nodes-1},

                    ...
                ]
                where the pattern is repeated value_size times, i.e., one for each component of 'f', and n_nodes = [number of DOFs in the cell] / [value_size]. In other words

                cell_dofs[j * n_nodes + i] = [index in f.values().get_local() corresponding to the j-th component of the field 'f' sitting on ith DOF in the cell 'cell']
            '''

            cell_dofs = dofmap.cell_dofs(cell.index())
            n_nodes = len(cell_dofs) // value_size

            for i in range(n_nodes):
                # run through all physical DOFs in `cell`

                # consider a DOF with coordinates `x`
                x = dof_coordinates[cell_dofs[i]][:2]

                key = None

                '''             
                1. check if `x` is  an interface vertex
                '''
                for v_id in cell.entities(0):
                    # run through all vertices that belong to `cell`


                    if np.allclose(x, coordinates[v_id], atol=tol) and (int(v_id) in spahe_vertex_ids):
                        ''' 
                        the DOF coordinate `x` under consideration coincides with one of the cell vertices (1st condition) and it is one of the shape vertices -> add  to `key`
                        ('v', [id of the vertex corresponding to `x`]) (2nd condition)
                        '''

                        key = ('v', int(v_id))
                        break

                ''' 
                2.: check if `x` lies on an interface facet of  `cell`
                this catches edge-interior DOFs for degree >= 2
                '''
                if key is None:
                    # `x` is not one of the cell vertices -> check whethe it lies on a facet lying on the shape
                    
                    for facet in facets(cell):
                        # run through all facets of `cell`

                        if mf_I[facet] == shape_id:
                            # `facet` belongs to the shape

                            # build list of IDs and coordinates extremal vertices of `facets`
                            facet_vertex_ids = facet.entities(0)
                            facet_vertex_coords = coordinates[facet_vertex_ids]

                            if cal.point_on_segment(x, facet_vertex_coords[0], facet_vertex_coords[1]):
                                # the coodinates `x` of the DOF under consideration lie on the segment of `facet` -> add it to `key`

                                key = get_key(x, facet_vertex_ids, facet.index(), coordinates, degree, tol=tol)
                                break

                if (key is not None) and (key in fluid_interface_map):
                    # the DOF with coordinates `x` is either a vertex belonging to the shape, or it lies in between an edge belonging to the shape -> it its on the shape -> overwrite into f_values the value of `f` stored in fluid_interface_map (i.e., the vlaues of `f` into region_0)
                    for j in range(value_size):

                        f_values[cell_dofs[j * n_nodes + i]] = fluid_interface_map[key][j]

    f.vector().set_local(f_values)
    f.vector().apply("insert")

'''
clear gmsh and geometry if gmsh is initialized

Inpuf values: 
    - `geometry`: gmsh's geometry object
'''
def clear_gmsh(geometry):

    if gmsh.isInitialized():
        
        gmsh.clear()
        geometry.__exit__()