from fenics import *
import numpy as np
import colorama as col
import gmsh
import meshio

import geometry as geo
import input_output as io


def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:, :2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(
        points=points, cells={cell_type: cells}, cell_data={"name_to_read": [cell_data]}
    )
    return out_mesh

# read the mesh form 'filename' and return it
def read_mesh(filename):
    mesh = Mesh()

    xdmf = XDMFFile( mesh.mpi_comm(), filename )
    xdmf.read( mesh )
    xdmf.close()

    return mesh

'''
read the mesh  from  the .msh file 'infile' and write the mesh components (tetrahedra, triangles, lines, vertices) to 'outfile' (tetrahedron_mesh.xdmf, triangle_mesh.xdmf ...)
the component type can be "tera", "triangle", "line" or "vertex"
if 'prune_z' = true (false), the z component will be removed from the mesh
'''
def write_mesh_components(infile, outfile, component_type, prune_z):
    mesh_from_file = meshio.read( infile )
    component_mesh = create_mesh( mesh_from_file, component_type, prune_z )
    meshio.write( outfile, component_mesh )


'''
given a mesh 'mesh', read its components of dimension 'dim' stored into 'filename' and returns the collection of components
Example: to read the lines of the mesh, call this method with 
cf = msh.read_mesh_components(mesh, 1, (args.input_directory) + "/line_mesh.xdmf")
'''
def read_mesh_components(mesh, dim, filename):
    mesh_value_collection = MeshValueCollection( "size_t", mesh, dim )
    with XDMFFile( filename ) as infile:
        infile.read( mesh_value_collection, "name_to_read" )
        infile.close()
    return cpp.mesh.MeshFunctionSizet( mesh, mesh_value_collection )

#compare the numerical value of the integral of a test function over a ds, dx, .... with the exact one and output the relative difference
def test_mesh_integral(exact_value, f_test, measure, label):
    numerical_value = assemble( f_test * measure )
    print( f"{label} = {numerical_value:.{4}}, should be {exact_value:.{4}}, relative error =  {col.Fore.YELLOW}{abs( (numerical_value - exact_value) / exact_value ):.{io.number_of_decimals}e}{col.Style.RESET_ALL}" )


class BoundaryMarker( SubDomain ):
    def inside(self, x, on_boundary):
        return on_boundary


# returns the boundary points of the mesh `mesh`
def boundary_points(mesh):
    # create a dummy function space of degree 1 which will be used only to extract the boundary points
    Q_dummy = FunctionSpace( mesh, 'CG', 1 )

    # a map which takes as an input a vertex of Q_dummy.mesh and returns its corresponding degree of freedom
    vertex_to_degree_of_freedom_map = vertex_to_dof_map( Q_dummy )

    # a function which takes as argument the mesh vertices
    vertex_function = MeshFunction( "size_t", mesh, 0 )

    # set vertex_function -> 1 on the vertices which are part of the boundary (vertex_function is zero elsewhere)
    vertex_function.set_all( 0 )
    BoundaryMarker().mark( vertex_function, 1 )

    # collect the vertices where the vertex_function = 1, i.e., the vertices on the boundary
    boundary_vertices = np.asarray( vertex_function.where_equal( 1 ) )

    degrees_of_freedom = vertex_to_degree_of_freedom_map[boundary_vertices]

    x = Q_dummy.tabulate_dof_coordinates()
    x = x[degrees_of_freedom]

    # csvfile = open( "test_boundary_points.csv", "w" )
    # for p in x:
    #     print( f"{p[0]},{p[1]}", file=csvfile )
    # csvfile.close()

    # print("Degrees of freedom on the boundary:")
    # for degree_of_freedom in degrees_of_freedom:
    # print(f"\t{x[degree_of_freedom]}, {geo.my_norm( x[degree_of_freedom])}")

    return x


# returns the bulk points of the mesh `mesh`
def bulk_points(mesh):
    # create a dummy function space of degree 1 which will be used only to extract the boundary points
    Q_dummy = FunctionSpace( mesh, 'CG', 1 )

    # a map which takes as an input a vertex of Q_dummy.mesh and returns its corresponding degree of freedom
    vertex_to_degree_of_freedom_map = vertex_to_dof_map( Q_dummy )

    # a function which takes as argument the mesh vertices
    vertex_function = MeshFunction( "size_t", mesh, 0 )

    # set vertex_function -> 1 on the vertices which are part of the boundary (vertex_function is zero elsewhere)
    vertex_function.set_all( 0 )
    BoundaryMarker().mark( vertex_function, 1 )

    # collect the vertices where the vertex_function = 0, i.e., the vertices in the bulk
    boundary_vertices = np.asarray( vertex_function.where_equal( 0 ) )

    degrees_of_freedom = vertex_to_degree_of_freedom_map[boundary_vertices]

    x = Q_dummy.tabulate_dof_coordinates()
    x = x[degrees_of_freedom]

    # csvfile = open( "test_bulk_points.csv", "w" )
    # for p in x:
    #     print( f"{p[0]},{p[1]}", file=csvfile )
    # csvfile.close()

    # print("Degrees of freedom on the boundary:")
    # for degree_of_freedom in degrees_of_freedom:
    # print(f"\t{x[degree_of_freedom]}, {geo.my_norm( x[degree_of_freedom])}")

    return x


#return the set of boundary points whose distance from the point c lies between r and R
def boundary_points_circle(mesh, r, R, c):
    points = boundary_points(mesh)

    x = []
    for point in points:
        if((geo.my_norm( point - c  ) > r) and (geo.my_norm( point - c  ) < R)):
            x.append( point )

    # csvfile = open( "test_boundary_points_circle.csv", "w" )
    # for p in x:
    #     print( f"{p[0]},{p[1]}", file=csvfile )
    # csvfile.close()

    return x

#compute the lowest and largest x and y values of points in the mesh and return them as a vector in the format [[x_min, x_max], [y_min, y_max]]
def extremal_coordinates(mesh):

    points = boundary_points(mesh)
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



'''
compute the difference between functions f and g on the boundary of the mesh on which f and g are defined, returning 
sqrt(\sum_{i \in {vertices in the boundary of the mesh} [f(x_i) - g(x_i)]^2/ (number of vertices in the boundary of the mesh})
'''
def difference_on_boundary(f, g):

    mesh = f.function_space().mesh()
    boundary_points_mesh = boundary_points( mesh )

    # print("\n\nx\tf(x)-g(x)")
    diff = 0.0
    for x in boundary_points_mesh:
        delta = f( x ) - g( x )
        diff += (delta ** 2)

    diff = np.sqrt( diff / len( boundary_points_mesh ) )

    return diff

'''
compute the difference between functions f and g in the bulk of the mesh on which f and g are defined, returning 
sqrt(\sum_{i \in {vertices in the bulk of the mesh} [f(x_i) - g(x_i)]^2/ (number of vertices in the bulk of the mesh})
'''
def difference_in_bulk(f, g):

    mesh = f.function_space().mesh()
    bulk_points_mesh = bulk_points( mesh )

    diff = 0.0
    for x in bulk_points_mesh:
        delta = f( x ) - g( x )
        diff += (delta ** 2)

    diff = np.sqrt( diff / len( bulk_points_mesh ) )

    return diff

# return sqrt(<(f-g)^2>_measure / <measure>), where measure can be dx, ds_...
def difference_wrt_measure(f, g, measure):
    return sqrt(assemble( ( ( f - g ) ** 2 * measure ) ) / assemble(Constant(1.0) * measure))

# return sqrt(<f^2>_measure / <measure>), where measure can be dx, ds_...
def abs_wrt_measure(f, measure):
    return difference_wrt_measure(f, Constant(0), measure)

'''
compute the difference between functions f and g on the boundary of the mesh, boundary_c, given by the boundary points whose distance from point c lies between r and R, returning 
sqrt(\sum_{i \in {vertices in boundary_c} [f(x_i) - g(x_i)]^2/ (number of vertices in boundary_c})
'''
def difference_on_boundary_circle(f, g, r, R, c):

    mesh = f.function_space().mesh()
    boundary_c_points = boundary_points_circle( mesh, r, R, c )

    diff = 0.0
    for x in boundary_c_points:
        delta = f( x ) - g( x )
        diff += (delta ** 2)

    diff = np.sqrt( diff / len( boundary_c_points ) )

    return diff


'''
write to csv file 'outfile' the coordinates of the start and end vertices which define the lines of the triangles of the mesh in the .msh file 'infile'
the vertices are written in the format
edge1_start[0], edge1_start[1], edge1_start[2], edge1_end[0], edge1_end[1], edge1_end[2]
edge2_start[0], edge2_start[1], edge2_start[2], edge2_end[0], edge2_end[1], edge2_end[2]
...
'''
def write_mesh_to_csv(infile, outfile):
    #open the .msh file
    gmsh.open( infile )

    #get the list of components with dimension 2 from the mesh (triangles)
    triangles = gmsh.model.mesh.getElements( dim=2 )
    # print( "triangles = ", triangles )

    # construct a map which, given the tag of a node, gives its coordinates
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_map = {node_tags[i]: node_coords[3 * i: 3 * (i + 1)] for i in range( len( node_tags ) )}
    # print( "node map = ", node_map )

    # Store unique edges from the triangle elements
    # initialize a 'list' of unique elements, this sets the list to empty
    edges = set()

    #loop over all triangle nodes
    triangle_nodes = triangles[2][0] if len( triangles[2] ) > 0 else []
    for i in range( 0, len( triangle_nodes ), 3 ):
        #store into pair_12 = [ID_1, ID_2] the IDs of the vertices which lie at the extremities of the line in the triangle, and similarly for pair_23, pair_31
        pair_12 = tuple( sorted( [triangle_nodes[i], triangle_nodes[i + 1]] ) )
        pair_23 = tuple( sorted( [triangle_nodes[i + 1], triangle_nodes[i + 2]] ) )
        pair_31 = tuple( sorted( [triangle_nodes[i + 2], triangle_nodes[i]] ) )

        # this pushes back the elements pair_12, pair_23, pair_31 to edges
        edges.update( [pair_12, pair_23, pair_31] )
        # print( f"pair_12 = {pair_12} pair_23 = {pair_23} pair_31 = {pair_31}" )


    # loop through the edges added before and write the endoints of their lines to file
    csvfile = open( outfile, "w" )
    print( f"\"start:0\",\"start:1\",\"start:2\",\"end:0\",\"end:1\",\"end:2\"", file=csvfile )
    for edge in edges:
        # apply node_map to obtain the coordinates of the starting vertex in edge from their IDs, and similarly for p_end
        p_start = node_map[edge[0]]
        p_end = node_map[edge[1]]
        # print( f"\tEdge from {edge[0]} to {edge[1]}: p_start = ({p_start[0]}, {p_start[1]}, {p_start[2]}), "p_end = ({p_end[0]}, {p_end[1]}, {p_end[2]})" )
        print( f"{p_start[0]}, {p_start[1]}, {p_start[2]},{p_end[0]}, {p_end[1]}, {p_end[2]}", file=csvfile )

    csvfile.close()