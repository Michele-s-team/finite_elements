from fenics import *
import numpy as np
import colorama as col

import geometry as geo
import input_output as io


# read the mesh form 'filename' and write it into 'mesh'
def read_mesh(mesh, filename):
    xdmf = XDMFFile( mesh.mpi_comm(), filename )
    xdmf.read( mesh )


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