from fenics import *
import numpy as np


#  norm of vector x
def my_norm(x):
    return (sqrt( np.dot( x, x ) ))

class BoundaryMarker( SubDomain ):
    def inside(self, x, on_boundary):
        return on_boundary

# returns a list of boundary points of the mesh `mesh`
def boundary_points(mesh):

    # create a dummy function space of degree 1 which will be used only to extract the boundary points
    Q_dummy = FunctionSpace( mesh, 'CG', 1 )

    # a map which takes as an input a vertex of Q_dummy.mesh and returns its corresponding degree of freedom
    vertex_to_degree_of_freedom_map = vertex_to_dof_map( Q_dummy )

    # a function which takes as argument the mesh vertices
    vertex_function = MeshFunction( "size_t", mesh, 0 )

    # set vertex_function -> 1 on the vertices which are part of the boundary (vertex_function is zero elsewhere)
    vertex_function.set_all(0)
    BoundaryMarker().mark( vertex_function, 1 )

    # collect the vertices where the vertex_function = 1, i.e., the vertices on the boundary
    boundary_vertices = np.asarray( vertex_function.where_equal( 1 ) )

    degrees_of_freedom = vertex_to_degree_of_freedom_map[boundary_vertices]

    x = Q_dummy.tabulate_dof_coordinates()
    x = x[degrees_of_freedom]

    csvfile = open( "test_boundary_points.csv", "w" )
    for p in x:
        print( f"{p[0]},{p[1]}", file=csvfile )
    csvfile.close()

    # print("Degrees of freedom on the boundary:")
    # for degree_of_freedom in degrees_of_freedom:
        # print(f"\t{x[degree_of_freedom]}, {my_norm( x[degree_of_freedom])}")

    return x


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
        diff += (delta**2)

    diff = np.sqrt(diff/len(boundary_points_mesh))

    # print(f"diff = {diff}")
    return diff