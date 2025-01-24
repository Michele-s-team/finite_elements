from fenics import *
from mshr import *
from dolfin import *
import numpy as np

#  norm of vector x
def my_norm(x):
    return (sqrt( np.dot( x, x ) ))

# this function prints a scalar field to csv file
def print_scalar_to_csvfile(f, filename):
    csvfile = open( filename, "w" )
    print( f"\"f\",\":0\",\":1\",\":2\"", file=csvfile )
    for x, val in zip( f.function_space().tabulate_dof_coordinates(), f.vector().get_local() ):
        print( f"{val},{x[0]},{x[1]},{0}", file=csvfile )
    csvfile.close()


# this function prints a vector field to csv file
def print_vector_to_csvfile(f, filename):
    i = 0
    list_val_x = []
    list_val_y = []
    list_x = []
    for x, val in zip( f.function_space().tabulate_dof_coordinates(), f.vector().get_local() ):
        if (i % 2 == 0):
            list_val_x.append( val )
            list_x.append( x )
        else:
            list_val_y.append( val )

        i += 1

    csvfile = open( filename, "w" )
    print( f"\"f:0\",\"f:1\",\"f:2\",\":0\",\":1\",\":2\"", file=csvfile )

    for x, val_x, val_y in zip( list_x, list_val_x, list_val_y ):
        print( f"{val_x},{val_y},{0},{x[0]},{x[1]},{0}", file=csvfile )

    csvfile.close()


def print_in_bulk(f):
    Q = f.function_space()
    mesh = Q.mesh()

    print( f"Function space: {Q}" )
    print( f"Dir(Q): {dir(Q)}" )
    print( f"Dir(Q.dofmap): {dir(Q.dofmap())}" )

    degrees_of_freedom_map = Q.dofmap()

    info( "DoF range owned by this process: %s" % str( degrees_of_freedom_map.ownership_range() ) )
    info( "DoFs block size: %s" % str( degrees_of_freedom_map.block_size() ) )
    info( "Vertex dofs: %s" % str( degrees_of_freedom_map.entity_dofs( mesh, 0 ) ) )
    info( "Length of vertex dofs: %s" % str( len(degrees_of_freedom_map.entity_dofs( mesh, 0 )) ) )
    info( "Facet dofs: %s" % str( degrees_of_freedom_map.entity_dofs( mesh, 1 ) ) )
    info( "Cell dofs: %s" % str( degrees_of_freedom_map.entity_dofs( mesh, 2 ) ) )
    info( "All DoFs (Vertex, Facet, and Cell) associated with cell 0: %s" % str( degrees_of_freedom_map.cell_dofs( 0 ) ) )
    info( "Local (process) to global DoF map: %s" % str( degrees_of_freedom_map.tabulate_local_to_global_dofs() ) )
    info( "******" )


    vertex_function = MeshFunction( "size_t", mesh, 0 )


    class BoundaryMarker( SubDomain ):
        def inside(self, x, on_boundary):
            return on_boundary


    BoundaryMarker().mark( vertex_function, 1 )
    boundary_vertices = np.asarray( vertex_function.where_equal( 1 ) )


    print("\nlength of boundary_vertices:", len(boundary_vertices))
    print("boundary_vertices:", boundary_vertices)

    my_vertex_to_dof_map = vertex_to_dof_map( Q )

    print("\nlength of my_vertex_to_dof_map:", len(my_vertex_to_dof_map))
    print("my_vertex_to_dof_map:", my_vertex_to_dof_map)

    dofs = my_vertex_to_dof_map[boundary_vertices]

    # x = Q.tabulate_dof_coordinates()
    # for dof in dofs:
    #     print( x[dof], f.vector()[dof], my_norm(x[dof]) )
