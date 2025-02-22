from fenics import *

number_of_decimals = 2



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


#print to the csv file 'filename' the coordinates of the vertices of 'mesh'
def print_vertices_to_csv_file(mesh, filename):

    # a dummy function space of order 1 used to tabulated the vertices
    Q = FunctionSpace( mesh, 'CG', 1 )
    coordinates = Q.tabulate_dof_coordinates()

    csvfile = open( filename, "w" )
    print( f"\":0\",\":1\"", file=csvfile )

    for i in range( Q.dim() ):
        print( f"{coordinates[i][0]}, {coordinates[i][1]}", file=csvfile )

    csvfile.close()




