from fenics import *
from mshr import *

def print_to_csvfile(f, filename):
    csvfile = open(filename, "w")
    print( f"\"f\",\":0\",\":1\",\":2\"", file=csvfile )
    for x, val in zip(f.function_space().tabulate_dof_coordinates(), f.vector().get_local()):
        print(f"{val},{x[0]},{x[1]},{0}", file=csvfile)
    csvfile.close()