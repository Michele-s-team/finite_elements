import argparse

parser = argparse.ArgumentParser()
parser.add_argument("problem")
# the directory where to read the mesh
parser.add_argument("input_directory")
# the directory where to read the input solution
parser.add_argument("solution_input_directory")
# the directory where to write the outuput solution
parser.add_argument("output_directory")
parser.add_argument("N")
parser.add_argument("i")
args = parser.parse_args()
