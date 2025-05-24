import argparse

parser = argparse.ArgumentParser()
parser.add_argument("mesh_old_directory")
parser.add_argument("path_solution_in")
parser.add_argument("path_solution_out")
parser.add_argument("N")
parser.add_argument("i")
args = parser.parse_args()
