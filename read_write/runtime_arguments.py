import argparse

parser = argparse.ArgumentParser()
parser.add_argument("mesh_old_directory")
parser.add_argument("solution_old_directory")
parser.add_argument("solution_new_directory")
parser.add_argument("N")
parser.add_argument("i")
args = parser.parse_args()
