import argparse

parser = argparse.ArgumentParser()
parser.add_argument("problem")
parser.add_argument("mesh_old_directory")
parser.add_argument("input_directory")
parser.add_argument("output_directory")
parser.add_argument("N")
parser.add_argument("i")
args = parser.parse_args()
