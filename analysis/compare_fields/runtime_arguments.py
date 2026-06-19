import argparse

parser = argparse.ArgumentParser()
parser.add_argument("problem")
parser.add_argument("input_directory")
parser.add_argument("solution_in_directory")
parser.add_argument("output_directory")
args = parser.parse_args()
