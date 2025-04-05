import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input_directory")
parser.add_argument("output_directory")
parser.add_argument("T")
parser.add_argument("k")
parser.add_argument("r")
parser.add_argument("e")
parser.add_argument("v")
parser.add_argument("N")
args = parser.parse_args()