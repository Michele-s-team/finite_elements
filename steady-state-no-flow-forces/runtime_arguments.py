import argparse

class rarg:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("input_directory")
        self.parser.add_argument("output_directory")
        self.args = self.parser.parse_args()