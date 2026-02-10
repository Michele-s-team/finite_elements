import sys

module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import runtime_arguments as rarg
import input_output as io


# add the path where to find the shared modules
xdmf_file_path = io.add_trailing_slash(rarg.args.output_directory)
h5_file_path = io.add_trailing_slash(rarg.args.output_directory) + 'h5/'
csv_files_path = io.add_trailing_slash(rarg.args.output_directory)
nodal_values_path = io.add_trailing_slash(csv_files_path + 'nodal_values')
