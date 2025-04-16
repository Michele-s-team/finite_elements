import input_output as io
import runtime_arguments as rarg

xdmf_file_path = io.add_trailing_slash(rarg.args.output_directory)
csv_files_path = io.add_trailing_slash(rarg.args.output_directory)
nodal_values_path = io.add_trailing_slash(csv_files_path + 'nodal_values')