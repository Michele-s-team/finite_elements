from symbol import return_stmt

import colorama as col
import os
from pathlib import Path
import subprocess

'''
checks out a commit
Input values: 
- 'commit_sha': the sha of the commit 
- 'success': A list with one entry: If it is True the checkout will be done, if not the checkout will not be done. If the checkout is successful success[0] will be set to True and to False otherwise
'''


def checkout(commit_sha, success):
    if (success[0]):

        print(f'{col.Fore.BLUE}Checking out {commit_sha}... {col.Fore.RESET}')
        run_command(f'git checkout {commit_sha}', success)
        print(f'{col.Fore.BLUE}...done.{col.Fore.RESET}')

    else:
        print('Stopping here.')


'''
Run a command in command line
Input values: 
- 'command' the command, e.g. 'pwd'
- 'success': A list with one entry: if it is True (False), the command will be (not) executed. If the command execution is successful, success[0] will be set to True and False otherwise. 
Return value: 
- the strings with the output and the error resulting from the command run. If this method is called with success[0] = False, then it returns '', 'run_command failed'
'''


def run_command(command, success):
    if (success[0]):

        print(f'Running command {command} ...')

        result = subprocess.run(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True  # <- instead of text=True
        )

        success[0] = (result.returncode == 0)

        print('... done.')
        print(f'\tsuccess = {success}')
        print(f'\toutput = {result.stdout}')
        print(f'\terror = {result.stderr}')

        return result.stdout, result.stderr

    else:
        print('Stopping here.')

        return '', 'run_command failed'


def command_empty_err_out(command, success):
    if success[0]:
        output_out, output_err = run_command(command, success)
        out_is_empty = (output_out.strip() == "")
        err_is_empty = (output_err.strip() == "")
        result = (out_is_empty and err_is_empty)

    else:
        print('Stopping here.')
        result = False

    return result


'''
check if a file exists
Input values: 
- 'path': the path to the file
Return values: 
- 'True' ('False') if the file exists (does not exist)
'''


def check_if_file_exists(path):
    file_path = Path(path)
    return file_path.exists()


'''
Set multiple global variables in a module from a dictionary
Input values: 
    - 'target_module': the moduel where the global variables will be set
    - 'variable_dictionary': a dictionary with variable names as keys
'''


def set_global_variables(target_module, variables_dictionary):
    for variable_name, variable_value in variables_dictionary.items():
        setattr(target_module, variable_name, variable_value)


'''
set the global variables of a gauge for the manifold and boundary geometry
Input values: 
    - 'gauge' the name of the gauge, for example 'monge' or 'arc_length'
'''
def set_gauge(gauge):

    if gauge == 'monge':

        import differential_geometry.manifold.geometry as module_manifold_geometry_write
        import differential_geometry.boundary.geometry as module_boundary_geometry_write
        import differential_geometry.manifold.gauges.monge_gauge as module_manifold_geometry_read
        import differential_geometry.boundary.gauges.monge_gauge as module_boundary_geometry_read

        # list of methods for the manifold geometry
        methods_manifold_geometry = { \
            'e': module_manifold_geometry_read.e, \
            'K': module_manifold_geometry_read.K, \
            'normal': module_manifold_geometry_read.normal, \
            'X': module_manifold_geometry_read.X}

        # list of methods for the boundary geometry
        methods_boundary_geometry = { \
            'Nt_circle': module_boundary_geometry_read.Nt_circle, \
            'Nn_circle': module_boundary_geometry_read.Nn_circle, \
            'dydtheta': module_boundary_geometry_read.dydtheta, \
            'sqrt_deth_circle': module_boundary_geometry_read.sqrt_deth_circle, \
            'sqrt_deth_lr': module_boundary_geometry_read.sqrt_deth_lr, \
            'sqrt_deth_tb': module_boundary_geometry_read.sqrt_deth_tb, \
            'Nt_lr': module_boundary_geometry_read.Nt_lr, \
            'Nn_lr': module_boundary_geometry_read.Nn_lr, \
            'Nt_tb': module_boundary_geometry_read.Nt_tb, \
            'Nn_tb': module_boundary_geometry_read.Nn_tb, \
            'n_lr': module_boundary_geometry_read.n_lr, \
            'n_tb': module_boundary_geometry_read.n_tb, \
            'n_circle': module_boundary_geometry_read.n_circle \
            }

        # set the gauge-specific methods for the manifold and boundary geometry
        set_global_variables(module_manifold_geometry_write, methods_manifold_geometry)
        set_global_variables(module_boundary_geometry_write, methods_boundary_geometry)

    elif gauge == 'arc_length':

        import differential_geometry.manifold.geometry as module_manifold_geometry_write
        import differential_geometry.boundary.geometry as module_boundary_geometry_write
        import differential_geometry.manifold.gauges.arc_length_gauge as module_manifold_geometry_read
        import differential_geometry.boundary.gauges.arc_length_gauge as module_boundary_geometry_read

        methods_manifold_geometry = { \
            'e': module_manifold_geometry_read.e, \
            'normal': module_manifold_geometry_read.normal, \
            'K': module_manifold_geometry_read.K \
            }

        methods_boundary_geometry = { \
            'sqrt_deth_lr': module_boundary_geometry_read.sqrt_deth_lr, \
            'Nt_lr': module_boundary_geometry_read.Nt_lr, \
            'n_lr': module_boundary_geometry_read.n_lr \
            }

        # set the gauge-specific methods for the manifold and boundary geometry
        set_global_variables(module_manifold_geometry_write, methods_manifold_geometry)
        set_global_variables(module_boundary_geometry_write, methods_boundary_geometry)

