from symbol import return_stmt

import colorama as col
import os
import subprocess


'''
checks out a commit
Input values: 
- 'commit_sha': the sha of the commit 
- 'success': A list with one entry: If it is True the checkout will be done, if not the checkout will not be done. If the checkout is successful success[0] will be set to True and to False otherwise
'''
def checkout(commit_sha, success):

    if(success[0]):

        print(f'{col.Fore.BLUE}Checking out {commit_sha}... {col.Fore.RESET}')
        run_command(f'git checkout {commit_sha}', success)
        print(f'{col.Fore.BLUE}...done.{col.Fore.RESET}')

    else:
        print('Stopping here.')


'''
goes to a given path
Input values:
- 'path': the path 
'''
def go_to_path(path):
    print(f'{col.Fore.BLUE}Entering {path}... {col.Fore.RESET}')
    os.system(f'cd {path}')
    print(f'{col.Fore.BLUE}...done.{col.Fore.RESET}')

'''
Run a command in command line
Input values: 
- 'command' the command, e.g. 'pwd'
- 'success': A list with one entry: if it is True (False), the command will be (not) executed. If the command execution is successful, success[0] will be set to True and False otherwise. 
Return value: 
- the strings with the output and the error resulting from the command run
'''
def run_command(command, success):

    if(success[0]):

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
        return '',''


def command_empty_err_out(command):
    success, output_out, output_err = run_command(command)
    out_is_empty = (output_out.strip() == "")
    err_is_empty = (output_err.strip() == "")
    result = (out_is_empty and err_is_empty)

    return result
