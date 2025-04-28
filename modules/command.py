from symbol import return_stmt

import colorama as col
import os
import subprocess


'''
checks out a commit
Input values: 
- 'commit_sha': the sha of the commit 
'''
def checkout(commit_sha):

    print(f'{col.Fore.BLUE}Checking out {commit_sha}... {col.Fore.RESET}')
    os.system(f'git checkout {commit_sha}')
    print(f'{col.Fore.BLUE}...done.{col.Fore.RESET}')


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
Return value: 
- the strings with the output and the error resulting from the command run
'''
def run_command(command):
    result = subprocess.run(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True  # <- instead of text=True
    )

    return result.stdout, result.stderr


def command_empty_err_out(command):
    output_out, output_err = run_command(command)
    out_is_empty = (output_out.strip() == "")
    err_is_empty = (output_err.strip() == "")
    result = (out_is_empty and err_is_empty)

    return result
