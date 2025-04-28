import colorama as col
import os


'''
checks out a commit
Input values: 
- 'commit_sha': the sha of the commit 
'''
def checkout(commit_sha):

    print(f'{col.Fore.CYAN}Checking out {commit_sha}... {col.Fore.RESET}')
    os.system(f'git checkout {commit_sha}')
    print(f'{col.Fore.CYAN}...done.{col.Fore.RESET}')


'''
goes to a given path
Input values:
- 'path': the path 
'''
def go_to_path(path):
    print(f'{col.Fore.CYAN}Entering {path}... {col.Fore.RESET}')
    os.system(f'cd {path}')
    print(f'{col.Fore.CYAN}...done.{col.Fore.RESET}')