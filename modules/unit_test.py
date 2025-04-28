import colorama as col
import os


'''
checks out a commit
Input values: 
- 'commit_sha': the sha of the commit 
'''
def checkout(commit_sha):

    print(f'{col.Fore.CYAN}Switching to {commit_sha}... {col.Fore.RESET}')
    os.system(f'git checkout {commit_sha}')
    print(f'{col.Fore.CYAN}...done.{col.Fore.RESET}')