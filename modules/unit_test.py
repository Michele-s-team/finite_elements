import os

import command as cmd
import input_output as io


def test_problem_and_mesh(commit_a,
                          commit_b,
                          root_path, mesh_path, code_path,
                          mesh_solution_path_a, problem_solution_path_a,
                          mesh_solution_path_b, problem_solution_path_b,
                          name_of_generate_mesh, mesh_resolution,
                          problem
                          ):
    # checkout commit_a, generate the mesh and solve the problem
    cmd.checkout(commit_a)

    os.system(f'cd {mesh_path}; rm -rf {mesh_solution_path_a}; mkdir -p {mesh_solution_path_a}; python3 {name_of_generate_mesh}.py {mesh_resolution} {mesh_solution_path_a}')
    os.system(f'cd {code_path}; rm -rf {problem_solution_path_a}; mkdir -p {problem_solution_path_a}; python3 solve.py {problem} {mesh_solution_path_a} {problem_solution_path_a}')

    # checkout commit_b, generate the mesh and solve the problem
    cmd.checkout(commit_b)
    os.system(f'cd {mesh_path}; rm -rf {mesh_solution_path_b}; mkdir -p {mesh_solution_path_b}; python3 {name_of_generate_mesh}.py {mesh_resolution} {mesh_solution_path_b}')
    os.system(f'cd {code_path}; rm -rf {problem_solution_path_b}; mkdir -p {problem_solution_path_b}; python3 solve.py {problem} {mesh_solution_path_b} {problem_solution_path_b}')

    # compare the mesh and problem solution for commit_a and commit_b
    mesh_check = cmd.command_empty_err_out(f'cd {root_path}; ./compare-csv-files.sh {mesh_solution_path_a} {mesh_solution_path_b}')
    problem_check = cmd.command_empty_err_out(f'cd {root_path}; ./compare-csv-files.sh {problem_solution_path_a} {problem_solution_path_b}')

    # if check = true, then commit_a and commit_b give the same result
    check = (mesh_check and problem_check)

    io.check_print(mesh_check, f'Mesh check = {mesh_check}')
    io.check_print(problem_check, f'Problem check = {problem_check}')

    return check
