from mm_rom import invariance_pde as iv
import tabulate

# set parameters for the problem to be solved
lims = [-1,1]
d = 2
n_state = 2
alpha = 2

max_deg = 2
solver2 = iv.low_dim_solver(lims, d, max_deg, alpha)
solver2.iterate_solution(transparent=True)
solver2.save_file()
solver2.visualize_result_overall(show=False, save=True)
print(solver2.pi_coefficients)
print(solver2.pi_evaluation)

max_deg = 4
solver4 = iv.low_dim_solver(lims, d, max_deg, alpha)
solver4.iterate_solution(transparent=True)
solver4.save_file()
solver4.visualize_result_overall(show=False, save=True)
print(solver4.pi_coefficients)
print(solver4.pi_evaluation)

max_deg = 6
solver6 = iv.low_dim_solver(lims, d, max_deg, alpha)
solver6.iterate_solution(transparent=True)
solver6.save_file()
solver6.visualize_result_overall(show=False, save=True)
print(solver6.pi_coefficients)
print(solver6.pi_evaluation)

print('\n\n')
print('Table of the average residual error for the Galerkin approximation of the invariant mapping for Test 1.')
table = [["Residual Error", solver2.pi_evaluation, solver4.pi_evaluation, solver6.pi_evaluation]]
print(tabulate.tabulate(table, headers=["","M = 2", "M = 4", "M = 6"]))


