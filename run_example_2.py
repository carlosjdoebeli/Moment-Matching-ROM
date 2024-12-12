from mm_rom import invariance_pde as iv
import tabulate

# set parameters for the problem to be solved
lims = [-1,1]
d = 2
n_state = 4
a_1 = 2
a_2 = 3
k_const = -2/3


max_deg = 2
solver2 = iv.pendulum_solver(lims, d, max_deg, a_1, a_2, k_const)
solver2.iterate_solution(transparent=True)
solver2.save_file()
solver2.visualize_result_overall(show=False, save=True)

max_deg = 4
solver4 = iv.pendulum_solver(lims, d, max_deg, a_1, a_2, k_const)
solver4.iterate_solution(transparent=True)
solver4.save_file()
solver4.visualize_result_overall(show=False, save=True)

max_deg = 6
solver6 = iv.pendulum_solver(lims, d, max_deg, a_1, a_2, k_const)
solver6.iterate_solution(transparent=True)
solver6.save_file()
solver6.visualize_result_overall(show=False, save=True)


table = [["Residual Error", solver2.pi_evaluation, solver4.pi_evaluation, solver6.pi_evaluation]]
print(tabulate.tabulate(table, headers=["","M = 2", "M = 4", "M = 6"]))
















