from mm_rom import invariance_pde as iv
from mm_rom import rom_model as rm
import numpy as np
import tabulate

# set parameters for the problem to be solved
d = 2
alpha = 1
kappa = 1.1
mu = 0.25

domain = 0.7

lims_array = [[-1,1], [-2,2], [-3,3]]
max_deg_array = [2,4,6]

omega_0 = np.array([0.2,0.2])
r_0 = np.array([0.,1])
t_max = 50

n_state = 2
evals_2 = np.zeros((len(lims_array), len(max_deg_array)))

for max_deg_index in range(len(max_deg_array)):
    for lims_index in range(len(lims_array)):
        max_deg = max_deg_array[max_deg_index]
        lims = lims_array[lims_index]

        solver = iv.rl_vdp_solver(lims, d, max_deg, n_state, alpha, kappa, mu, from_file=True)
        solver.evaluate_solution(domain=domain)

        x_0 = np.zeros(solver.n_state)
        rom = rm.rom_rl_vdp(solver, omega_0, x_0, r_0, t_max)
        rom.verify_rom(show=False)
        evals_2[lims_index][max_deg_index] = rom.rms_relative

n_state = 100
evals_100 = np.zeros((len(lims_array), len(max_deg_array)))

for max_deg_index in range(len(max_deg_array)):
    for lims_index in range(len(lims_array)):
        max_deg = max_deg_array[max_deg_index]
        lims = lims_array[lims_index]

        solver = iv.rl_linear_solver(lims, d, max_deg, n_state, alpha, kappa, mu, from_file=True)
        solver.evaluate_solution(domain=domain)

        x_0 = np.zeros(solver.n_state)
        rom = rm.rom_rl_linear(solver, omega_0, x_0, r_0, t_max)
        rom.verify_rom(show=False)
        evals_100[lims_index][max_deg_index] = rom.rms_relative

n_state = 1000
evals_1000 = np.zeros((len(lims_array), len(max_deg_array)))

for max_deg_index in range(len(max_deg_array)):
    for lims_index in range(len(lims_array)):
        max_deg = max_deg_array[max_deg_index]
        lims = lims_array[lims_index]

        solver = iv.rl_linear_solver(lims, d, max_deg, n_state, alpha, kappa, from_file=True)
        solver.evaluate_solution(domain=domain)

        x_0 = np.zeros(solver.n_state)
        rom = rm.rom_rl_linear(solver, omega_0, x_0, r_0, t_max)
        rom.verify_rom(show=False)
        evals_1000[lims_index][max_deg_index] = rom.rms_relative


print()
print('Table of relative RMS error for n = 2')
table_headers = ['Omega = ' + str(lims_array[0]) + '^2','Omega = ' + str(lims_array[1]) + '^2','Omega = ' + str(lims_array[2]) + '^2']
table = np.c_[table_headers,evals_2]
print(tabulate.tabulate(table, headers=["","M = 2", "M = 4", "M = 6"]))

print()
print('Table of relative RMS error for n = 100')
table_headers = ['Omega = ' + str(lims_array[0]) + '^2','Omega = ' + str(lims_array[1]) + '^2','Omega = ' + str(lims_array[2]) + '^2']
table = np.c_[table_headers,evals_100]
print(tabulate.tabulate(table, headers=["","M = 2", "M = 4", "M = 6"]))

print()
print('Table of relative RMS error for n = 1000')
table_headers = ['Omega = ' + str(lims_array[0]) + '^2','Omega = ' + str(lims_array[1]) + '^2','Omega = ' + str(lims_array[2]) + '^2']
table = np.c_[table_headers,evals_1000]
print(tabulate.tabulate(table, headers=["","M = 2", "M = 4", "M = 6"]))