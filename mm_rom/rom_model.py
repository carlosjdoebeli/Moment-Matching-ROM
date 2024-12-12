from mm_rom import invariance_pde as iv
import numpy as np
from scipy.sparse import diags
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
import os

class rom_model:
    '''
    Class for producing a reduced-order model as described by the paper "A POLYNOMIAL APPROXIMATION SCHEME FOR NONLINEAR MODEL REDUCTION BY MOMENT MATCHING" by Carlos Doebeli, Allesandro Astolfi, Dante Kalise, Alessio Moreschini, Giordano Scarciotti, and Joel Simard. 

    Given a solver from the module invariance_pde, that has already found the coefficients for the approximate invariant mapping pi, or has loaded it from a file, this class creates an object for creating the reduced-order model for the high-dimensional problems considered in the paper.
    '''
    def __init__(self, solver: iv.solver, omega_0, x_0, r_0, t_max):
        '''
        Initializes the reduced order mdoel values. 

        Parameters:
            - solver: a solver object from invariance_pde, that has already found the coefficients for pi.
            - omega_0: the initial conditions for the signal generator.
            - x_0: the initial conditions for the full-order model.
            - r_0: the initial conditions for the reduced-order model.
            - t_max: the maximum time for the simulation
        '''
        self.solver = solver
        self.omega_0 = omega_0
        self.x_0 = x_0
        self.r_0 = r_0
        self.t_max = t_max

        self.rms = None
        self.rms_relative = None

    def combined_fom(self, t, state):
        '''
        A function that returns the time derivative of the combined state (omega_0, x_0) for the full-order model. 

        Parameters:
            - t: the current time
            - state: the current state consisting of (omega_0, x_0)
        '''
        omega = state[0:self.solver.d]
        x = state[self.solver.d:]
        u = self.ell(omega)
        
        s = self.s(omega)
        f = self.f_fom(x,u)
        
        return np.concatenate((s,f))

    def combined_rom(self, t, state):
        '''
        A function that returns the time derivative of the combined state (omega_0, r_0) for the reduced-order model. 

        Parameters:
            - t: the current time
            - state: the current state consisting of (omega_0, r_0)
        '''
        omega = state[0:self.solver.d]
        x = state[self.solver.d:]
        u = self.ell(omega)
        
        s = self.s(omega)
        f = self.f_rom(x,u)
        
        return np.concatenate((s,f))

    def combined_both(self, t, state):
        '''
        A function that returns the time derivative of the combined state (omega_0, r_0, x_0) for the interconnected system with both the reduced-order model and the full-order model. 

        Parameters:
            - t: the current time
            - state: the current state consisting of (omega_0, r_0, x_0)
        '''
        omega = state[0:self.solver.d]
        x_rom = state[self.solver.d:2*self.solver.d]
        x_fom = state[2*self.solver.d:]
        
        u = self.ell(omega)
        
        s = self.s(omega)
        f_rom = self.f_rom(x_rom, u)
        f_fom = self.f_fom(x_fom, u)
        
        return np.concatenate((s,f_rom,f_fom))
    
    def plot_rom(self, omega_0=None, r_0=None, t_max=None, int_points=30000, show=True, save=False):
        '''
        Plots the time evolution of the output of the reduced-order model. 

        Parameters:
            - omega_0: the initial condition for the signal generator. If it is None, then it takes the initialized value of self.omega_0.
            - r_0: the initial condition for the reduced-order model. If it is None, then it takes the initialized value of self.r_0.
            - t_max: the maximum time for the simulation. If it is None, then it takes the initialized value of self.t_max. 
            - int_points: the number of points to take when carrying out the numerical solution of the ODE.
            - show: whether or not to show the plot.
            - save: whether or not to save the plot to a file.
        '''
        if omega_0 is None:
            omega_0 = self.omega_0
        if r_0 is None:
            r_0 = self.r_0
        if t_max is None:
            t_max = self.t_max

        t_span = (0.0, self.t_max)

        state_0 = np.concatenate((omega_0, r_0))

        result_solve_ivp = solve_ivp(self.combined_rom, t_span, state_0, dense_output=True)

        t = np.linspace(0, t_max, int_points)    
        z = result_solve_ivp.sol(t)

        r1 = z[2,:]
        r2 = z[3,:]
        rom_output = self.h(self.solver.get_pi(self.solver.pi_coefficients, r1, r2))

        fig, axs = plt.subplots(1,1,figsize=(12,6))

        axs.plot(t, rom_output)
        axs.set_title("plot of $y_r(t)$ vs. $t$ for the reduced order model for the " + self.solver.string_code + " problem ""\n over a domain of " + str(self.solver.limsx) + " x " + str(self.solver.limsy) + " with a maximum degree of " + str(int(self.solver.max_deg)))
        axs.set_xlabel("$t$", fontsize=30)
        axs.set_ylabel("$y_r(t)$", fontsize=30)

        if show:
            plt.show()
        
        if save:
            self.solver.get_file_name()
            dir_name = self.folder_name + 'plot_rom/'
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            plt.savefig(dir_name + self.solver.file_name + '_rom.png')
        
        plt.close()

    def plot_fom(self, omega_0=None, x_0=None, t_max=None, int_points=30000, show=True, save=False):
        '''
        Plots the time evolution of the output of the full-order model. 

        Parameters:
            - omega_0: the initial condition for the signal generator. If it is None, then it takes the initialized value of self.omega_0.
            - x_0: the initial condition for the full-order model. If it is None, then it takes the initialized value of self.x_0.
            - t_max: the maximum time for the simulation. If it is None, then it takes the initialized value of self.t_max. 
            - int_points: the number of points to take when carrying out the numerical solution of the ODE.
            - show: whether or not to show the plot.
            - save: whether or not to save the plot to a file.
        '''
        if omega_0 is None:
            omega_0 = self.omega_0
        if x_0 is None:
            x_0 = self.x_0
        if t_max is None:
            t_max = self.t_max

        t_span = (0.0, self.t_max)

        state_0 = np.concatenate((omega_0, x_0)) 

        result_solve_ivp = solve_ivp(self.combined_fom, t_span, state_0, dense_output=True)

        t = np.linspace(0, t_max, int_points)    
        z = result_solve_ivp.sol(t)

        x_fom = z[2:,:]
        fom_output = self.h(x_fom)

        fig, axs = plt.subplots(1,1,figsize=(12,6))

        axs.plot(t, fom_output)
        axs.set_title("plot of $y(t)$ vs. $t$ for the full order model for the " + self.solver.string_code + " problem ""\n over a domain of " + str(self.solver.limsx) + " x " + str(self.solver.limsy) + " with a maximum degree of " + str(int(self.solver.max_deg)))
        axs.set_xlabel("$t$", fontsize=30)
        axs.set_ylabel("$y(t)$", fontsize=30)

        if show:
            plt.show()
        
        if save:
            self.solver.get_file_name()
            dir_name = self.folder_name + 'plot_fom/'
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            plt.savefig(dir_name + self.solver.file_name + '_fom.png')
        
        plt.close()

    def verify_rom(self, omega_0=None, r_0=None, x_0=None, t_max=None, t_test=None, int_points=30000, log=True, show=True, save=False):
        '''
        Plots the time evolution of the output of both the reduced-order model and the full-order model, and also plots the difference between them. 

        Parameters:
            - omega_0: the initial condition for the signal generator. If it is None, then it takes the initialized value of self.omega_0.
            - r_0: the initial condition for the reduced-order model. If it is None, then it takes the initialized value of self.r_0.
            - x_0: the initial condition for the full-order model. If it is None, then it takes the initialized value of self.x_0.
            - t_max: the maximum time for the simulation. If it is None, then it takes the initialized value of self.t_max. 
            - t_test: the time after which the steady-state error will be measured. If set to None, it will be automatically set to 0.5 * t_max.
            - int_points: the number of points to take when carrying out the numerical solution of the ODE.
            - log: whether to use a logarithmic axis scale for the error plot.
            - show: whether or not to show the plot.
            - save: whether or not to save the plot to a file.
        '''
        if omega_0 is None:
            omega_0 = self.omega_0
        if r_0 is None:
            r_0 = self.r_0
        if x_0 is None:
            x_0 = self.x_0
        if t_max is None:
            t_max = self.t_max
        if t_test is None:
            t_test = 0.5 * self.t_max

        t_span = (0.0, self.t_max)

        state_0 = np.concatenate((self.omega_0, self.r_0, self.x_0))

        result_solve_ivp = solve_ivp(self.combined_both, t_span, state_0, dense_output=True)

        t = np.linspace(0, t_max, int_points)    
        z = result_solve_ivp.sol(t)
        
        r1 = z[2,:]
        r2 = z[3,:]

        x_fom = z[4:,:]

        rom_output = self.h(self.solver.get_pi(self.solver.pi_coefficients, r1, r2))
        fom_output = self.h(x_fom)
        output_error = rom_output - fom_output

        ss_index = np.argmax(t > t_test)

        rms = np.sqrt(np.mean(output_error[ss_index:]**2))
        amplitude = (np.max(fom_output[ss_index:]) - np.min(fom_output[ss_index:]))/2

        self.rms = rms
        self.rms_relative = rms / amplitude

        # print("rms: ", self.rms)
        # print("relative rms: ", self.rms_relative)

        fig, axs = plt.subplots(2,1,figsize=(12,8), gridspec_kw={'height_ratios': [1, 1]})
        fig.subplots_adjust(hspace=0.8)

        axs[0].plot(t, fom_output, label='$y(t)$')
        axs[0].plot(t, rom_output, linestyle='dashed', label='$y_r(t)$')
        axs[0].set_title("plot of $y(t)$ vs. $t$ and $y_r(t)$ vs. $t$ for the " + self.solver.string_code + " problem ""\n over a domain of " + str(self.solver.limsx) + " x " + str(self.solver.limsy) + " with a maximum degree of " + str(int(self.solver.max_deg)))
        axs[0].set_xlabel("$t$", fontsize=30)
        axs[0].set_ylabel("signal", fontsize=30)
        axs[0].legend()

        axs[1].plot(t,
            np.abs(rom_output - fom_output))
        axs[1].set_title("plot of $\\log |y(t) - y_r(t)|$ vs. $t$ for the " + self.solver.string_code + " problem ""\n over a domain of " + str(self.solver.limsx) + " x " + str(self.solver.limsy) + " with a maximum degree of " + str(int(self.solver.max_deg)))
        axs[1].set_xlabel("$t$", fontsize=30)
        axs[1].set_ylabel("signal error", fontsize=30)

        if log:
            axs[1].set_yscale('log')
        
        if show:
            plt.show()
        
        if save:
            self.solver.get_file_name()
            dir_name = self.folder_name + 'verify_rom/'
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            plt.savefig(dir_name + self.solver.file_name + '_verify.png')
        
        plt.close()   


class rom_rl_linear(rom_model):
    '''
    Subclass of rom_model corresponding to Test 3 in the paper, for a resistor-inductor ladder problem with a linear oscillator as a signal generator. 
    '''
    def __init__(self, solver, omega_0, x_0, r_0, t_max, const=10):
        '''
        RL ladder linear oscillator ROM constructor. If you are saving to files or reading from files, change the path to where your figures are stored in 'self.folder_name'.

        Parameters:
            - solver: a solver object from invariance_pde, that has already found the coefficients for pi.
            - omega_0: the initial conditions for the signal generator.
            - x_0: the initial conditions for the full-order model.
            - r_0: the initial conditions for the reduced-order model.
            - t_max: the maximum time for the simulation
            - const: a constant used to create the reduced-order model. For the RL ladder linear oscillator problem, it is set to 10 by default.
        '''
        super().__init__(solver, omega_0, x_0, r_0, t_max)

        self.const = const
        self.folder_name = 'rom_folder/rl_linear/'

    def s(self, omega):
        '''
        Function that returns the value of s(omega) for the signal generator.

        Parameters:
            - omega: the signal generator state

        Returns:
            - s(omega) for the linear oscillator
        '''
        s = np.array([self.solver.alpha * omega[1], -self.solver.alpha * omega[0]])
        return s
    
    def ell(self, omega):
        '''
        Returns the output ell(omega) for the signal generator.

        Parameters:
            - omega: the signal generator state

        Returns: 
            - ell(omega) = omega_2 for this problem
        '''
        return omega[1]
    
    def f_fom(self, x, u):
        '''
        Returns the function f(x,u) determining the dynamics of the full order model. 

        Parameters:
            - x: the full-order model's state
            - u: the control input
        
        Returns: 
            f(x,u) for the RL ladder problem.
        '''
        k = [np.ones(self.solver.n_state - 1), -2 * self.solver.kappa * np.ones(self.solver.n_state), np.ones(self.solver.n_state - 1)]
        offset = [-1,0,1]
        A_mat = diags(k,offset).toarray()
        gamma_vec = np.zeros(self.solver.n_state)
        gamma_vec[0] = 1
        
        return np.matmul(A_mat, x) - np.power(x, 2) / 2 - np.power(x, 3) / 3 + gamma_vec * u
    
    def h(self, x):
        '''
        Returns the function h(x) that defines the output of the full-order model.

        Parameters:
            - x: the full-order model's state
        
        Returns: 
            h(x) = x_0, the output of the RL ladder model.
        '''
        return x[0]

    def f_rom(self, r, u):
        '''
        Returns the function \\bar{f}(x,u) determining the dynamics of the reduced-order model.

        Parameters:
            - r: the reduced-order model's state
            - u: the control input
        
        Returns: 
            \\bar{f}(r,u) for the reduced-order model.
        '''
        return self.s(r) - self.g(r) * self.ell(r) + self.g(r) * u
    
    def g(self, r):
        '''
        Returns the function \\bar{g}(r) determining the dynamics of the reduced-order model. This is used in computing \\bar{f}.

        Parameters:
            - r: the reduced-order model's state

        Returns: 
            - \\bar{g}(r) for the RL ladder with the linear oscillator signal generator
        '''
        return np.array([0,self.const])

class rom_rl_vdp(rom_model):
    '''
    Subclass of rom_model corresponding to Test 4 in the paper, for a resistor-inductor ladder problem with a Van der Pol oscillator as a signal generator. 
    '''
    def __init__(self, solver, omega_0, x_0, r_0, t_max, const=2):
        '''
        RL ladder Van der Pol oscillator ROM constructor. If you are saving to files or reading from files, change the path to where your figures are stored in 'self.folder_name'.

        Parameters:
            - solver: a solver object from invariance_pde, that has already found the coefficients for pi.
            - omega_0: the initial conditions for the signal generator.
            - x_0: the initial conditions for the full-order model.
            - r_0: the initial conditions for the reduced-order model.
            - t_max: the maximum time for the simulation
            - const: a constant used to create the reduced-order model. For the RL ladder linear oscillator problem, it is set to 2 by default.
        '''
        super().__init__(solver, omega_0, x_0, r_0, t_max)

        self.const = const
        self.folder_name = 'rom_folder/rl_vdp/'

    def s(self, omega):
        '''
        Function that returns the value of s(omega) for the signal generator.

        Parameters:
            - omega: the signal generator state

        Returns:
            - s(omega) for the Van der Pol oscillator
        '''
        s = self.solver.alpha * np.array([omega[1],-omega[0] + self.solver.mu * (1 -  omega[0]**2) * omega[1]])
        return s
    
    def ell(self, omega):
        '''
        Returns the output ell(omega) for the signal generator.

        Parameters:
            - omega: the signal generator state

        Returns: 
            - ell(omega) = omega_2 for this problem
        '''
        return omega[1]

    def f_fom(self, x, u):
        '''
        Returns the function f(x,u) determining the dynamics of the full order model. 

        Parameters:
            - x: the full-order model's state
            - u: the control input
        
        Returns: 
            f(x,u) for the RL ladder problem.
        '''
        k = [np.ones(self.solver.n_state - 1), -2 * self.solver.kappa * np.ones(self.solver.n_state), np.ones(self.solver.n_state - 1)]
        offset = [-1,0,1]
        A_mat = diags(k,offset).toarray()
        gamma_vec = np.zeros(self.solver.n_state)
        gamma_vec[0] = 1
        
        return np.matmul(A_mat, x) - np.power(x, 2) / 2 - np.power(x, 3) / 3 + gamma_vec * u
    
    def h(self, x):
        '''
        Returns the function h(x) that defines the output of the full-order model.

        Parameters:
            - x: the full-order model's state
        
        Returns: 
            h(x) = x_0, the output of the RL ladder model.
        '''
        return x[0]
    
    def f_rom(self, r, u):
        '''
        Returns the function \\bar{f}(x,u) determining the dynamics of the reduced-order model.

        Parameters:
            - r: the reduced-order model's state
            - u: the control input
        
        Returns: 
            \\bar{f}(r,u) for the reduced-order model.
        '''
        return self.s(r) - self.g(r) * self.ell(r) + self.g(r) * u
    
    def g(self, r):
        '''
        Returns the function \\bar{g}(r) determining the dynamics of the reduced-order model. This is used in computing \\bar{f}.

        Parameters:
            - r: the reduced-order model's state

        Returns: 
            - \\bar{g}(r) for the RL ladder with the Van der Pol oscillator signal generator
        '''
        return np.array([0, self.solver.mu*(1-r[0]**2) + self.const])