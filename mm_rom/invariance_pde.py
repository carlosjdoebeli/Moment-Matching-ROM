# import necessary python modules
import numpy as np
from scipy import integrate
from scipy import special
from scipy.sparse import diags
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import time
import itertools
import re
import csv
import os
import operator as op
from functools import reduce

from abc import ABC, abstractmethod

plt.style.use('seaborn-v0_8-poster')

class solver:
    '''
    Generic solver class for finding solutions to the Galerkin method as described by the paper "A POLYNOMIAL APPROXIMATION SCHEME FOR NONLINEAR MODEL REDUCTION BY MOMENT MATCHING" by Carlos Doebeli, Allesandro Astolfi, Dante Kalise, Alessio Moreschini, Giordano Scarciotti, and Joel Simard. 
    
    Creates a solver object for each problem that can find and evaluate the coefficients for the approximate solution pi to the invariance mapping. Each subclass corresponds to carrying out this method for a particular problem.
    '''
    def __init__(self, lims, d, max_deg, n_state, n_gauss=20, total_degree=True):
        '''
        Initializes the solver values and the gaussian quadrature points and weights. 

        Parameters:
            - lims: a 2D array of limits of integration, for example [-1,1]. Assumed to be symmetrical, and the same in each dimension of omega.
            - d: the dimensionality of the signal generator.
            - max_deg: the maximum degree of polynomial expansion.
            - n_state: the dimensionality of the system.
            - n_gauss: the number of quadrature points to take for numerical integration.
            - total_degree: whether or not we are using total degree of monomial expansions rather than full tensor product. Usually, this should be set to True.
        '''
        self.limsx = lims
        self.limsy = lims
        self.lims_arr = np.array([self.limsx, self.limsy])
        self.d = d
        self.max_deg = max_deg
        self.n_state = n_state
        self.total_degree = total_degree

        self.ndim = solver.monomial_dimension(self.d, self.max_deg)

        self.n_gauss = n_gauss
        self.gauss_points, self.gauss_weights = np.polynomial.legendre.leggauss(self.n_gauss)
        self.scaled_gauss_points = np.zeros((self.d, self.n_gauss))
        self.scaled_gauss_weights = np.zeros((self.d, self.n_gauss))
        self.initialize_gauss()

        # Boolean to indicate if we want to take the pseudoinverse. It is False in most cases except for the pendulum problem.
        self.pinv = False

        print("\n\nInitializing Solver: ")
        print("Omega: ", self.lims_arr[0])
        print("Maximum degree: ", self.max_deg)
        print("n: ", + self.n_state)
    
    def iterate_solution(self, pi_starting=None, iterations=100, tolerance=0.0000001, transparent=False):
        '''
        Carries out the Newton method given a starting guess. 

        Parameters:
            - pi_starting: the initial guess for the coefficients
            - iterations: maximum number of iterations
            - tolerance: the level of error before the iterations stop
            - transparent (bool): option to print out guesses and error at each iteration
        '''
        if pi_starting is None:
            pi_starting = np.zeros(self.n_state * self.ndim)

        pi_current = pi_starting
        
        start_time = time.time()
        current_time = start_time
        
        print("Starting iterations...")

        for i in range(iterations):
            F_current, JF_current = self.Functions(pi_current)

            # update the next pi
            if self.pinv:
                pi_next = pi_current - np.matmul(np.linalg.pinv(JF_current), F_current)
            else:
                pi_next = pi_current - np.matmul(np.linalg.inv(JF_current), F_current)
            
            F_size = np.sum(np.abs(F_current))
                    
            if F_size < tolerance:
                print("DONE")
                print("Iterations needed: ", i)
                print("Time needed: ", time.time() - start_time)
                print("Final residual error: ", F_size)
                print()
                break
                
            if transparent:
                print("Iteration: ", i)
                print("Error: ", F_size)
                print("--- %s seconds ---" % (time.time() - current_time))
                current_time = time.time()

            # if we haven't converged, update the coefficients
            pi_current = pi_next

        if F_size > tolerance:
            print("Solution did not converge")
            print("Starting guess: ", pi_starting)
            print("Final coefficients: ", pi_current)
            print("Final error: ", F_size)
            print()

        self.pi_coefficients = pi_current
        self.evaluate_solution(pi_current)
            
        return pi_current
    
    def visualize_result_overall(self, pi_final=None, plot_lims=False, domain=False, show=True, save=False):
        '''
        Plots the residual function averaged over the four dimensions, over the domain under consideration, and returns the average value.

        Parameters:
            - pi_final: the vector of coefficients
            - plot_lims: specific limits for the z axis of the plot, if desired
            - domain: the domain on which to plot the residual. If set to None, it will take the same domain Omega as was used for the Galerkin method.
            - show: whether or not to show the plot.
            - save: whether or not to save the plot to a file.

        Returns: The average magnitude of the pointwise residual on the graph.  
        '''
        if pi_final is None:
            pi_final = self.pi_coefficients

        fig = plt.figure(figsize = (26,10))
        ax = plt.axes(projection='3d')
        
        if domain is False:
            x = np.linspace(self.limsx[0], self.limsx[1], 50)
            y = np.linspace(self.limsy[0], self.limsy[1], 50)
        else:
            x = np.linspace(-domain, domain, 50)
            y = np.linspace(-domain, domain, 50)

        X, Y = np.meshgrid(x, y)   
        
        Z_eval = self.check_function(pi_final, X, Y)
        Z_out = np.sqrt(np.sum(np.square(Z_eval)))
        
        plot = ax.plot_surface(X, Y, Z_out, cmap = plt.cm.hot, vmin=-2, vmax=2)

        # Set axes label
        ax.set_xlabel('$\\omega_1$', labelpad=25)
        ax.set_ylabel('$\\omega_2$', labelpad=25)
        ax.set_zlabel('F' + '($\\omega_1$, $\\omega_2$)', labelpad=30, rotation=-300)
        
        if plot_lims:
            ax.set_zlim(plot_lims)

        fig.colorbar(plot, shrink=0.5, aspect=8, pad=0.1)

        title_string = "Plot of F" + "($\\omega_1$, $\\omega_2$) for the "+ self.string_code + " problem ""\n over a domain of " + str(self.limsx) + " x " + str(self.limsy) + " with a maximum degree of " + str(int(self.max_deg))

        ax.set_title(title_string)

        if show:
            plt.show()
        
        if save:
            self.get_file_name()
            if not os.path.exists(self.fig_folder_name):
                os.makedirs(self.fig_folder_name)
            plt.savefig(self.fig_folder_name + self.file_name + '.png')
        
        plt.close()
        
        return np.average(Z_out)
    
    def evaluate_solution(self, pi_final=None, domain=False):
        '''
        Returns a metric giving the L_2 norm of the residual function weighted by the relative size of the coefficients of pi.

        Parameters:
            - pi_final: the vector of coefficients
            - domain: optional, the domain over which you want to evaluate the residual
        '''
        # If used without a value for pi_final, use the coefficients already generated by the solver.
        if pi_final is None:
            pi_final = self.pi_coefficients

        lims_arr = np.zeros((self.d,2))
        
        if domain is False:
            lims_arr[0][0] = self.limsx[0]
            lims_arr[0][1] = self.limsx[1]
            lims_arr[1][0] = self.limsy[0]
            lims_arr[1][1] = self.limsy[1]
        else:
            lims_arr[0][0] = - domain
            lims_arr[0][1] = domain
            lims_arr[1][0] = - domain
            lims_arr[1][1] = domain

        n_gauss = self.n_gauss
        gauss_points, gauss_weights = np.polynomial.legendre.leggauss(n_gauss)
        scaled_gauss_points = np.zeros((self.d, n_gauss))
        scaled_gauss_weights = np.zeros((self.d, n_gauss))

        for i in range(self.d):
            for j in range(n_gauss):
                scaled_gauss_points[i][j] = gauss_points[j] * (lims_arr[i][1] - lims_arr[i][0]) / 2 + (lims_arr[i][1] + lims_arr[i][0]) / 2
                scaled_gauss_weights[i][j] = gauss_weights[j] * (lims_arr[i][1] - lims_arr[i][0]) / 2
        
        Xg, Yg = np.meshgrid(scaled_gauss_points[0], scaled_gauss_points[1])
        Wx, Wy = np.meshgrid(scaled_gauss_weights[0], scaled_gauss_weights[1])
        Wg = np.multiply(Wx, Wy)

        Zg_eval = self.check_function(pi_final, Xg, Yg)
            
        l2_weights = self.l_two_weights(pi_final)
        
        l_2 = self.l_two(Wg, Zg_eval, l2_weights)
                
        self.pi_evaluation = l_2

        return l_2
    
    def l_two_weights(self, pi):
        '''
        Evaluates the relative weights of the coefficients in each dimension

        Parameters:
            - pi: the vector of coefficients

        Returns: an array of length four (as n = 4 here) with the relative weights
        '''
        pi_weights = np.zeros(self.n_state)
        
        for i in range(self.n_state):
            pi_i = pi[i*self.ndim:(i+1)*self.ndim]
            pi_weights[i] = np.linalg.norm(pi_i)
        
        return pi_weights
            
    def l_two(self, W, Z_eval, l2_weights):
        '''
        Finds the l2 metric on the residuals given the weights of each point, their residual evaluations, and the weights of the coefficients.

        Parameters:
            - W: the scaled gauss weighs of each point
            - Z_eval: the evaluation of the residual at each point
            - l2_weights: the weights of the coefficients in each dimension

        Returns: a metric showing the overall performance of the solution for a given pi.
        '''
        integrals = np.zeros(len(Z_eval))
        for i in range(len(Z_eval)):
            integrals[i] = np.sqrt(np.sum(np.multiply(W, Z_eval[i]**2)))
        
        return np.average(integrals, weights=l2_weights)
    
    @staticmethod
    def ncr(n, r):
        '''
        Returns the value of n choose r. 

        Parameters: 
            - n (int): integer
            - r (int): integer less than or equal to n

        Returns: 
            - ncr (int): combinatorial operation n choose r, which should be an inte

        If using Python 3, we can also call math.comb(). This functions, however, works for Python 2 as well.
        '''
        r = min(r, n-r)
        numer = reduce(op.mul, range(n, n-r, -1), 1)
        denom = reduce(op.mul, range(1, r+1), 1)
        return numer / denom

    '''
    The following three functions are basic monomial functions that allow us to generate our monomial basis. They work for using a maximum degree of expansion in terms of the full tensor product or in terms of total degree.
    '''

    @staticmethod
    def monomial_dimension(d, max_deg, total_degree=True):
        '''
        Returns the number of monomials for a given problem's parameters.

        Parameters: 
            - d (int): the dimensionality of the signal generator
            - max_deg (int): the maximum dimensionality of the monomial expansion
            - total_degree (bool): whether or not we are using the total degree of monomial expansion to truncate the basis. Default is set to True.

        Output: 
            - dim (int): the number of monomial dimensions.
        '''
        if total_degree:
            result = int((max_deg + 1) / d * solver.ncr(max_deg+d, d-1))
        else: 
            result = int((max_deg+1)**d)
        return result - 1

    @staticmethod
    def get_monomial(d, max_deg, index, total_degree=True):
        '''
        Returns a certain monomial from a given index. It is the inverse of the function get_index, if the value of d, max_deg, and total_degree are the same.

        Parameters:
            - d: the dimensionality of the signal generator
            - max_deg: the maximum dimensionality of the monomial expansion
            - index: the index of monomial we would like to analyze, ranging from 0 to the maximum monomial dimension

        Returns: 
            - monomial: an array of the form [i_1, i_2, ..., i_d] where the monomial is then given as omega_1^{i_1} * omega_2^{i_2} * ... * omega_d^{i_d}
        '''
        index = index + 1
        if index > solver.monomial_dimension(d, max_deg,total_degree=total_degree):
                print("ERROR: OUT OF RANGE OF MAXIMUM DEGREE")
        
        if total_degree:
            output_arr = np.zeros(d)
            degree = 0
            for i in range(max_deg+1):
                level = solver.ncr(i+d-1, d-1)
                if index <= level - 1:
                    degree = i
                    break
                else:
                    index -= level
            monomial_rep = list(itertools.combinations_with_replacement(range(d), degree))[int(index)]
            monomial_arr = np.zeros(d)
            for elem in monomial_rep:
                monomial_arr[elem] += 1
        
        else:     
            output_arr = np.zeros(d)
            working_index = index
            for i in range(d):
                working_power = (max_deg + 1)**(d - (i + 1))
                output_arr[i] = int(working_index / working_power)
                working_index = working_index % working_power
            monomial_arr = output_arr[::-1]

        return monomial_arr

    @staticmethod
    def get_index(d, max_deg, monomial, total_degree=True):
        '''
        Takes a monomial and returns the index representing it. It is the inverse of the function get_monomial, if the value of d, max_deg, and total_degree are the same.

        Parameters:
            - d: the dimensionality of the signal generator
            - max_deg: the maximum dimensionality of the monomial expansion
            - monomial: an array of the form [i_1, i_2, ..., i_d] representing the monomial
            - total_degree: whether or not we are using total degree of monomial expansions rather than full tensor product. Usually, this should be set to True.

        Returns: 
            - an integer index between 0 and the maximum monomial degree that represents this monomial.
        '''
        if total_degree:
            index = 0
            level = 0
            for i in range(int(sum(monomial)+1)):
                index += solver.ncr(i+d-1, d-1)-1
                level = i
            monomial_rep = list()
            for j in range(d):
                for k in range(int(monomial[j])):
                    monomial_rep.append(j)        
            index += list(itertools.combinations_with_replacement(range(d), level)).index(tuple(monomial_rep))
        
        else:
            reversed_monomial = monomial[::-1]
            index = 0
            for i in range(d):
                index += reversed_monomial[i] * (max_deg + 1)**(d - (i + 1))
                
        return int(index-1)
    
    def update_pi_folder(self, folder_path):
        '''
        Update the path to the folder where you want to save your coefficient files.
        '''
        self.pi_folder_name = folder_path
    
    def update_fig_folder(self, folder_path):
        '''
        Update the path to the folder where you want to save your figures.
        '''
        self.fig_folder_name = folder_path
    
    def get_pi(self, pi, w1, w2):
        '''
        Evaluates the function pi(omega) pointwise at (omega_1, omega_2)

        Parameters:
            - pi: the coefficients for the mapping pi
            - w1, w2: the point to be evaluated at, with w1 = omega_1 and w2 = omega_2

        Returns:
            A vector (pi_1(omega), pi_2(omega), .., pi_n(omega))
        '''
        pi_sum = []
        for i in range(self.n_state):
            pi_i = pi[i*self.ndim:(i+1)*self.ndim]
            pi_i_sum = 0
            for j in range(self.ndim):
                j_phi = solver.get_monomial(self.d, self.max_deg, j, total_degree=self.total_degree)
                pi_i_sum += pi_i[j] * w1**j_phi[0] * w2**j_phi[1]
            pi_sum.append(pi_i_sum)
        pi_sum = np.asarray(pi_sum)
        return pi_sum

    def get_dpi(self, pi, w1, w2):
        '''
        Evaluates the function pi(omega) pointwise at (omega_1, omega_2)

        Parameters:
            - pi: the coefficients for the mapping pi
            - w1, w2: the point to be evaluated at, with w1 = omega_1 and w2 = omega_2

        Returns:
            A 4x2 matrix whose entries are given by dpi_i/domega_j
        '''
        dpi_sum = []
        for i in range(self.n_state):
            pi_i = pi[i*self.ndim:(i+1)*self.ndim]
            dpi_i_dw1_sum = 0
            dpi_i_dw2_sum = 0
            for j in range(self.ndim):
                j_phi = solver.get_monomial(self.d, self.max_deg, j, total_degree=self.total_degree)
                
                dpi_i_dw1_sum += pi_i[j] * j_phi[0] * w1**(j_phi[0]-1) * w2**j_phi[1]
                dpi_i_dw2_sum += pi_i[j] * w1**j_phi[0] * j_phi[1] * w2**(j_phi[1]-1)
            dpi_sum.append([dpi_i_dw1_sum, dpi_i_dw2_sum])
        dpi_sum = np.asarray(dpi_sum)
        return dpi_sum
    
    def get_pi_from_file(self):
        '''
        Get coefficients for the expansion of pi from a file. Called during the constructor if from_file is set to True. Can also be called by itself later.
        '''
        print("Getting coefficients from file...")

        self.get_file_name()

        file_path = self.pi_folder_name + self.file_name + '.csv'

        try: 
            with open(file_path, newline='') as csvfile:
                data = list(csv.reader(csvfile))[0]
                pi_final = np.array([float(i) for i in data])
                self.pi_coefficients = pi_final
                self.evaluate_solution(pi_final)
                print("Successfully got coefficients.")
        except IOError:
           print("Error: File does not appear to exist.")
        
        print()
        
    def save_file(self):
        '''
        Saves the pi coefficients to a file. 
        '''
        print("Saving file")
        if not os.path.exists(self.pi_folder_name):
            os.makedirs(self.pi_folder_name)
        
        self.get_file_name()

        file_path = self.pi_folder_name + self.file_name + '.csv'

        with open(file_path, 'w+') as f:
            self.pi_coefficients.tofile(file_path,sep=',')
        print("Done saving file.")

    def initialize_gauss(self):
        '''
        Initializes the scaled gauss points and weights for numerical integration.
        '''
        # Populate the arrays for the scaled gauss points and scaled gauss weights for integration
        for i in range(self.d):
            for j in range(self.n_gauss):
                self.scaled_gauss_points[i][j] = self.gauss_points[j] * (self.lims_arr[i][1] - self.lims_arr[i][0]) / 2 + (self.lims_arr[i][1] + self.lims_arr[i][0]) / 2
                self.scaled_gauss_weights[i][j] = self.gauss_weights[j] * (self.lims_arr[i][1] - self.lims_arr[i][0]) / 2

class rl_solver(solver):
    '''
    Subclass of solver corresponding to Test 3 and Test 4 in the paper, for a resistor-inductor ladder problem. 
    '''
    def __init__(self, lims, d, max_deg, n_state, alpha, kappa, n_gauss=20, total_degree=True):
        '''
        RL solver constructor. 

        Parameters:
            - lims: a 2D array of limits of integration, for example [-1,1].
            - d: the dimensionality of the signal generator. For this problem, it should be equal to 2.
            - max_deg: the maximum degree of polynomial expansion
            - n_state: the dimensionality of the system.
            - alpha: constant for the RL problem
            - n_gauss: the number of quadrature points to take for numerical integration
            - total_degree: whether or not we are using total degree of monomial expansions rather than full tensor product. Usually, this should be set to True.
        '''
        super().__init__(lims, d, max_deg, n_state, n_gauss=n_gauss, total_degree=total_degree)

        self.alpha = alpha
        self.kappa = kappa

    def Functions(self, pi):
        '''
        Computes the function F and its Jacobian JF for a given set of coefficients.

        Parameters:
            - pi: the coefficients for the mapping pi

        Returns:
            - F: the 1 x ndim * n_state vector function to be minimized
            - JF: the ndim * n_state x ndim * n_state Jacobian of F
        '''
        F = np.zeros(self.ndim * self.n_state)
        JF = np.zeros((self.ndim * self.n_state, self.ndim * self.n_state))
        
        for i in range(self.n_state):
            pi_i = pi[i*self.ndim:(i+1)*self.ndim]
            pi_i_minus = np.zeros(self.ndim) if i == 0 else pi[(i-1)*self.ndim:i*self.ndim]
            pi_i_plus = np.zeros(self.ndim) if i == self.n_state - 1 else pi[(i+1)*self.ndim:(i+2)*self.ndim]
            
            F_i = np.matmul(self.P(pi_i),pi_i) - np.matmul(self.M,pi_i_minus) - np.matmul(self.M,pi_i_plus) - self.gamma * (1 - np.count_nonzero(i))
            
            F[i*self.ndim:(i+1)*self.ndim] = F_i
            

            for j in range(max(0,i-1),min(self.n_state,i+2)):
                if i == j:
                    JF_ij = self.Q(pi_i)
                else: 
                    JF_ij = - self.M
                
                JF[i*self.ndim:(i+1)*self.ndim,j*self.ndim:(j+1)*self.ndim] = JF_ij
        
        return (F, JF)
    
    def P(self, pi):
        '''
        Vector function P to help in constructing the function F.

        Parameters:
            - pi: the coefficients for the mapping pi

        Returns:
            - P: the 1 x ndim vector P
        '''
        return self.A + 2 * self.kappa * self.M + 1 / 2 * self.N_tilde(pi) + 1 / 3 * self.O_tilde(pi)

    def Q(self, pi):
        '''
        Matrix function Q to help in constructing the Jacobian JF.

        Parameters:
            - pi: the coefficients for the mapping pi

        Returns:
            - Q: the ndim x ndim matrix Q
        '''
        return self.A + 2 * self.kappa * self.M + self.N_tilde(pi) + self.O_tilde(pi)
    
    def N_tilde(self, v):
        '''
        Function N tilde that compresses the 3x3 tensor N into a 2D matrix for a given vector of coefficients v.

        Parameters:
            - v: a vector of coefficients

        Returns:
            - Ntil: a ndim x ndim matrix 
        '''
        Ntil = np.zeros((self.ndim, self.ndim))
        
        for i in range(self.ndim):
            for j in range(i, self.ndim):
                for k in range(self.ndim):
                    if self.total_degree:
                        Ntil[i][j] += v[k] * self.N[i][j][k]
                    else:
                        N_index = rl_solver.get_n_index(i,j,k,self.ndim)
                        Ntil[i][j] += v[k] * self.N[N_index]
        Ntil = Ntil + Ntil.T - np.diag(Ntil.diagonal())
        return Ntil

    def O_tilde(self, v):
        '''
        Function O tilde that compresses the 4x4 tensor O into a 2D matrix for a given vector of coefficients v.

        Parameters:
            - v: a vector of coefficients

        Returns:
            - Otil: a ndim x ndim matrix 
        '''
        Otil = np.zeros((self.ndim, self.ndim))
    
        for i in range(self.ndim):
            for j in range(i, self.ndim):
                for k in range(self.ndim):
                    for l in range(self.ndim):
                        if self.total_degree:
                            Otil[i][j] += v[k] * v[l] * self.O[i][j][k][l]
                        else: 
                            O_index = rl_solver.get_o_index(i,j,k,l,self.ndim)
                            Otil[i][j] += v[k] * v[l] * O[O_index]

        Otil = Otil + Otil.T - np.diag(Otil.diagonal())    
        return Otil

    '''
    The following functions are used to speed up the creation of the tensors N and O if total_degree is set to false.
    '''
    @staticmethod
    def get_n_count(n):
        '''
        Returns the number of unique triplets (i,j,k) there are up to a degree n, such that 0 <= i <= j <= k <= n. 

        Parameters:
            - n: the dimensionality of the system
        '''
        return int(np.round(1/6 * n * (n+1) * (n+2)))

    @staticmethod
    def get_o_count(n):
        '''
        Returns the number of unique quadruplets (i,j,k,l) there are up to a degree n, such that 0 <= i <= j <= k <= l <= n.

        Parameters:
            - n: the dimensionality of the system
        '''
        return int(np.round(1/24 * n * (n+1) * (n+2) * (n+3)))

    @staticmethod
    def get_n_index(i,j,k,n):
        '''
        Returns the index given a triplet (i,j,k) in order to retrieve the value from N

        Parameters:
            - i,j,k: the triplet of indices of interest
            - n: the dimensionality of the system

        Returns: 
            - the index in the array N corresponding to that entry.
        '''
        i,j,k = np.sort([i,j,k])
        
        counti = (1/6 * i * (i**2 - 3 * i * (n + 1) + 3 * n**2 + 6 * n + 2))
        countj = (1/2 * (i - j) * (i + j - 2*n - 1))
        countk = k-j
        
        return int(np.round(counti + countj + countk))
        
    @staticmethod
    def get_o_index(i,j,k,l,n):
        '''
        Returns the index given a quadruplet (i,j,k,l) in order to retrieve the value from N

        Parameters:
            - i,j,k,l: the quadruplet of indices of interest
            - n: the dimensionality of the system

        Returns: 
            - the index in the array O corresponding to that entry.
        '''
        i,j,k,l = np.sort([i,j,k,l])
        
        counti = -1/24 * i * (i - 2*n - 3) * (i**2 - 2*i*n - 3*i + 2*n**2 + 6*n + 2)
        countj = -1/6 * (i-j) * (i**2 + i*(j - 3*(n+1)) + j**2 - 3*j*(n+1) + 3*n**2 + 6*n + 2)
        countk = 1/2 * (j-k) * (j + k - 2*n - 1)
        countl = l-k
        
        return int(np.round(counti + countj + countk + countl))

class rl_linear_solver(rl_solver):
    '''
    Subclass of rl_solver corresponding to Test 3 in the paper, for a resistor-inductor ladder problem with a linear oscillator as a signal generator. 
    '''
    def __init__(self, lims, d, max_deg, n_state, alpha, kappa, n_gauss=20, total_degree=True, from_file=False):
        '''
        RL ladder linear oscillator solver constructor. If you are saving to files or reading from files, change the path to where your files are stored in 'self.pi_folder_name' and where your figures are stored in 'self.fig_folder_name'.

        Parameters:
            - lims: a 2D array of limits of integration, for example [-1,1].
            - d: the dimensionality of the signal generator. For this problem, it should be equal to 2.
            - max_deg: the maximum degree of polynomial expansion
            - n_state: the dimensionality of the system.
            - alpha, kappa: constants for the RL linear oscillator problem
            - n_gauss: the number of quadrature points to take for numerical integration
            - total_degree: whether or not we are using total degree of monomial expansions rather than full tensor product. Usually, this should be set to True.
            - from_file: whether or not we are reading existing from a file. Usually, this will be set to false unless we have already run the code before.
        '''
        super().__init__(lims, d, max_deg, n_state, alpha, kappa, n_gauss=n_gauss, total_degree=total_degree)

        self.pi_folder_name = 'pi_folder/rl_linear/'
        self.fig_folder_name = 'fig_folder/rl_linear/'

        self.string_code = 'RL linear oscillator'

        if not from_file:
            self.initialize_problem()

        if from_file:
            self.get_pi_from_file()

    def get_file_name(self):
        '''
        Gets the file name given the parameters and coefficients used.
        '''
        file_name = 'pi_final_rl_linear_'
        
        file_name += ('dim' + str(int(self.n_state)) + '_')
        file_name += ('lims' + str(round(self.lims_arr[0][1],2)) + '_')
        file_name += ('deg' + str(int(self.max_deg)) + '_')
        file_name += ('kappa' + str(round(self.kappa, 2)) + '_')
        file_name += ('alpha' + str(round(self.alpha, 2)) + '_')
        
        if not self.total_degree:
            file_name += ('fullrank')
            
        file_name = re.sub(r'\.0*_', '_', file_name)
        file_name = file_name.replace('.','-')

        self.file_name = file_name
        
        return file_name

    def initialize_problem(self):
        '''
        Initializes the necessary matrices and vectors for use in the Galerkin method. Builds the vector gamma, the matrices A and M, and the tensors N and M.
        '''
        print()
        print('Starting initialization of matrices...')

        start_time = time.time()

        self.A = np.zeros((self.ndim, self.ndim))
        self.M = np.zeros((self.ndim, self.ndim))
        self.gamma = np.zeros(self.ndim)

        if self.total_degree:
            self.N = np.zeros((self.ndim,self.ndim,self.ndim))
            self.O = np.zeros((self.ndim,self.ndim,self.ndim,self.ndim))
        else:
            self.N = np.zeros(rl_solver.get_n_count(self.ndim))
            self.O = np.zeros(rl_solver.get_o_count(self.ndim))

            N_index = 0
            O_index = 0

        for i in range(self.ndim):
            i_phi = solver.get_monomial(self.d, self.max_deg, i, total_degree=self.total_degree)
            
            # update gamma
            product = 1
            for dim in range(self.d):
                summed_index = i_phi[dim] + 1
                if dim == 1:
                    summed_index += 1
                limits = self.lims_arr[dim]
                product = product * ((limits[1]**(summed_index) - limits[0]**(summed_index)) / (summed_index))
            self.gamma[i] = product
            
            for j in range(self.ndim): 
                j_phi = solver.get_monomial(self.d, self.max_deg, j, total_degree=self.total_degree)
                
                # update A
                product_1 = self.alpha * j_phi[0]
                product_2 = -self.alpha * j_phi[1]
                for dim in range(self.d):
                    # works for 2 dimensions which is what we are working in
                    summed_index_1 = i_phi[dim] + j_phi[dim] + 2 * dim
                    summed_index_2 = i_phi[dim] + j_phi[dim] + 2 - 2 * dim
                    limits = self.lims_arr[dim] 
                    if product_1 != 0:
                        product_1 = product_1 * ((limits[1]**(summed_index_1) - limits[0]**(summed_index_1)) / (summed_index_1))
                    if product_2 != 0:    
                        product_2 = product_2 * ((limits[1]**(summed_index_2) - limits[0]**(summed_index_2)) / (summed_index_2))
                self.A[i][j] = product_1 + product_2
                
                # update M
                product = 1
                for dim in range(self.d):
                    summed_index = i_phi[dim] + j_phi[dim] + 1
                    limits = self.lims_arr[dim]
                    product = product * ((limits[1]**(summed_index) - limits[0]**(summed_index)) / (summed_index))
                self.M[i][j] = product
                
            if self.total_degree:
                for j in range(self.ndim):
                    j_phi = solver.get_monomial(self.d, self.max_deg, j, total_degree=self.total_degree)
                    for k in range(self.ndim):
                        k_phi = solver.get_monomial(self.d, self.max_deg, k, total_degree=self.total_degree)
                        product = 1
                        for dim in range(self.d):
                            summed_index = i_phi[dim] + j_phi[dim] + k_phi[dim] + 1
                            limits = self.lims_arr[dim]
                            product = product * ((limits[1]**(summed_index) - limits[0]**(summed_index)) / (summed_index))
                        self.N[i][j][k] = product

                        for l in range(self.ndim):
                            l_phi = solver.get_monomial(self.d, self.max_deg, l, total_degree=self.total_degree)
                            product = 1
                            for dim in range(self.d):
                                summed_index = i_phi[dim] + j_phi[dim] + k_phi[dim] + l_phi[dim] + 1
                                limits = self.lims_arr[dim]
                                product = product * ((limits[1]**(summed_index) - limits[0]**(summed_index)) / (summed_index)) 
                            self.O[i][j][k][l] = product
                    
            else:
                
                for j in range(i,self.ndim):
                    j_phi = solver.get_monomial(self.d, self.max_deg, j, total_degree=self.total_degree)
                    for k in range(j,self.ndim):
                        k_phi = solver.get_monomial(self.d, self.max_deg, k, total_degree=self.total_degree)

                        product = 1
                        for dim in range(self.d):
                            summed_index = i_phi[dim] + j_phi[dim] + k_phi[dim] + 1
                            limits = self.lims_arr[dim]
                            product = product * ((limits[1]**(summed_index) - limits[0]**(summed_index)) / (summed_index))
                        self.N[N_index] = product
                        N_index += 1

                        for l in range(k,self.ndim):
                            l_phi = self.get_monomial(self.d, self.max_deg, l, total_degree=self.total_degree)

                            product = 1
                            for dim in range(self.d):
                                summed_index = i_phi[dim] + j_phi[dim] + k_phi[dim] + l_phi[dim] + 1
                                limits = self.lims_arr[dim]
                                product = product * ((limits[1]**(summed_index) - limits[0]**(summed_index)) / (summed_index)) 

                            self.O[O_index] = product
                            O_index += 1
        
        print('Time taken: ', time.time() - start_time)
        print('Finished initialization')
        print('')

    def check_function(self, pi, w1, w2):
        '''
        Evaluates the residual equation at a point, given a vector of coefficients and a point in space.

        Parameters:
            - pi: the coefficients for the mapping pi
            - w1, w2: the point to be evaluated at, with w1 = omega_1 and w2 = omega_2

        Returns: A vector of the residual function (R_1, ... R_n)
        '''
        pi_eval = self.get_pi(pi, w1, w2)
        pi_deriv = self.get_dpi(pi, w1, w2)
        
        F_eval = np.zeros(self.n_state,dtype=object)

        for i in range(self.n_state):
            pi_eval_minus = 0 if i == 0 else pi_eval[i-1]
            pi_eval_plus = 0 if i == self.n_state - 1 else pi_eval[i+1]       
            
            F_eval[i] = pi_deriv[i][0] * self.alpha * w2 - pi_deriv[i][1] * self.alpha * w1 - pi_eval_minus + 2 * self.kappa * pi_eval[i] \
                        - pi_eval_plus + 1/2 * pi_eval[i]**2 + 1/3 * pi_eval[i]**3 - w2 * (1 - np.count_nonzero(i))
                        
        return F_eval

class rl_vdp_solver(rl_solver):
    '''
    Subclass of rl_solver corresponding to Test 4 in the paper, for a resistor-inductor ladder problem with a Van der Pol oscillator as a signal generator. 
    '''
    def __init__(self, lims, d, max_deg, n_state, alpha, kappa, mu, n_gauss=20, total_degree=True, from_file=False):
        '''
        RL ladder Van der Pol oscillator solver constructor. If you are saving to files or reading from files, change the path to where your files are stored in 'self.pi_folder_name' and where your figures are stored in 'self.fig_folder_name'.

        Parameters:
            - lims: a 2D array of limits of integration, for example [-1,1].
            - d: the dimensionality of the signal generator. For this problem, it should be equal to 2.
            - max_deg: the maximum degree of polynomial expansion
            - n_state: the dimensionality of the system.
            - alpha, kappa, mu: constants for the RL linear oscillator problem
            - n_gauss: the number of quadrature points to take for numerical integration
            - total_degree: whether or not we are using total degree of monomial expansions rather than full tensor product. Usually, this should be set to True.
            - from_file: whether or not we are reading existing from a file. Usually, this will be set to false unless we have already run the code before.
        '''
        super().__init__(lims, d, max_deg, n_state, alpha, kappa, n_gauss=n_gauss, total_degree=total_degree)

        self.pi_folder_name = 'pi_folder/rl_vdp/'
        self.fig_folder_name = 'fig_folder/rl_vdp/'

        self.string_code = 'RL Van der Pol oscillator'

        self.mu = mu

        if not from_file:
            self.initialize_problem()

        if from_file:
            self.get_pi_from_file()

    def get_file_name(self):
        '''
        Gets the file name given the parameters and coefficients used.
        '''
        file_name = 'pi_final_rl_vdp_'
        
        file_name += ('dim' + str(int(self.n_state)) + '_')
        file_name += ('lims' + str(round(self.lims_arr[0][1],2)) + '_')
        file_name += ('deg' + str(int(self.max_deg)) + '_')
        file_name += ('kappa' + str(round(self.kappa, 2)) + '_')
        file_name += ('alpha' + str(round(self.alpha, 2)) + '_')
        file_name += ('mu' + str(round(self.mu, 2)) + '_')
        
        if not self.total_degree:
            file_name += ('fullrank')
            
        file_name = re.sub(r'\.0*_', '_', file_name)
        file_name = file_name.replace('.','-')

        self.file_name = file_name
        
        return file_name

    def initialize_problem(self):
        '''
        Initializes the necessary matrices and vectors for use in the Galerkin method. Builds the vector gamma, the matrices A and M, and the tensors N and M.
        '''
        print()
        print('Starting initialization of matrices...')

        start_time = time.time()

        self.A = np.zeros((self.ndim, self.ndim))
        self.M = np.zeros((self.ndim, self.ndim))
        self.gamma = np.zeros(self.ndim)

        if self.total_degree:
            self.N = np.zeros((self.ndim,self.ndim,self.ndim))
            self.O = np.zeros((self.ndim,self.ndim,self.ndim,self.ndim))
        else:
            self.N = np.zeros(rl_solver.get_n_count(self.ndim))
            self.O = np.zeros(rl_solver.get_o_count(self.ndim))

            N_index = 0
            O_index = 0

        for i in range(self.ndim):
            i_phi = solver.get_monomial(self.d, self.max_deg, i, total_degree=self.total_degree)
            
            # update gamma
            product = 1
            for dim in range(self.d):
                summed_index = i_phi[dim] + 1
                if dim == 1:
                    summed_index += 1
                limits = self.lims_arr[dim]
                product = product * ((limits[1]**(summed_index) - limits[0]**(summed_index)) / (summed_index))
            self.gamma[i] = product
            
            for j in range(self.ndim): 
                j_phi = solver.get_monomial(self.d, self.max_deg, j, total_degree=self.total_degree)
                
                # update A
                product_1 = self.alpha * j_phi[0]
                product_2 = -self.alpha * j_phi[1]
                product_3 = self.alpha*self.mu*j_phi[1]
                product_4 = -self.alpha*self.mu*j_phi[1]

                for dim in range(self.d):
                    # works for 2 dimensions which is what we are working in
                    summed_index_1 = i_phi[dim] + j_phi[dim] + 2 * dim
                    summed_index_2 = i_phi[dim] + j_phi[dim] + 2 - 2 * dim
                    summed_index_3 = i_phi[dim] + j_phi[dim] + 1
                    summed_index_4 = i_phi[dim] + j_phi[dim] + 3 - 2 * dim
                    limits = self.lims_arr[dim]

                    if product_1 != 0:
                        product_1 = product_1 * ((limits[1]**(summed_index_1) - limits[0]**(summed_index_1)) / (summed_index_1))
                    if product_2 != 0:    
                        product_2 = product_2 * ((limits[1]**(summed_index_2) - limits[0]**(summed_index_2)) / (summed_index_2))
                    if product_3 != 0:
                        product_3 = product_3 * ((limits[1]**(summed_index_3) - limits[0]**(summed_index_3)) / (summed_index_3))
                    if product_4 != 0:    
                        product_4 = product_4 * ((limits[1]**(summed_index_4) - limits[0]**(summed_index_4)) / (summed_index_4))
                self.A[i][j] = product_1 + product_2 + product_3 + product_4
                
                # update M
                product = 1
                for dim in range(self.d):
                    summed_index = i_phi[dim] + j_phi[dim] + 1
                    limits = self.lims_arr[dim]
                    product = product * ((limits[1]**(summed_index) - limits[0]**(summed_index)) / (summed_index))
                self.M[i][j] = product
                
            if self.total_degree:
                for j in range(self.ndim):
                    j_phi = solver.get_monomial(self.d, self.max_deg, j, total_degree=self.total_degree)
                    for k in range(self.ndim):
                        k_phi = solver.get_monomial(self.d, self.max_deg, k, total_degree=self.total_degree)
                        product = 1
                        for dim in range(self.d):
                            summed_index = i_phi[dim] + j_phi[dim] + k_phi[dim] + 1
                            limits = self.lims_arr[dim]
                            product = product * ((limits[1]**(summed_index) - limits[0]**(summed_index)) / (summed_index))
                        self.N[i][j][k] = product

                        for l in range(self.ndim):
                            l_phi = solver.get_monomial(self.d, self.max_deg, l, total_degree=self.total_degree)
                            product = 1
                            for dim in range(self.d):
                                summed_index = i_phi[dim] + j_phi[dim] + k_phi[dim] + l_phi[dim] + 1
                                limits = self.lims_arr[dim]
                                product = product * ((limits[1]**(summed_index) - limits[0]**(summed_index)) / (summed_index)) 
                            self.O[i][j][k][l] = product
                    
            else:
                
                for j in range(i,self.ndim):
                    j_phi = solver.get_monomial(self.d, self.max_deg, j, total_degree=self.total_degree)
                    for k in range(j,self.ndim):
                        k_phi = solver.get_monomial(self.d, self.max_deg, k, total_degree=self.total_degree)

                        product = 1
                        for dim in range(self.d):
                            summed_index = i_phi[dim] + j_phi[dim] + k_phi[dim] + 1
                            limits = self.lims_arr[dim]
                            product = product * ((limits[1]**(summed_index) - limits[0]**(summed_index)) / (summed_index))
                        self.N[N_index] = product
                        N_index += 1

                        for l in range(k,self.ndim):
                            l_phi = self.get_monomial(self.d, self.max_deg, l, total_degree=self.total_degree)

                            product = 1
                            for dim in range(self.d):
                                summed_index = i_phi[dim] + j_phi[dim] + k_phi[dim] + l_phi[dim] + 1
                                limits = self.lims_arr[dim]
                                product = product * ((limits[1]**(summed_index) - limits[0]**(summed_index)) / (summed_index)) 

                            self.O[O_index] = product
                            O_index += 1
        
        print('Time taken: ', time.time() - start_time)
        print('Finished initialization')
        print('')

    def check_function(self, pi, w1, w2):
        '''
        Evaluates the residual equation at a point, given a vector of coefficients and a point in space.

        Parameters:
            - pi: the coefficients for the mapping pi
            - w1, w2: the point to be evaluated at, with w1 = omega_1 and w2 = omega_2

        Returns: A vector of the residual function (R_1, ... R_n)
        '''
        pi_eval = self.get_pi(pi, w1, w2)
        pi_deriv = self.get_dpi(pi, w1, w2)
        
        F_eval = np.zeros(self.n_state,dtype=object)

        for i in range(self.n_state):
            pi_eval_minus = 0 if i == 0 else pi_eval[i-1]
            pi_eval_plus = 0 if i == self.n_state - 1 else pi_eval[i+1]       
            
            F_eval[i] = pi_deriv[i][0] * self.alpha * w2 + pi_deriv[i][1] * self.alpha * (self.mu * (1 - w1**2) * w2 - w1) \
                        - pi_eval_minus + 2 * self.kappa * pi_eval[i] - pi_eval_plus + 1/2 * pi_eval[i]**2 + 1/3 * pi_eval[i]**3 \
                        - w2 * (1 - np.count_nonzero(i))
                        
        return F_eval

class low_dim_solver(solver):
    '''
    Subclass of solver corresponding to Test 1 in the paper, for the low-dimensional example problem.
    '''
    def __init__(self, lims, d, max_deg, alpha, n_gauss=20, total_degree=True, from_file=False):
        '''
        Low dimensional problem solver constructor. If you are saving to files or reading from files, change the path to where your files are stored in 'self.pi_folder_name' and where your figures are stored in 'self.fig_folder_name'.

        Parameters:
            - lims: a 2D array of limits of integration, for example [-1,1].
            - d: the dimensionality of the signal generator. For this problem, it should be equal to 2.
            - max_deg: the maximum degree of polynomial expansion
            - alpha: constants for the RL linear oscillator problem
            - n_gauss: the number of quadrature points to take for numerical integration
            - total_degree: whether or not we are using total degree of monomial expansions rather than full tensor product. Usually, this should be set to True.
            - from_file: whether or not we are reading existing from a file. Usually, this will be set to false unless we have already run the code before.
        '''
        n_state = 2
        super().__init__(lims, d, max_deg, n_state, n_gauss=n_gauss, total_degree=total_degree)

        # CHANGE THESE TO WHICHEVER FOLDER YOU WOULD LIKE
        self.pi_folder_name = 'pi_folder/lowdim/'
        self.fig_folder_name = 'fig_folder/lowdim/'

        self.string_code = 'low dimensional'

        self.alpha = alpha

        if not from_file:
            self.initialize_problem()

        if from_file:
            self.get_pi_from_file()

    def get_file_name(self):
        '''
        Gets the file name given the parameters and coefficients used.
        '''
        file_name = 'pi_final_lowdim_'
        
        file_name += ('lims' + str(round(self.lims_arr[0][1],2)) + '_')
        file_name += ('deg' + str(int(self.max_deg)) + '_')
        file_name += ('alpha' + str(round(self.alpha, 2)) + '_')
        
        if not self.total_degree:
            file_name += ('fullrank')
            
        file_name = re.sub(r'\.0*_', '_', file_name)
        file_name = file_name.replace('.','-')

        self.file_name = file_name
        
        return file_name

    def initialize_problem(self):
        '''
        Initializes the necessary matrices and vectors for use in the Galerkin method. Builds the vector gamma, and the matrices A, M and P.
        '''
        print()
        print('Starting initialization of matrices...')

        start_time = time.time()

        self.A = np.zeros((self.ndim, self.ndim))
        self.M = np.zeros((self.ndim, self.ndim))
        self.P = np.zeros((self.ndim, self.ndim))
        self.gamma = np.zeros(self.ndim)

        # constructing the vector gamma
        for i in range(self.ndim):
            i_phi = solver.get_monomial(self.d, self.max_deg, i)
            product = 1
            for dim in range(self.d):
                summed_index = i_phi[dim] + 1
                if dim == 0:
                    summed_index += 1
                limits = self.lims_arr[dim]
                product = product * ((limits[1]**(summed_index) - limits[0]**(summed_index)) / (summed_index))
            self.gamma[i] = product
        
            # constructing the matrices now
            for j in range(self.ndim):
                j_phi = solver.get_monomial(self.d, self.max_deg, j)

                # update A
                product_1 = self.alpha * j_phi[0]
                product_2 = -self.alpha * j_phi[1]
                for dim in range(self.d):
                    # works for 2 dimensions which is what we are working in
                    summed_index_1 = i_phi[dim] + j_phi[dim] + 2 * dim
                    summed_index_2 = i_phi[dim] + j_phi[dim] + 2 - 2 * dim
                    limits = self.lims_arr[dim] 
                    if product_1 != 0:
                        product_1 = product_1 * ((limits[1]**(summed_index_1) - limits[0]**(summed_index_1)) / (summed_index_1))
                    if product_2 != 0:    
                        product_2 = product_2 * ((limits[1]**(summed_index_2) - limits[0]**(summed_index_2)) / (summed_index_2))
                self.A[i][j] = product_1 + product_2

                # update M
                product = 1
                for dim in range(self.d):
                    summed_index = i_phi[dim] + j_phi[dim] + 1
                    limits = self.lims_arr[dim]
                    product = product * ((limits[1]**(summed_index) - limits[0]**(summed_index)) / (summed_index))
                self.M[i][j] = product
                
                # update P
                product = 1
                for dim in range(self.d):
                    summed_index = i_phi[dim] + j_phi[dim] + 1
                    if dim == 0:
                        summed_index += 1
                    limits = self.lims_arr[dim]
                    product = product * ((limits[1]**(summed_index) - limits[0]**(summed_index)) / (summed_index))
                self.P[i][j] = product

        print('Time taken: ', time.time() - start_time)
        print('Finished initialization')
        print('')

    def check_function(self, pi, w1, w2):
        '''
        Evaluates the residual equation at a point, given a vector of coefficients and a point in space.

        Parameters:
            - pi: the coefficients for the mapping pi
            - w1, w2: the point to be evaluated at, with w1 = omega_1 and w2 = omega_2

        Returns: A vector of the residual function (R_1, R_2)
        '''
        pi_eval = self.get_pi(pi, w1, w2)
        pi_deriv = self.get_dpi(pi, w1, w2)
        
        F_eval = np.zeros(self.n_state,dtype=object)
        
        F_eval[0] = pi_deriv[0][0] * self.alpha * w2 - pi_deriv[0][1] * self.alpha * w1 + pi_eval[0] - w1
        F_eval[1] = pi_deriv[1][0] * self.alpha * w2 - pi_deriv[1][1] * self.alpha * w1 + pi_eval[1] - w1 * pi_eval[0]
                
        return F_eval

    def Functions(self, pi):
        '''
        Computes the function F and its Jacobian JF for a given set of coefficients.

        Parameters:
            - pi: the coefficients for the mapping pi

        Returns:
            - F: the 1 x ndim * n_state vector function to be minimized
            - JF: the ndim * n_state x ndim * n_state Jacobian of F
        '''
        F_e = np.zeros(self.ndim * self.n_state)
        JF_e = np.zeros((self.ndim * self.n_state, self.ndim * self.n_state))
        
        pi_1 = pi[0:self.ndim]
        pi_2 = pi[self.ndim:2*self.ndim]
        
        F_e[0:self.ndim] = np.matmul(self.A,pi_1) + np.matmul(self.M,pi_1) - self.gamma
        F_e[self.ndim:2*self.ndim] = np.matmul(self.A,pi_2) + np.matmul(self.M,pi_2) - np.matmul(self.P,pi_1)
        
        JF_e[0:self.ndim,0:self.ndim] = self.A + self.M
        JF_e[0:self.ndim,self.ndim:2*self.ndim] = np.zeros((self.ndim,self.ndim))
        JF_e[self.ndim:2*self.ndim,0:self.ndim] = -self.P
        JF_e[self.ndim:2*self.ndim,self.ndim:2*self.ndim] = self.A + self.M
        
        return (F_e, JF_e)

class pendulum_solver(solver):
    '''
    Subclass of solver corresponding to Test 2 in the paper, for the cart pendulum problem. 
    '''
    def __init__(self, lims, d, max_deg, a_1, a_2, k_const, n_gauss=20, total_degree=True, from_file=False):
        '''
        Pendulum solver constructor. If you are saving to files or reading from files, change the path to where your files are stored in 'self.pi_folder_name' and where your figures are stored in 'self.fig_folder_name'.

        Parameters:
            - lims: a 2D array of limits of integration, for example [-1,1].
            - d: the dimensionality of the signal generator. For this problem, it should be equal to 2.
            - max_deg: the maximum degree of polynomial expansion
            - a_1, a_2, k_const: constants for the pendulum problem
            - n_gauss: the number of quadrature points to take for numerical integration
            - total_degree: whether or not we are using total degree of monomial expansions rather than full tensor product. Usually, this should be set to True.
            - from_file: whether or not we are reading existing from a file. Usually, this will be set to false unless we have already run the code before.
        '''
        n_state = 4

        super().__init__(lims, d, max_deg, n_state, n_gauss=n_gauss, total_degree=total_degree)

        print(self.n_state)

        # CHANGE THIS TO WHICHEVER FOLDER YOU WOULD LIKE
        self.pi_folder_name = 'pi_folder/pendulum/'
        self.fig_folder_name = 'fig_folder/pendulum/'

        self.string_code = 'pendulum'

        self.a_1 = a_1
        self.a_2 = a_2
        self.k_const = k_const

        self.pinv = True

        # First condition necessary for the system is that k < - 1/a_2
        print('Condition: k < -1/a_2: ', self.k_const < - 1 / self.a_2)

        if not (self.k_const < - 1 / self.a_2):
            print('ERROR: THE CONSTANTS DO NOT FULFIL THE CONDITIONS ON THE SYSTEM.')
        
        if not from_file:
            self.initialize_problem()

        if from_file:
            self.get_pi_from_file()

    def get_file_name(self):
        '''
        Gets the file name given the parameters and coefficients used.
        '''
        file_name = 'pi_final_pendulum_'
        
        file_name += ('lims' + str(round(self.lims_arr[0][1],2)) + '_')
        file_name += ('deg' + str(int(self.max_deg)) + '_')
        file_name += ('a1' + str(round(self.a_1, 2)) + '_')
        file_name += ('a2' + str(round(self.a_2, 2)) + '_')
        file_name += ('k' + str(round(self.k_const, 2)) + '_')
        
        if not self.total_degree:
            file_name += ('fullrank')
            
        file_name = re.sub(r'\.0*_', '_', file_name)
        file_name = file_name.replace('.','-')

        self.file_name = file_name
        
        return file_name
    
    def initialize_problem(self):
        '''
        Initializes the necessary matrices and vectors for use in the Galerkin method.

        Checks the condition on the coefficients for the problem to be valid, creates the scaled gauss points and weights. Builds the vector gamma, the matrix A, and the matrix M.
        '''
        print()
        print('Starting initialization of matrices...')

        start_time = time.time()

        self.A = np.zeros((self.ndim, self.ndim))
        self.M = np.zeros((self.ndim, self.ndim))
        self.gamma = np.zeros(self.ndim)

        # here, we build the NxN matrix A
        for i in range(self.ndim):
            i_phi = solver.get_monomial(self.d, self.max_deg, i)
            for j in range(self.ndim):
                j_phi = solver.get_monomial(self.d, self.max_deg, j)
                product_1 = j_phi[0]
                product_2 = j_phi[1]
                for dim in range(self.d):
                    summed_index_1 = i_phi[dim] + j_phi[dim] + 2 * dim
                    summed_index_2 = i_phi[dim] + j_phi[dim]
                    limits = self.lims_arr[dim]
                    if product_1 != 0:
                        product_1 = product_1 * ((limits[1]**(summed_index_1) - limits[0]**(summed_index_1)) / (summed_index_1))
                    if product_2 != 0:
                        if dim == 0:
                            quad_sum = 0
                            for k in range(self.n_gauss):
                                x = self.scaled_gauss_points[dim][k]
                                w = self.scaled_gauss_weights[dim][k]
                                quad_sum = quad_sum + w * self.A_func(x, i_phi[dim], j_phi[dim])
                            product_2 = product_2 * quad_sum
                        elif dim == 1:
                            product_2 = product_2 * ((limits[1]**(summed_index_2) - limits[0]**(summed_index_2)) / (summed_index_2))
                self.A[i][j] = product_1 + product_2
        
        # here, we build the NxN matrix M
        for i in range(self.ndim):
            i_phi = solver.get_monomial(self.d, self.max_deg, i)
            for j in range(self.ndim):
                j_phi = solver.get_monomial(self.d, self.max_deg, j)
                product = 1
                for dim in range(self.d):
                    summed_index = i_phi[dim] + j_phi[dim] + 1
                    limits = self.lims_arr[dim]
                    product = product * ((limits[1]**(summed_index) - limits[0]**(summed_index)) / (summed_index))
                self.M[i][j] = product

        # here, we build the Nx1 vector gamma
        for i in range(self.ndim):
            i_phi = solver.get_monomial(self.d, self.max_deg, i)
            product = 1
            for dim in range(self.d):
                if dim == 0:
                    quad_sum = 0
                    for k in range(self.n_gauss):
                        x = self.scaled_gauss_points[dim][k]
                        w = self.scaled_gauss_weights[dim][k]
                        quad_sum = quad_sum + w * self.gamma_func(x, i_phi[dim])
                    product = product * quad_sum
                elif dim == 1:
                    summed_index_2 = i_phi[dim] + 1
                    limits = self.lims_arr[dim]
                    product = product * ((limits[1]**(summed_index_2) - limits[0]**(summed_index_2)) / (summed_index_2))
            self.gamma[i] = product

        print('Time taken: ', time.time() - start_time)
        print('Finished initialization')
        print('')
    
    def check_function(self, pi, w1, w2):
        '''
        Evaluates the residual equation at a point, given a vector of coefficients and a point in space.

        Parameters:
            - pi: the coefficients for the mapping pi
            - w1, w2: the point to be evaluated at, with w1 = omega_1 and w2 = omega_2

        Returns: A vector of the residual function (R_1, R_2, R_3, R_4)
        '''
        pi_eval = self.get_pi(pi, w1, w2)
        pi_deriv = self.get_dpi(pi, w1, w2)
        
        numer = self.a_1 * np.sin(w1)
        denom = 1 + self.k_const * self.a_2 * np.cos(w1)
        lhs_eval = numer / denom
        u_eval = self.k_const * numer / denom   
        
        F_eval = np.zeros(self.n_state,dtype=object)
        
        F_eval[0] = pi_deriv[0][0] * w2 + pi_deriv[0][1] * lhs_eval - pi_eval[2]
        F_eval[1] = pi_deriv[1][0] * w2 + pi_deriv[1][1] * lhs_eval - pi_eval[3]
        F_eval[2] = pi_deriv[2][0] * w2 + pi_deriv[2][1] * lhs_eval - self.a_1 * np.sin(pi_eval[0]) + self.a_2 * np.cos(pi_eval[0]) * u_eval
        F_eval[3] = pi_deriv[3][0] * w2 + pi_deriv[3][1] * lhs_eval - u_eval

        F_eval = np.array(F_eval)
        
        return F_eval
    
    def Functions(self, pi_current):
        '''
        Computes the function F and its Jacobian JF for a given set of coefficients.

        Parameters:
            - pi_current: the coefficients for the mapping pi

        Returns:
            - F_current: the 1 x ndim * n_state vector function to be minimized
            - JF_current: the ndim * n_state x ndim * n_state Jacobian of F
        '''
        c_current = pi_current[0:self.ndim]
        d_current = pi_current[self.ndim:2*self.ndim]
        m_current = pi_current[2*self.ndim:3*self.ndim]
        n_current = pi_current[3*self.ndim:]
        
        F_current = self.F(c_current, d_current, m_current, n_current)
        JF_current = self.JF(c_current)

        return (F_current, JF_current)
        
    def A_func(self, x, i_1, j_1):
        '''
        Function that takes a point x, as well as exponents i_1 and j_1, and returns the nonlinear part of the function, in terms of omega_1, to be integrated in building A

        Parameters:
            - x: the point in space to be analyzed. In particular, this is the value of omega_1 in the integrand for building the matrix A
            - i_1: the exponent for omega_1 for the test function phi_i
            - j_1: the exponent for omega_1 for the basis function phi_j
        '''
        numer = self.a_1 * np.sin(x)
        denom = 1 + self.k_const * self.a_2 * np.cos(x)
        return numer / denom * (x ** (i_1 + j_1))

    def gamma_func(self, x, i_1):
        '''
        Function that takes a point x and the exponent i_1, and returns the nonlinear integand for gamma.
        
        Parameters:
            - x: the point in space to be analyzed. This is the value of omega_1 in the integrand for building the vector gamma
            - i_1: the exponent for omega_1 for the test function phi_i'''
        numer = self.k_const * self.a_1 * np.sin(x)
        denom = 1 + self.k_const * self.a_2 * np.cos(x)
        return numer / denom * (x ** i_1)

    def G_func(self, c, i):
        '''
        Function that takes a vector of coefficients c and and index i, and returns the ith row of G, the nonlinear part of the function F, which shows up in F_3.
        
        Parameters:
            - c: the vector of coefficients
            - i: the index of the function G
        '''
        i_phi = solver.get_monomial(self.d, self.max_deg, i)
        i_1 = i_phi[0]
        i_2 = i_phi[1]
        
        sum_1 = 0
        sum_2 = 0
        
        for k in range(self.n_gauss):
            x_1 = self.scaled_gauss_points[0][k]
            w_1 = self.scaled_gauss_weights[0][k]
            for l in range(self.n_gauss):
                x_2 = self.scaled_gauss_points[1][l]
                w_2 = self.scaled_gauss_weights[1][l]
                
                inside_sum = 0
                for j in range(self.ndim):
                    j_phi = solver.get_monomial(self.d, self.max_deg, j)
                    j_1 = j_phi[0]
                    j_2 = j_phi[1]
                    
                    inside_sum += c[j] * x_1**j_1 * x_2**j_2
                    
                inside_frac = (self.k_const * self.a_1 * np.sin(x_1)) / (1 + self.k_const * self.a_2 * np.cos(x_1))
                
                sum_1 = sum_1 + w_1 * w_2 * self.a_1 * np.sin(inside_sum) * x_1**i_1 * x_2**i_2
                sum_2 = sum_2 + w_1 * w_2 * self.a_2 * np.cos(inside_sum) * inside_frac * x_1**i_1 * x_2**i_2
        
        return sum_2 - sum_1

    def dG_func(self, c, i, j):
        '''
        Takes a vector of coefficients c and two indices i and j, and returns the the derivative of ith row of G with respect to the jth coefficient of c, which shows up in the Jacobian JF.
        
        Parameters:
            - c: the vector of coefficients
            - i: the index of the function G
        '''
        i_phi = solver.get_monomial(self.d, self.max_deg, i)
        i_1 = i_phi[0]
        i_2 = i_phi[1]
        
        j_phi = solver.get_monomial(self.d, self.max_deg, j)
        j_1 = j_phi[0]
        j_2 = j_phi[1]
        
        sum_1 = 0
        sum_2 = 0
        
        for k in range(self.n_gauss):
            x_1 = self.scaled_gauss_points[0][k]
            w_1 = self.scaled_gauss_weights[0][k]
            for l in range(self.n_gauss):
                x_2 = self.scaled_gauss_points[1][l]
                w_2 = self.scaled_gauss_weights[1][l]
                
                inside_sum = 0
                for j_inside in range(self.ndim):
                    j_inside_phi = solver.get_monomial(self.d, self.max_deg, j_inside)
                    j_inside_1 = j_inside_phi[0]
                    j_inside_2 = j_inside_phi[1]
                    inside_sum += c[j_inside] * x_1**j_inside_1 * x_2**j_inside_2
                
                inside_frac = (self.k_const * self.a_1 * np.sin(x_1)) / (1 + self.k_const * self.a_2 * np.cos(x_1))
                
                sum_1 = sum_1 + w_1 * w_2 * self.a_1 * np.cos(inside_sum) * x_1**(i_1 + j_1) * x_2**(i_2 + j_2)
                sum_2 = sum_2 + w_1 * w_2 * self.a_2 * np.sin(inside_sum) * inside_frac * x_1**(i_1 + j_1) * x_2**(i_2 + j_2)
        return - sum_1 - sum_2

    def F(self, c_vec, d_vec, m_vec, n_vec):
        '''
        Function that takes a vector of coefficients in each dimension, and outputs the function F evaluated on that vector.
        
        Parameters:
            - c_vec: vector of coefficients in dimension 1
            - d_vec: vector of coefficients in dimension 2
            - m_vec: vector of coefficients in dimension 3
            - n_vec: vector of coefficients in dimension 4

        Returns: 
            The function F = (F1, F2, F2, F4)
        '''
        return np.concatenate((self.F1(c_vec,m_vec), self.F2(d_vec,n_vec), self.F3(c_vec,m_vec), self.F4(n_vec)))
    
    def JF(self, c_vec):
        '''
        Takes a vector of coefficients in the first dimension, and outputs the Jacobian JF of the function F evaluated on that vector.

        Parameters:
            - c_vec: vector of coefficients in dimension 1
        
        Returns: 
            THe Jacobian JF
        '''
        zero_mat = np.zeros((self.ndim, self.ndim))
        dG_ret = np.zeros((self.ndim,self.ndim))

        for i in range(self.ndim):
            for j in range(self.ndim):
                dG_ret[i][j] = self.dG_func(c_vec, i, j)
        
        JF = np.block([[self.A, zero_mat, -self.M, zero_mat], [zero_mat, self.A, zero_mat, -self.M], [dG_ret, zero_mat, self.A, zero_mat], [zero_mat, zero_mat, zero_mat, self.A]])    
        return JF
    
    '''
    The following functions get the four dimensions of the function F and aid in its calculation.
    '''
    def F1(self, c_vec, m_vec):
        retval = np.matmul(self.A, c_vec) - np.matmul(self.M, m_vec)
        return retval

    def F2(self, d_vec, n_vec):
        retval = np.matmul(self.A, d_vec) - np.matmul(self.M, n_vec)
        return retval
        
    def F3(self, c_vec, m_vec):
        G_ret = np.zeros(self.ndim)
        for i in range(self.ndim):
            G_ret[i] = self.G_func(c_vec, i)
        
        retval = np.matmul(self.A,m_vec) + G_ret
        return retval

    def F4(self, n_vec):
        retval = np.matmul(self.A, n_vec) - self.gamma
        return retval







