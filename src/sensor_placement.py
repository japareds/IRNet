#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 17:21:04 2023

@author: jparedes
"""
import numpy as np
import cvxpy as cp
from scipy import linalg
import pickle
import time
import warnings
import sys


"""
=============================================================================
Sensor placement algorithms
=============================================================================
"""
class SensorsErrors(Exception):
    pass

class SensorPlacement:
    def __init__(self,algorithm:str='rankMax',n:int=100,s:int=10,n_refst:int=10,n_lcs:int=0,n_unmonitored:int=0,var_refst:float=1e-2,var_lcs:int=1):

        """
        Network and sensor placement parameters

        Args:
            algorithm (str): sensor placement algorithm to use
            n (int): number of potential locations
            s (int): signal sparsity
            n_refst (int): number of reference stations
            n_lcs (int): number of low-cost sensors (LCSs)
            n_unmonitored (int): number of unmonitored locations
            var_refst (float): reference stations devices noise variance
            var_lcs (vloat): LCSs devices noise variance
        """
        self.algorithm = algorithm
        self.n = n
        self.s = s
        self.n_refst = n_refst
        self.n_lcs = n_lcs
        self.n_unmonitored = n_unmonitored
        self.var_refst = var_refst
        self.var_lcs = var_lcs

    """ Sensor placement algorithms """

    def JB_placement(self,Psi):
            """
            Simple D-optimal sensor placement for single class of sensors

            Parameters
            ----------
            Psi : np array
                Reduced basis of shape (number of locations, number of vectors)
            alpha : float
                regularization parameter for rank minimization constraint

            Returns
            -------
            None.

            """
            h = cp.Variable(shape=self.n,value=np.zeros(self.n))
            objective = -1*cp.log_det(cp.sum([h[i]*Psi[i,:][None,:].T@Psi[i,:][None,:] for i in range(self.n)]))
            constraints = [cp.sum(h) == self.n_refst + self.n_lcs,
                        h >=0,
                        h <= 1,
                        ]
            problem = cp.Problem(cp.Minimize(objective),constraints)


            if not problem.is_dcp():
                print('Problem not dcp\nCheck constraints and objective function')
            self.h = h
            self.problem = problem
    
    def rankMax_placement(self,Psi,alpha,substract=False):
        """
        Sensor placement proposed heuristics for unmonitored locations (p_empty !=0)
        Maximize the rank for reference stations locations while minimizing volume of ellipsoid for LCSs

        Parameters
        ----------
        Psi : np array
            Reduced basis of shape (number of locations, number of vectors)
        alpha : float
            regularization parameter for rank minimization constraint

        Returns
        -------
        None.

        """

        h_lcs = cp.Variable(shape=Psi.shape[0],value=np.zeros(Psi.shape[0]))
        h_refst = cp.Variable(shape = Psi.shape[0],value=np.zeros(Psi.shape[0]))
        objective_eps = -1*cp.log_det(cp.sum([h_lcs[i]*Psi[i,:][None,:].T@Psi[i,:][None,:] for i in range(Psi.shape[0])]))
        if substract:
            print('Objective function for trace will be negative: logdet(LCS) - alpha*Tr(RefSt)')
            objective_zero = -alpha*cp.trace(cp.sum([h_refst[i]*Psi[i,:][None,:].T@Psi[i,:][None,:] for i in range(Psi.shape[0])]))
        else:
            objective_zero = alpha*cp.trace(cp.sum([h_refst[i]*Psi[i,:][None,:].T@Psi[i,:][None,:] for i in range(Psi.shape[0])]))

        objective = objective_eps + objective_zero
        constraints = [cp.sum(h_refst) == self.n_refst,
                       cp.sum(h_lcs) == self.n_lcs,
                       h_refst >=0,
                       h_refst <= 1,
                       h_lcs >=0,
                       h_lcs <=1,
                       h_lcs + h_refst <=1]
        problem = cp.Problem(cp.Minimize(objective),constraints)
        
        if not problem.is_dcp():
            print('Problem not dcp\nCheck constraints and objective function')
        self.h_lcs = h_lcs
        self.h_refst = h_refst
        self.problem = problem

   

    def multiClass_joshiBoyd_placement(self,Psi):
        """
        D-optimal sensor placement for two classes of sensors with variances var_zero and var_eps
        Maximize logdet of precision matrix

        Parameters
        ----------
        Psi : np array
            Reduced basis of shape (number of locations, number of vectors)

        Returns
        -------
        None.

        """
        h_refst = cp.Variable(shape = self.n,value=np.zeros(self.n))
        h_lcs = cp.Variable(shape=self.n,value=np.zeros(self.n))

        if self.n_lcs != 0 and self.n_refst != 0: # there are LCSs and ref st.
            objective = -1*cp.log_det(cp.sum([h_lcs[i]*(self.var_lcs**-1)*Psi[i,:][None,:].T@Psi[i,:][None,:] + h_refst[i]*(self.var_refst**-1)*Psi[i,:][None,:].T@Psi[i,:][None,:] for i in range(self.n)]))
            constraints = [cp.sum(h_refst) == self.n_refst,
                           cp.sum(h_lcs) == self.n_lcs,
                           h_refst >=0,
                           h_refst <= 1,
                           h_lcs >=0,
                           h_lcs <=1,
                           h_lcs + h_refst <=1]


        elif self.n_lcs == 0 and self.n_refst != 0:# there are no LCSs
            objective = -1*cp.log_det(cp.sum([h_refst[i]*(self.var_refst**-1)*Psi[i,:][None,:].T@Psi[i,:][None,:] for i in range(self.n)]))
            constraints = [cp.sum(h_refst) == self.n_refst,
                           h_refst >=0,
                           h_refst <= 1,
                           ]


        elif self.n_lcs != 0 and self.n_refst == 0:# there are no ref.st.
            objective = -1*cp.log_det(cp.sum([h_lcs[i]*(self.var_lcs**-1)*Psi[i,:][None,:].T@Psi[i,:][None,:] for i in range(self.n)]))
            constraints = [cp.sum(h_lcs) == self.n_lcs,
                           h_lcs >=0,
                           h_lcs <= 1,
                           ]


        problem = cp.Problem(cp.Minimize(objective),constraints)
        if not problem.is_dcp():
            print('Problem not dcp\nCheck constraints and objective function')
        self.h_lcs = h_lcs
        self.h_refst = h_refst
        self.problem = problem

    def networkPlanning_singleclass(self,Psi,rho):
        h = cp.Variable(shape = self.n,value=np.zeros(self.n))
        objective = cp.norm(h,1)
        constraints = [h >=0,
                       h <= 1,
                       cp.sum(h) >= self.s,
                       ]
        In = np.identity(self.n)
        for i in range(self.n):
            e_i = In[:,i]
            r1 = cp.hstack([Psi.T@cp.diag(h)@Psi,Psi.T@e_i[:,None]])
            r2 = cp.hstack([e_i[:,None].T@Psi,np.array([rho])[:,None]])
            M_i = cp.vstack([r1,r2])
            constraints += [M_i >> 0]

        problem = cp.Problem(cp.Minimize(objective),constraints)


        if not problem.is_dcp():
            print('Problem not dcp\nCheck constraints and objective function')
        self.h = h
        self.problem = problem

    def networkPlanning_singleclass_iterative(self,Psi,rho,w,locations_monitored=[],locations_unmonitored=[]):
        In = np.identity(self.n)
        if len(locations_monitored)!=0:
            h_init = np.zeros(self.n)
            h_init[locations_monitored] = 1
            h = cp.Variable(shape = self.n,value=h_init) 
        else:
            h = cp.Variable(shape = self.n,value=np.zeros(self.n))
        objective = w.T@h
        constraints = [h >=0,
                    h <= 1,
                    cp.sum(h) >= self.s
                    ]
        if len(locations_monitored) !=0:
            constraints += [h[locations_monitored]==1]
        if len(locations_unmonitored) != 0:
            constraints += [h[locations_unmonitored]==0]

        for i in range(self.n):
            e_i = In[:,i]
            r1 = cp.hstack([Psi.T@cp.diag(h)@Psi,Psi.T@e_i[:,None]])
            r2 = cp.hstack([e_i[:,None].T@Psi,np.array([rho])[:,None]])
            M_i = cp.vstack([r1,r2])
            constraints += [M_i >> 0]
        problem = cp.Problem(cp.Minimize(objective),constraints)

        if not problem.is_dcp():
            print('Problem not dcp\nCheck constraints and objective function')
        self.h = h
        self.problem = problem
        
        
    # other algorithms
    def WCS_placement(self,Psi):
        """
        Worst case scenario sensor placement algorithm
        Solve PSD to minimize maximum diagonal entry of covariance matrix

        Parameters
        ----------
        Psi : np array
            reduced basis


        Returns
        -------
        None.
        self.problem added to class

        """
        # compute covariance matrix
        h_eps = cp.Variable(shape=self.n,value=np.zeros(self.n))
        h_zero = cp.Variable(shape = self.n,value=np.zeros(self.n))
        t = cp.Variable(shape=(1,1))

        #S = np.zeros((self.r,self.r))
        # precision matrix as sum of both: LCSs and ref.st
        # for i in range(self.n):
        #     psi = Psi[i,:][None,:]
        #     S+=h_eps[i]*(var_eps**-1)*psi.T@psi + h_zero[i]*(var_zero**-1)*psi.T@psi
        if self.p_eps != 0:
            S = cp.sum([h_eps[i]*(self.var_eps**-1)*Psi[i,:][None,:].T@Psi[i,:][None,:] + h_zero[i]*(self.var_zero**-1)*Psi[i,:][None,:].T@Psi[i,:][None,:] for i in range(self.n)])
        else:
            S = cp.sum([h_zero[i]*(self.var_zero**-1)*Psi[i,:][None,:].T@Psi[i,:][None,:] for i in range(self.n)])

        constraints = [t>=0,
                       cp.sum(h_zero)==self.p_zero,
                       h_zero >= 0,
                       h_zero <=1]

        if self.p_eps !=0:
            constraints += [cp.sum(h_eps)==self.p_eps,
                            h_eps >= 0,
                            h_eps <=1,
                            h_eps + h_zero <=1]


        Ir = np.identity(self.r)
        # PSD constraints: r block matrices
        for j in np.arange(self.r):
            constraints += [cp.bmat([[t,Ir[:,j][:,None].T],[Ir[:,j][:,None],S]]) >> 0]
        problem = cp.Problem(cp.Minimize(t),constraints)
        if not problem.is_dcp():
            print('Problem not dcp\nCheck constraints and objective function')

        self.h_eps = h_eps
        self.h_zero = h_zero
        self.problem = problem

    def Liu_placement_uncorr(self,Psi):
        """
        Liu-based sensor placement for weakly-correlated noise sensors

        Parameters
        ----------
        Psi : numpy array
            low-rank basis

        Returns
        -------
        None.

        """
        R = np.identity(self.n)
        diag = np.concatenate((np.repeat(self.var_zero**-1,self.p_zero),
                               np.repeat(self.var_eps**-1,self.p_eps),
                               np.repeat(0,self.p_empty)))
        np.fill_diagonal(R, diag)
        h = cp.Variable(shape=self.n,value = np.zeros(self.n))
        H = cp.Variable(shape=(self.n,self.n),value=np.zeros((self.n,self.n)))

        F_mat = Psi.T@(cp.multiply(H, R))@Psi

        Z = cp.Variable(shape=(self.r,self.r))
        Ir = np.identity(self.r)
        M = cp.bmat([F_mat,Ir],[Ir,Z])

        objective = cp.trace(Z)


        constraints = [M >= 0,
                       cp.trace(H)<= self.p_zero + self.p_eps,
                       cp.diag(H) == h,
                       cp.bmat([ [H,h[:,None]],[h[:,None].T,np.ones((1,1))] ] )>= 0]


        problem = cp.Problem(cp.Minimize(objective),constraints)
        if not problem.is_dcp():
            print('Problem not dcp\nCheck constraints and objective function')
        self.problem = problem
        self.h = h


# =============================================================================
#        Problem initialization
# =============================================================================


    def check_consistency(self):
        """
        Check number of sensors consistency

        Raises
        ------
        SensorsErrors
            Error if the proposed configuration is not correct

        Returns
        -------
        None.

        """

        # check number of sensors consistency
        # if self.p_zero >self.r and self.p_eps!=0:
        #     raise SensorsErrors(f'The number of reference stations is larger than the dimension of the low-rank basis {self.r}')
        if self.n_unmonitored + self.s > self.n:
            raise SensorsErrors('Not enough sensors for basis sampling')
        elif any(np.array([self.n_lcs,self.n_refst,self.n_lcs])<0):
            raise SensorsErrors('Negative number of sensors')
        elif all(np.array([self.n_lcs,self.n_refst,self.n_unmonitored])>=0) and np.sum([self.n_refst,self.n_lcs,self.n_unmonitored]) != self.n:
            raise SensorsErrors('Number of sensors and empty locations mismatch total number of possible locations')
        else:
            print(f'No initialization errors found.\nNetwork summary:\n- network size: {self.n}\n- signal sparsity: {self.s}\n- number of reference stations: {self.n_refst}\n- number of LCSs: {self.n_lcs}\n- number of unmonitored locations: {self.n_unmonitored}\n- reference staions variance: {self.var_refst}\n- LCSs variance: {self.var_lcs}')

    def initialize_problem(self,Psi:np.ndarray,alpha:float=0.,rho:float=100.,w = cp.Variable(),locations_monitored:list = [],locations_unmonitored:list = []):
        """
        Initializes sensor placement problem

        Args:
            Psi (numpy array): low-rank basis size: (num_locations,sparsity)
            alpha (float, optional): rankMax regularization hyperparameter. Defaults to 0.
            rho (float, optional): network planning sensor placement accuracy threshold
        """

        self.check_consistency()
        
        algorithms = ['rankMax','MCJB','JB','NetworkPlanning','NetworkPlanning_iterative']

        if self.algorithm not in algorithms:
            print(f'Sensor placement algorithm {self.algorithm} not implemented yet')

        elif self.algorithm == 'rankMax':
            print(f'Setting {self.algorithm} sensor placement algorithm with regularization value {alpha}')
            if self.n_unmonitored == 0:
                self.rankMax_placement(Psi,alpha)
            elif self.n_lcs == 0: # no LCS - use Doptimal
                self.var_refst = 1e0
                self.multiClass_joshiBoyd_placement(Psi)
            else:
                self.rankMax_placement(Psi,alpha)

        elif self.algorithm == 'MCJB':
            print(f'Setting {self.algorithm} sensor placement algorithm')
            self.multiClass_joshiBoyd_placement(Psi)
        
        elif self.algorithm == 'JB':
            print(f'Setting {self.algorithm} sensor placement algorithm')
            self.JB_placement(Psi)

        elif self.algorithm == 'NetworkPlanning':
            print(f'Setting {self.algorithm} sensor placement algorithm')
            self.networkPlanning_singleclass(Psi,rho)
        
        elif self.algorithm == 'NetworkPlanning_iterative':
            print(f'Setting {self.algorithm} sensor placement algorithm')
            self.networkPlanning_singleclass_iterative(Psi,rho,w,locations_monitored,locations_unmonitored)

        # others
        elif self.algorithm == 'WCS':
            self.WCS_placement(Psi)

        elif self.algorithm == 'rankMax_FM':
            self.simple_Dopt_placement(Psi)
            self.Psi_phase1 = Psi.copy()
            self.alpha_phase2 = alpha

        elif self.algorithm == 'Dopt-Liu':
            self.Liu_placement_uncorr(Psi)


    """Solve sensor problem"""

    def LoadLocations(self,dicts_path,alpha_reg,var_zero):
        """
        Load locations from previous training

        Parameters
        ----------
        dicts_path : str
            path to files
        alpha_reg : flaot
            regularization value
        var_zero : float
            refst covariance

        Returns
        -------
        self.dict_results : dict
            (LCSs, ref.st,empty) locations

        """
        if self.algorithm == 'rankMax' or self.algorithm == 'rankMax_FM' :
            fname = dicts_path+f'DiscreteLocations_{self.algorithm}_vs_p0_{self.n}N_r{self.r}_pEmpty{self.p_empty}_alpha{alpha_reg:.1e}.pkl'
            with open(fname,'rb') as f:
                self.dict_locations = pickle.load(f)

            fname = dicts_path+f'Weights_{self.algorithm}_vs_p0_{self.n}N_r{self.r}_pEmpty{self.p_empty}_alpha{alpha_reg:.1e}.pkl'
            with open(fname,'rb') as f:
                self.dict_weights = pickle.load(f)
        else:
            fname = dicts_path+f'DiscreteLocations_{self.algorithm}_vs_p0_{self.n}N_r{self.r}_pEmpty{self.p_empty}_varZero{var_zero:.1e}.pkl'
            with open(fname,'rb') as f:
                self.dict_locations = pickle.load(f)

            fname = dicts_path+f'Weights_{self.algorithm}_vs_p0_{self.n}N_r{self.r}_pEmpty{self.p_empty}_varZero{var_zero:.1e}.pkl'
            with open(fname,'rb') as f:
                self.dict_weights = pickle.load(f)

    def solve(self,n_it=50):
        """
        Solve sensor placement problem and print objective function optimal value
        and sensors weights

        Returns
        -------
        None.

        """
        time_init = time.time()
        print(f'Solving sensor placement using {self.algorithm} strategy')

        # some special steps for special algorithms
        if self.algorithm == 'WCS':
            solver = cp.MOSEK
            try:
                self.problem.solve(verbose=True,solver=solver)
                if self.problem.status not in ["infeasible", "unbounded"]:
                    print(f'Optimal value: {self.problem.value}')
            except:
                print('Problem not solved.\nIt is possible solver failed or problem status is unknown.')

        elif self.algorithm == 'rankMax_FM':
            try:
                # phase 1) solve simple D-optimal to determine unmonitored locations
                print('Solving basic D-optimal problem')
                self.problem_1.solve(verbose=True)
                if self.problem_1.status not in ["infeasible", "unbounded"]:
                    print(f'Optimal value: {self.problem_1.value}')
                # phase2) solve fully monitored rankMax to distribute sensors
                self.loc_unmonitored = np.sort(np.argsort(self.h.value)[:self.p_empty])
                self.loc_monitored = np.sort(np.argsort(self.h.value)[-(self.p_zero + self.p_eps):])
                Psi_restricted = self.Psi_phase1[self.loc_monitored,:]
                self.rankMax_placement(Psi_restricted, self.alpha_phase2)
                self.problem.solve(verbose=True)
                print('Solving rankMax for sensor distribution')
                if self.problem.status not in ["infeasible", "unbounded"]:
                    print(f'Optimal value: {self.problem.value}')

            except:
                print('Problem not solved.\nIt is possible solver failed or problem status is unknown.')
        
        # general solving process
        else:
            try:
                self.problem.solve(verbose=True)
                if self.problem.status not in ["infeasible", "unbounded"]:
                    print(f'Optimal value: {self.problem.value}')
            except:
                print('Problem not solved.\nIt is possible solver failed or problem status is unknown.')

        time_end = time.time()
        self.exec_time = time_end - time_init
        print(f'Optimization problem solved in {self.exec_time:.2f}s')


    def discretize_solution(self):
        """
        Convert continuous weights [0,1] to discrete values {0,1}
        Get maximum entries of h and use those locations to place sensors

        Returns
        -------
        None.

        """
        
        if self.algorithm == 'rankMax_FM':
            order_refst = self.loc_monitored[np.sort(np.argsort(self.h_zero.value)[-self.n_refst:])]
            order_lcs = np.sort(np.setdiff1d(self.loc_monitored,order_refst))
            order_unmonitored = np.sort(self.loc_unmonitored)
        
        elif self.algorithm == 'JB':
            order_lcs = []
            order_refst = np.sort(np.argsort(self.h.value)[-self.n_refst:])
            order_unmonitored = np.sort([i for i in np.arange(self.n) if i not in order_refst])
        
        elif self.algorithm == 'NetworkPlanning':
            self.n_refst = int(np.round(self.h.value.sum()))
            order_lcs = []
            order_refst = np.sort(np.argsort(self.h.value)[-self.n_refst:])
            order_unmonitored = np.sort([i for i in np.arange(self.n) if i not in order_refst])

        else:
            if self.h_lcs.value.sum() == 0.0 and self.h_refst.value.sum() == 0.0: # placement failure
                print('Sensor placement failed. Setting to zero')
                order_refst, order_lcs = np.zeros(self.n), np.zeros(self.n)
            elif self.n_refst == 0:# no ref.st.
                print('There are no Reference stations in the network. Placing LCSs')
                order_refst = []
                order_lcs = np.sort(np.argsort(self.h_lcs.value)[-self.n_lcs:])
            elif self.n_lcs == 0:# no LCSs
                print('There are no LCSs in the network. Placing reference stations')
                order_refst= np.sort(np.argsort(self.h_refst.value)[-self.n_refst:])
                order_lcs = []
            else:
                order_refst = np.sort(np.argsort(self.h_refst.value)[-self.n_refst:])
                order_lcs = np.sort([i for i in np.flip(np.argsort(self.h_lcs.value)) if i not in order_refst][:self.n_lcs])

            order_unmonitored = np.sort([i for i in np.arange(self.n) if i not in order_lcs and i not in order_refst])

            if (set(order_lcs).issubset(set(order_refst)) or set(order_refst).issubset(set(order_lcs))) and (self.n_lcs!=0 and self.n_refst!=0) :
                print('Some locations of LCSs are already occupied by a reference station')
                order_lcs = [i for i in np.arange(self.n) if i not in order_refst]

        self.locations = [order_lcs,order_refst,order_unmonitored]
        print(f'Monitoring locations:\n- reference stations ({len(self.locations[1])} locations in total): {self.locations[1]}\n- LCSs ({len(self.locations[0])} locations in total): {self.locations[0]}\n- unmonitored locations ({len(self.locations[2])} locations in total): {self.locations[2]}')
        

    def compute_Doptimal(self,Psi,alpha):
        """
        Computes D-optimal metric of continuous index obtained from rankMin_reg solution

        Parameters
        ----------
        Psi : numpy array
            low-rank basis.
        alpha : float
            reference station rank regularization parameter

        Returns
        -------
        None.

        """
        precision_matrix = (self.var_eps**-1)*Psi.T@np.diag(self.h_eps.value)@Psi + (self.var_zero**-1)*Psi.T@np.diag(self.h_zero.value)@Psi
        self.Dopt_metric = -1*np.log(np.linalg.det(precision_matrix))

        self.logdet_eps = np.log(np.linalg.det(Psi.T@np.diag(self.h_eps.value)@Psi))
        self.trace_zero = alpha*np.trace(Psi.T@np.diag(self.h_zero.value)@Psi)





# =============================================================================
#         Compute covariance matrix and regressor from solutions
# =============================================================================


    def C_matrix(self):
        """
        Convert indexes of LCSs, RefSt and Emtpy locations into
        measurement matrix C

        Returns
        -------
        self.C: list
            C matrix for LCSs, RefSt, Empty


        """
        # select certain rows from identity matrix
        In = np.identity(self.n)
        C_lcs = In[self.locations[0],:]
        C_refst = In[self.locations[1],:]

        if self.n_unmonitored != 0:
            C_unmonitored = In[self.locations[2],:]
        else:
            C_unmonitored = []
        self.C = [C_lcs,C_refst,C_unmonitored]

    def compute_convex_covariance_matrix(self,Psi,weights,var_zero,var_eps,metric='logdet'):
        C_eps = np.diag(weights[0])
        C_refst = np.diag(weights[1])
        Theta_eps = C_eps@Psi
        Theta_refst = C_refst@Psi

        Precision_matrix = (var_eps**(-1)*Theta_eps.T@Theta_eps) + (var_zero**(-1)*Theta_refst.T@Theta_refst)
        try:
            self.Cov_convex = np.linalg.inv(Precision_matrix)
        except:
            print('Computing pseudo-inverse')
            self.Cov_convex = np.linalg.pinv(Precision_matrix)

        if metric=='logdet':
            self.metric_convex = np.log(np.linalg.det(self.Cov_convex))


    def covariance_matrix(self,Psi,metric='logdet',alpha=0.1,activate_error_solver=True):
        """
        Compute covariance matrix from C matrices and compute a metric

        Parameters
        ----------
        Psi : np array
            low-rank basis used
        metric : str
            metric to compute covariance optimizality: D-optimal, E-optimal, WCS
        alpha: float
            regularization parameter for proposed algorithm

        Returns
        -------
        None.

        """
        C_eps = self.C[0]
        C_zero = self.C[1]

        Theta_eps = C_eps@Psi
        Theta_zero = C_zero@Psi
        if self.p_eps !=0:
            self.Precision_matrix = (self.var_eps**(-1)*Theta_eps.T@Theta_eps) + (self.var_zero**(-1)*Theta_zero.T@Theta_zero)
            try:
                self.Cov = np.linalg.inv( self.Precision_matrix )
            except:
                print('Computing pseudo-inverse')
                self.Cov = np.linalg.pinv( self.Precision_matrix )

        else:
            self.Precision_matrix = (self.var_zero**(-1)*Theta_zero.T@Theta_zero)
            try:
                self.Cov = np.linalg.inv(self.Precision_matrix)
            except:
                print('Computing pseudo-inverse')
                self.Cov = np.linalg.pinv(self.Precision_matrix)

        # compute cov-matrix metric

        if metric == 'logdet':
            self.metric = np.log(np.linalg.det(self.Cov))
            self.metric_precisionMatrix = np.log(np.linalg.det(self.Precision_matrix))

        elif metric == 'eigval':
            self.metric = np.max(np.real(np.linalg.eig(self.Cov)[0]))
            self.metric_precisionMatrix = np.min(np.real(np.linalg.eig(self.Precision_matrix)[0]))

        elif metric == 'WCS':
            self.metric = np.diag(self.Cov).max()

        elif metric == 'logdet_rank':
            self.metric = np.log(np.linalg.det(self.var_eps**(-1)*Theta_eps.T@Theta_eps)) + alpha*np.trace(self.var_zero**(-1)*Theta_zero.T@Theta_zero)

        if activate_error_solver:
            if type(self.problem.value) == type(None):# error when solving
                self.metric = np.inf
                self.metric_precisionMatrix = -np.inf

    def covariance_matrix_GLS(self,Psi):
        """
        Compute covariance matrix from GLS. Use pseudo-inverse to account for unstable behavior

        Parameters
        ----------
        Psi : numpy array
            low-rank basis

        Returns
        -------
        None.

        """
        C_lcs = self.C[0]
        C_refst = self.C[1]
        Theta_lcs = C_lcs@Psi
        Theta_refst = C_refst@Psi

        if C_lcs.shape[0] == 0:# no LCSs
            Precision_matrix = (self.var_zero**-1)*Theta_refst.T@Theta_refst
        elif C_refst.shape[0] == 0:#no Ref.St.
            Precision_matrix = (self.var_eps**-1)*Theta_lcs.T@Theta_lcs
        else:
            Precision_matrix = (self.var_zero**-1)*Theta_refst.T@Theta_refst + (self.var_eps**-1)*Theta_lcs.T@Theta_lcs

        try:
            S = np.linalg.svd(Precision_matrix)[1]
            rcond_pinv = rcond_pinv = (S[-1]+S[-2])/(2*S[0])
            Cov = np.linalg.pinv(Precision_matrix,rcond_pinv)
        except:
            Cov = np.linalg.pinv(Precision_matrix,hermitian=True)

        self.Cov = Cov

    def covariance_matrix_limit(self,Psi):
        """
        Compute covariance matrix in the limit var_zero = 0

        Parameters
        ----------
        Psi : numpy array
            low-rank basis


        Returns
        -------
        None.

        """
        C_lcs = self.C[0]
        C_refst = self.C[1]
        Theta_lcs = C_lcs@Psi
        Theta_refst = C_refst@Psi


        if C_lcs.shape[0] == 0:#no LCSs
            self.Cov = np.zeros(shape=(self.r,self.r))
            return
        elif C_refst.shape[0] == 0:#no RefSt
            Precision_matrix = (self.var_eps**-1)*Theta_lcs.T@Theta_lcs
            S = np.linalg.svd(Precision_matrix)[1]
            rcond_pinv = rcond_pinv = (S[-1]+S[-2])/(2*S[0])
            self.Cov = np.linalg.pinv( Precision_matrix,rcond_pinv)

            return

        else:
            # compute covariance matrix using projector
            refst_matrix = Theta_refst.T@Theta_refst

            Is = np.identity(self.r)
            try:
                P = Is - refst_matrix@np.linalg.pinv(refst_matrix)
                #P = Is - refst_matrix@np.linalg.pinv(refst_matrix,hermitian=True,rcond=1e-10)
            except:
                P = Is - refst_matrix@np.linalg.pinv(refst_matrix,hermitian=True,rcond=1e-10)

            rank1 = np.linalg.matrix_rank(Theta_lcs@P,tol=1e-10)
            rank2 = np.linalg.matrix_rank(P@Theta_lcs.T,tol=1e-10)

            S1 = np.linalg.svd(Theta_lcs@P)[1]
            S2 = np.linalg.svd(P@Theta_lcs.T)[1]

            if rank1==min((Theta_lcs@P).shape):
                try:
                    rcond1_pinv = (S1[-1]+S1[-2])/(2*S1[0])
                except:
                    rcond1_pinv = 1e-15
            else:
                rcond1_pinv = (S1[rank1]+S1[rank1-1])/(2*S1[0])

            if rank2==min((P@Theta_lcs.T).shape):
                try:
                    rcond2_pinv = (S2[-1]+S2[-2])/(2*S2[0])
                except:
                    rcond2_pinv = 1e-15
            else:
                rcond2_pinv = (S2[rank2]+S2[rank2-1])/(2*S2[0])

            self.Cov = self.var_eps*np.linalg.pinv(Theta_lcs@P,rcond=rcond1_pinv)@np.linalg.pinv(P@Theta_lcs.T,rcond=rcond2_pinv)
            #self.Cov = self.var_eps*P@np.linalg.pinv(Theta_lcs.T@Theta_lcs,hermitian=True,rcond=1e-10)@P




    def beta_estimated_GLS(self,Psi,y_refst,y_lcs):
        """
        Compute estimated regressor (beta) from sensor measurements

        Parameters
        ----------
        Psi : numpy array
            sparse basis
        y_refst : numpy array
            reference stations vector of measurements
        y_lcs : numpy array
            LCSs vector of measurements

        Returns
        -------
        self.beta_hat : numpy array
                estimated regressor
                Regressor evolution over time (r,num_samples)

        """
        C_eps = self.C[0]
        C_zero = self.C[1]
        Theta_eps = C_eps@Psi
        Theta_zero = C_zero@Psi
        second_term = (self.var_zero**-1)*Theta_zero.T@y_refst + (self.var_eps**-1)*Theta_eps.T@y_lcs
        self.beta_hat = self.Cov@second_term


    def beta_estimated_limit(self,Psi,y_refst,y_lcs):
        """
        Compute estimated regressor (beta) from sensor measurements
        in the limit variances refst goes to zero (limit of GLS)

        Parameters
        ----------
        Psi : numpy array
            sparse basis
        y_refst : numpy array
            reference stations vector of measurements
        y_lcs : numpy array
            LCSs vector of measurements

        Returns
        -------
        self.beta_hat : numpy array
                estimated regressor over time (r,num_samples)

        """
        C_lcs = self.C[0]
        C_refst = self.C[1]
        Theta_lcs = C_lcs@Psi
        Theta_refst = C_refst@Psi
        refst_matrix = Theta_refst.T@Theta_refst

        Is = np.identity(self.r)
        P = Is - refst_matrix@np.linalg.pinv(refst_matrix)

        term_refst = np.linalg.pinv(Theta_refst)
        term_lcs = np.linalg.pinv(Theta_lcs@P)@np.linalg.pinv(P@Theta_lcs.T)@Theta_lcs.T

        self.beta_hat = term_lcs@y_lcs + term_refst@y_refst


    def save_locations(self,files_path):
        """
        Save generated list of locations. Solutions to sensor placement problem for a given basis.
        """
        fname = f'{files_path}SensorsLocations_N{self.n}_RefSt{self.n_refst}_LCSs{self.n_lcs}_Sparsity{self.s}.pkl'
        with open(fname,'wb') as f:
            pickle.dump(self.locations, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Sensor placement locations results saved at: {fname}')

if __name__ == '__main__':
    """
    Network parameters:
        - number of sensors per class
        - signal sparsity
        - variance of sensors
    """
    N = 44
    SIGNAL_SPARSITY = 20
    N_LCS, N_REFST = 0,44
    N_UNMONITORED = N - N_REFST - N_LCS
    VAR_LCS, VAR_REFST = 1,1e-2

    """ Create test measurement matrix, snapshots matrix and low-rank basis """
    rng = np.random.default_rng(seed=40)
    NUM_SAMPLES = int(1e4)
    X = rng.random((NUM_SAMPLES,N))
    snapshots_matrix = X.T
    snapshots_matrix_centered = snapshots_matrix - snapshots_matrix.mean(axis=1)[:,None]
    U,S,Vt = np.linalg.svd(snapshots_matrix,full_matrices=False)
    Psi = U[:,:SIGNAL_SPARSITY]

    fully_monitored_network_worst_accuracy = np.diag(Psi@np.linalg.inv(Psi.T@Psi)@Psi.T).max()
    deployed_network_accuracy_threshold = 1.5*fully_monitored_network_worst_accuracy
    
    """ Solve network planning problem """
    algorithm = 'NetworkPlanning_iterative'
    sensor_placement = SensorPlacement(algorithm, N, SIGNAL_SPARSITY,N_REFST,N_LCS,N_UNMONITORED)
    
    if algorithm == 'NetworkPlanning_iterative':
        # algorithm initialization
        h_prev = np.zeros(N)
        epsilon = 1e-3
        w = 1/(h_prev+epsilon)
        n_it = 3
        it = 0
        locations_monitored = []
        locations_unmonitored = []

        # iterative method
        time_init = time.time()
        while len(locations_monitored) + len(locations_unmonitored) != N:
            # solve sensor placement with constraints
            sensor_placement.initialize_problem(Psi,rho=deployed_network_accuracy_threshold,
                                                w=w,locations_monitored=locations_monitored,locations_unmonitored=locations_unmonitored)
            sensor_placement.solve()
            # update sets
            locations_monitored += [i[0] for i in np.argwhere(sensor_placement.h.value >= 1-epsilon) if i[0] not in locations_monitored]
            locations_unmonitored += [i[0] for i in np.argwhere(sensor_placement.h.value <= epsilon) if i[0] not in locations_unmonitored]
            # check convergence
            if np.linalg.norm(sensor_placement.h.value - h_prev)<=epsilon or it==n_it:
                locations_monitored += [[i for i in np.argsort(sensor_placement.h.value)[::-1] if i not in locations_monitored][0]]
                it = 0
            h_prev = sensor_placement.h.value
            w = 1/(h_prev + epsilon)
            it +=1
            print(f'{len(locations_monitored)} Locations monitored: {locations_monitored}\n{len(locations_unmonitored)} Locations unmonitored: {locations_unmonitored}\n')
        time_end = time.time()
        sensor_placement.locations = [[],np.sort(locations_monitored),np.sort(locations_unmonitored)]
        sensor_placement.C_matrix()

    elif algorithm == 'NetworkPlanning':
        sensor_placement.initialize_problem(Psi,rho=deployed_network_accuracy_threshold)
        time_init = time.time()
        sensor_placement.solve()
        sensor_placement.discretize_solution()
        time_end = time.time()
        sensor_placement.C_matrix()
    deployed_network_accuracy = np.diag(Psi@np.linalg.inv(Psi.T@sensor_placement.C[1].T@sensor_placement.C[1]@Psi)@Psi.T).max()
    print(f'Discretized network solution:\n- Number of potentiaql locations: {sensor_placement.n}\n- Number of monitored locations: {sensor_placement.locations[1].shape}\n- Number of unmonitored locations: {sensor_placement.locations[2].shape}\n- Fully monitoring network maximum signal variance: {fully_monitored_network_worst_accuracy:.2f}\n- Deployed monitoring network threshold: {deployed_network_accuracy_threshold:.2f}\n- Deployed monitoring network maximum variance: {deployed_network_accuracy:.2f}')
    print("\u0332".join(f"Finished in {time_end-time_init:.2f}s"))
    
    sys.exit()