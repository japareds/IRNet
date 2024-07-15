#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 16:00:53 2023

@author: jparedes
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import pickle
import matplotlib.pyplot as plt
import time
import warnings
import os
# =============================================================================
# Synthetic Network Dataset Generator
# =============================================================================

class SyntheticNetwork():
    def __init__(self,n_clusters,n_stations,seed=0):
        """
        Initialize number of clusters and locations in the interval [0,1)

        Parameters
        ----------
        n_clusters : int
            number of clusters in the network
        n_stations: int
            number of reference stations in the network
        seed : float, optional
            pseudo-random number generator seed. The default is 0.

        Returns
        -------
        self.n_clusters: int
            class number of clusters
        slef.n_stations:int
            class number of reference stations in the network
        self.cluster_centers: numpy array
            coordinates of cluster centers
        """
        
        self.rng = np.random.default_rng(seed=seed)
        self.n_clusters = n_clusters
        self.n_stations = n_stations
        #self.cluster_centers = np.array(self.rng.random(size=(n_clusters,2)))
        self.cluster_centers = {el:i for el,i in zip(range(self.n_clusters),np.array(self.rng.random(size=(self.n_clusters,2))))}
        
    def stations_locations(self):
        """
        Create stations coordinates from number of clusters

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        p = self.rng.uniform(low=0.1,high=0.9,size=self.n_clusters)
        self.percentage_stations_cluster = p/p.sum()
        #self.n_stations_cluster = (self.n_stations*self.percentage_stations_cluster).round()
        self.n_stations_cluster = {el:int(i) for el,i in zip(range(self.n_clusters),(self.n_stations*self.percentage_stations_cluster).round())}
        
        # consistency check
        if np.sum([i for i in self.n_stations_cluster.values()]) != self.n_stations:
           warnings.warn(f'Total number of generated stations ({(self.n_stations_cluster).sum()}) doesnt match user specified total number of stations ({self.n_stations})')
         
        #noise = [self.rng.uniform(low=-1.,high=1.,size=int(self.n_stations_cluster[i])) for i in range(self.n_stations_cluster.shape[0])]
        radial_noise = {el:self.rng.uniform(low=-0.5,high=0.5,size=self.n_stations_cluster[el]) for el in range(self.n_clusters)}
        angle_noise = {el:self.rng.uniform(low=0,high=2*np.pi,size=self.n_stations_cluster[el]) for el in range(self.n_clusters)}
        
        #locations = [[cluster_centers[j] + noise[j][i] for i in range(noise[j].shape[0])] for j in range(self.n_clusters)]
        
        locations = []
        for i in range(self.n_clusters):
            locations.append([ np.array([[np.cos(angle_noise[i][j]),-np.sin(angle_noise[i][j])],[np.sin(angle_noise[i][j]),np.cos(angle_noise[i][j])]]) @ (self.cluster_centers[i] + radial_noise[i][j]).reshape((2,1)) for j in range(radial_noise[i].shape[0])])
            
            #locations.append(np.array([self.cluster_centers[i] + radial_noise[i][j] for j in range(self.radial_noise[i].shape[0])]))
            
        self.locations = np.vstack(locations).reshape((self.n_stations,2))
        self.locations = (self.locations - self.locations.min()) / (self.locations.max() - self.locations.min())
        
    def compute_distance(self):
        """
        Compute pairwise l2 distance

        Returns
        -------
        None.

        """
        distances = squareform(pdist(self.locations))
        self.distances = np.exp(-distances ** 2)
        
    def compute_laplacian_matrix(self,threshold=0.4):
        """
        Compute Graph matrices:
            - adjacency
            - degree
            - laplacian
            - covariance

        Parameters
        ----------
        threshold : flaot, optional
            distance threshold set to zero. The default is 0.4.

        Returns
        -------
        None.

        """
        self.adjacency = np.where(self.distances < threshold, 0, self.distances)
        self.degree = np.diag(np.sum(self.adjacency, axis=1))
        self.laplacian = self.degree - self.adjacency
        self.covariance_matrix = np.linalg.pinv(self.laplacian)
        
        
    def generate_signal(self,n_years = 10,n_measurements_day = 24):
        """
        Generate synthetic signal

        Parameters
        ----------
        n_years : int, optional
            number of years of data. The default is 10.
        n_measurements_day : int, optional
            number of measurements per day. The default is 24 (1hr frequency).

        Returns
        -------
        None.

        """
        N_DAYS_YEAR = 365
        AMPLITUDE = 1
        PHASE_M = 0.8
        PHASE_A =  0.7
        
        self.n_years = n_years
        self.n_measurements_day = n_measurements_day
        self.total_measurements = n_years * N_DAYS_YEAR * n_measurements_day
        
        # mean values
        mean_M0 = self.rng.uniform(low=55, high=65, size=self.n_stations)
        mean_A0 = self.rng.uniform(low=10, high=35, size=self.n_stations)
        mean_phi = np.zeros(self.n_stations)
        
        # Generate random samples
        M0_samples = self.rng.multivariate_normal(mean_M0, self.covariance_matrix, self.total_measurements)
        A0_samples = self.rng.multivariate_normal(mean_A0, self.covariance_matrix, self.total_measurements)
        phi_samples = self.rng.multivariate_normal(mean_phi, self.covariance_matrix, self.total_measurements)
        
        # Generate time series data
        time_series_data = np.zeros((self.n_stations, self.total_measurements))
        D = 0
        time_init = time.time()
        for i in range(self.n_stations):
            for j in range(self.total_measurements):
                
                t = j % n_measurements_day  # Get the time within a day (0-23)
                D += 1 if t == 0 else 0  # Increase D by 1 when a new day starts
                M0 = M0_samples[j, i] * (AMPLITUDE + PHASE_M* np.sin(2 * np.pi * D / N_DAYS_YEAR))
                A0 = A0_samples[j,i] * (AMPLITUDE + PHASE_A * np.sin(2 * np.pi * D / N_DAYS_YEAR))
                phi = phi_samples[j, i]
                noise = 0.5 * self.rng.normal(0, 1)  # Generate white noise
                time_series_data[i, j] = M0 + A0 * np.sin((2 * np.pi * t / n_measurements_day) + 0.1*phi) + noise
        time_end = time.time()
        print(f'Dataset generated in {time_end-time_init:.2f} seconds')
        self.snapshots_matrix = time_series_data.copy()
        
    def format_dataset(self):
        """
        Format snapshots matrix to match main code format.
        The measurements simulate measurements starting at 2011-01-01 00:00:00

        Returns
        -------
        None.

        """
        first_year = 2011
        last_year = first_year + self.n_years -1
        if self.n_measurements_day == 24:
            freq = '1H'
            
        start_date = f'{first_year}-01-01 00:00:00'
        end_date = f'{last_year}-12-31 23:00:00'
        range_date = pd.date_range(start_date,end_date,freq=freq)
        # remove leap days
        range_date = range_date[~((range_date.month==2) & (range_date.day==29))]
        
        self.dataset = pd.DataFrame(self.snapshots_matrix.T,index=range_date)
        
        
    def plot_signal(self):
        """
        Plot network and signal

        Returns
        -------
        None.

        """
        
        # scatter plot locations
        fig = plt.figure(figsize=(3.5,2.5))
        ax = fig.add_subplot(111)
        ax.scatter(self.locations[:,0],self.locations[:,1])
        ax.set_title('Synthetic stations spatial distribution')
        
        # first signal plot
        fig = plt.figure(figsize=(3.5,2.5))
        ax = fig.add_subplot(111)
        ax.plot(self.snapshots_matrix[0,:])
        ax.plot(self.snapshots_matrix[-1,:])
        ax.set_title('First and last sensors whole period measurements')
        
        # 2-days pattern
        fig = plt.figure(figsize=(3.5,2.5))
        ax = fig.add_subplot(111)
        ax.plot(self.snapshots_matrix[0,:48])
        ax.plot(self.snapshots_matrix[-1,:48])
        ax.set_title('First sensor 2-days pattern')
        
    def save_dataset(self,fname,save_path):
        self.dataset.to_csv(save_path+fname)
        print(f'Saving dataset in {save_path+fname}')
        fname = f'{save_path}coordinates.csv'
        np.savetxt(fname,self.locations,delimiter=',')

#%%

if __name__ == '__main__':
    abs_path = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/files/synthetic/'
    
    # location_variables
    n_clusters = 3
    n_stations = 100
    locations_seed = 0
    distance_threshold = 0.4
    # signal variables
    n_years = 12
    n_measurements_day = 24
    
    show_plot = True
    save_dataset = True
    print(f'Synthetic dataset generator:\n -number of years:{n_years}\n- measurements per day:{n_measurements_day}\n -number of clusters:{n_clusters}\n -number of stations:{n_stations}\n -distance threshold for adjacency matrix:{distance_threshold}')
    # generate synthetic dataset
    synthetic_data = SyntheticNetwork(n_clusters, n_stations,seed=locations_seed)
    synthetic_data.stations_locations()
    synthetic_data.compute_distance()
    synthetic_data.compute_laplacian_matrix(threshold=distance_threshold)
    synthetic_data.generate_signal(n_years = n_years,n_measurements_day = n_measurements_day)
    synthetic_data.format_dataset()

    print(f'Generated dataset measurements has {synthetic_data.dataset.shape[0]} measurements for {synthetic_data.dataset.shape[1]} locations. Measurements from {synthetic_data.dataset.index[0]} until {synthetic_data.dataset.index[-1]}:\n{synthetic_data.dataset.head()}')
    if save_dataset:
        fname = f'SyntheticData_{synthetic_data.dataset.index[0].year}-{synthetic_data.dataset.index[0].month}-{synthetic_data.dataset.index[0].day}_{synthetic_data.dataset.index[-1].year}-{synthetic_data.dataset.index[-1].month}-{synthetic_data.dataset.index[-1].day}.csv'
        synthetic_data.save_dataset(fname, save_path)

    if show_plot:
        synthetic_data.plot_signal()
        plt.show()
    
    
    
    
    