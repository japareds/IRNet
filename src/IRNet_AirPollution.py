#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 17:21:04 2023

@author: jparedes
"""
import os
import time
import pandas as pd
import geopy.distance
from sklearn.model_selection import train_test_split
import numpy as np
import sys
import warnings
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from geopandas import GeoDataFrame

import sensor_placement as sp


""" Obtain signal sparsity and reconstruct signal at different temporal regimes"""

# perturbate measurements
def add_noise_signal(X:pd.DataFrame,seed:int=92,var:float=1.)->pd.DataFrame:
    """
    Add noise to measurements dataset. The noise ~N(0,var).
    The noise is the same for all sensors during all the time.

    Args:
        X (pd.DataFrame): dataset with measurements
        seed (int): random number generator seed
        var (float): noise variance

    Returns:
        pd.DataFrame: _description_
    """
    rng = np.random.default_rng(seed=seed)
    noise = rng.normal(loc=0.0,scale=var,size=X.shape)
    X_noisy = X + noise
    #X_noisy[X_noisy<0] = 0.
    return X_noisy

def define_rois_distance(distances:pd.Series,distance_thresholds:list,n_regions:int)-> dict: 
    """
    Generates Regions of Interest (ROIs) based on distance from certain station

    Args:        
        distances (pd.Series): distance of each location from origin station
        distance_thresholds (list): thresholds for each ROI
        n_regions (int): number of ROIs

    Raises:
        ValueError: Check if number of specified distance thresholds matches number of ROIs

    Returns:
        dict: Indices of each ROI. Key specifies the distance threshold
    """
    print(f'Determining indices that belong to each ROI. {n_regions} regions with thresholds: {distance_thresholds}')
    if type(distance_thresholds) is not list:
        distance_thresholds = [distance_thresholds]
    if len(distance_thresholds) != n_regions:
        raise ValueError(f'Number of distance thresholds: {distance_thresholds} mismatch specified number of regions: {n_regions}')
    roi_idx = {el:[] for el in distance_thresholds}
    #distance_thresholds = np.insert(distance_thresholds,0,0)
    for i in range(len(distance_thresholds[:-1])):
        print(f'Distance threshold between {distance_thresholds[i]} and {distance_thresholds[i+1]}')
        stations = [j for j in distances[np.logical_and(distances>=distance_thresholds[i],distances<distance_thresholds[i+1])].index]
        print(f'Stations ({len(stations)}): {stations}')
        idx_stations = np.where(np.isin(distances.index,stations))[0]
        roi_idx[distance_thresholds[i]] = idx_stations
    stations = [j for j in distances[distances>=distance_thresholds[-1]].index]
    print(f'Stations with a distance larger than {distance_thresholds[-1]} ({len(stations)}): {stations}')
    idx_stations = np.where(np.isin(distances.index,stations))[0]
    roi_idx[distance_thresholds[-1]] = idx_stations

    return roi_idx

def define_rois_variance(coordinate_error_variance_fullymonitored:list,variance_thresholds:list,n_regions:int)->dict:
    print(f'Determining indices that belong to each ROI. {n_regions} regions with thresholds: {variance_thresholds}')
    if type(variance_thresholds) is not list:
        variance_thresholds = [variance_thresholds]
    if len(variance_thresholds) != n_regions:
        raise ValueError(f'Number of variance thresholds: {variance_thresholds} mismatch specified number of regions: {n_regions}')
    roi_idx = {el:[] for el in variance_thresholds}
    for i in range(len(variance_thresholds[:-1])):
        print(f'Variance threshold between {variance_thresholds[i]} and {variance_thresholds[i+1]}')
        stations = [j for j in coordinate_error_variance_fullymonitored[np.logical_and(coordinate_error_variance_fullymonitored>=variance_thresholds[i],coordinate_error_variance_fullymonitored<variance_thresholds[i+1])]]
        print(f'{len(stations)} stations')
        idx_stations = np.where(np.isin(coordinate_error_variance_fullymonitored,stations))[0]
        roi_idx[variance_thresholds[i]] = idx_stations
    stations = [j for j in coordinate_error_variance_fullymonitored[coordinate_error_variance_fullymonitored>=variance_thresholds[-1]]]
    print(f'{len(stations)} stations with a distance larger than {variance_thresholds[-1]}')
    idx_stations = np.where(np.isin(coordinate_error_variance_fullymonitored,stations))[0]
    roi_idx[variance_thresholds[-1]] = idx_stations
    
    return roi_idx
        





    

# signal reconstruction functions
def signal_reconstruction_svd(U:np.ndarray,snapshots_matrix_train:np.ndarray,snapshots_matrix_val_centered:np.ndarray,X_val:pd.DataFrame,s_range:np.ndarray) -> pd.DataFrame:
    """
    Decompose signal keeping s-first singular vectors using training set data
    and reconstruct validation set.

    Args:
        U (numpy array): left singular vectors matrix
        snapshots_matrix_train (numpy array): snaphots matrix of training set data
        snapshots_matrix_val_centered (numpy array): snapshots matrix of validation set data
        X_val (pandas dataframe): validation dataset
        s_range (numpy array): list of sparsity values to test

    Returns:
        rmse_sparsity: dataframe containing reconstruction errors at different times for each sparsity threshold in the range
    """
    print(f'Determining signal sparsity by decomposing training set and reconstructing validation set.\nRange of sparsity levels: {s_range}')
    rmse_sparsity = pd.DataFrame()
    for s in s_range:
        # projection
        Psi = U[:,:s]
        snapshots_matrix_val_pred_svd = (Psi@Psi.T@snapshots_matrix_val_centered) + snapshots_matrix_train.mean(axis=1)[:,None]
        X_pred_svd = pd.DataFrame(snapshots_matrix_val_pred_svd.T)
        X_pred_svd.columns = X_val.columns
        X_pred_svd.index = X_val.index
        
        #RMSE across different signal measurements
        rmse = pd.DataFrame(np.sqrt(((X_val - X_pred_svd)**2).mean(axis=1)),columns=[s],index=X_val.index)
        rmse_sparsity = pd.concat((rmse_sparsity,rmse),axis=1)
    return rmse_sparsity

def signal_reconstruction_regression(Psi:np.ndarray,locations_measured:np.ndarray,X_test:pd.DataFrame,X_test_measurements:pd.DataFrame=[],snapshots_matrix_train:np.ndarray=[],snapshots_matrix_test_centered:np.ndarray=[],projected_signal:bool=False)->pd.DataFrame:
    """
    Signal reconstyruction from reduced basis measurement.
    The basis Psi and the measurements are sampled at indices in locations_measured.
    Compute reconstruction error


    Args:
        Psi (np.ndarray): low-rank basis
        locations_measured (np.ndarray): indices of locations measured
        X_test (pd.DataFrame): testing dataset which is measured and used for error estimation
        X_test_measurements (pd.DataFrame): testing dataset measurements projected onto subspace spanned by Psi
        snapshots_matrix_train (np.ndarray): training set snapshots matrix used for computing average
        snapshots_matrix_val_centered (np.ndarray): testing set centered snapshots matrix used for signal reconstruction
        

    Returns:
        rmse (pd.DataFrame): mean reconstruction error between validation data set and reconstructed data
        error_max (pd.DataFrame): max reconstruction error when comparing validation data with reconstructed data
    """
    # basis measurement
    n_sensors_reconstruction = len(locations_measured)
    C = np.identity(Psi.shape[0])[locations_measured]
    Psi_measured = C@Psi
    # regression
    if projected_signal:
        beta_hat = np.linalg.pinv(Psi_measured)@X_test_measurements.iloc[:,locations_measured].T
        snapshots_matrix_predicted = Psi@beta_hat
    else:
        beta_hat = np.linalg.pinv(Psi_measured)@snapshots_matrix_test_centered[locations_measured,:]
        snapshots_matrix_predicted_centered = Psi@beta_hat
        snapshots_matrix_predicted = snapshots_matrix_predicted_centered + snapshots_matrix_train.mean(axis=1)[:,None]
    # compute prediction
    X_pred = pd.DataFrame(snapshots_matrix_predicted.T)
    X_pred.columns = X_test.columns
    X_pred.index = X_test.index
    # compute error metrics
    error = X_test - X_pred
    rmse = pd.DataFrame(np.sqrt(((error)**2).mean(axis=1)),columns=[n_sensors_reconstruction],index=X_test.index)
    error_variance = error.var()
    """
    error_max = pd.DataFrame(np.abs(error).max(axis=1),columns=[n_sensors_reconstruction],index=X_test.index)
    error_var = np.zeros(shape = error.shape)
    for i in range(error.shape[0]):
        error_var[i,:] = np.diag(error.iloc[i,:].to_numpy()[:,None]@error.iloc[i,:].to_numpy()[:,None].T)
    error_var = pd.DataFrame(error_var,index=X_test.index,columns=X_test.columns)
    """
    return rmse, error_variance

def hourly_signal_reconstruction(Psi:np.ndarray,X_train:pd.DataFrame,X_val:pd.DataFrame,signal_sparsity:int=1,locations_measured:np.ndarray=[])->dict:
    """
    Compute reconstruction error at different times using low-rank basis
    Args:
        Psi (np.ndarray): monitored low-rank basis
        X_train (pd.DataFrame): training set measurements 
        X_val (pd.DataFrame): validation set measurements
        signal_sparsity (int): sparsity threshold
        locations_measured (np.ndarray): indices of monitored locations

    Returns:
        dict: rmse for multiple measurements at different times
    """
    hours_range = np.sort(X_train.index.hour.unique())
    rmse_time = {el:[] for el in hours_range}
    for h in hours_range:
        # get measurements at certain hour and rearrange as snapshots matrix
        X_train_hour = X_train.loc[X_train.index.hour == h]
        X_val_hour = X_val.loc[X_val.index.hour==h]
        snapshots_matrix_train_hour = X_train_hour.to_numpy().T
        snapshots_matrix_train_hour_centered = snapshots_matrix_train_hour - snapshots_matrix_train_hour.mean(axis=1)[:,None]
        snapshots_matrix_val_hour = X_val_hour.to_numpy().T
        snapshots_matrix_val_hour_centered = snapshots_matrix_val_hour - snapshots_matrix_val_hour.mean(axis=1)[:,None]
        if len(locations_measured) != 0:
            rmse_hour = signal_reconstruction_regression(Psi,locations_measured,snapshots_matrix_train_hour,snapshots_matrix_val_hour_centered,X_val_hour)
        else:# not using sensor placement procedure. Use simple svd reconstruction
            rmse_hour = signal_reconstruction_svd(Psi,snapshots_matrix_train_hour,snapshots_matrix_val_hour_centered,X_val_hour,[signal_sparsity])
        rmse_time[h] = rmse_hour
    return rmse_time

def networkPlanning_iterative(sensor_placement:sp.SensorPlacement,N:int,Psi:np.ndarray,deployed_network_variance_threshold:float,epsilon:float,h_prev:np.ndarray,weights:np.ndarray,n_it:int,locations_monitored:list=[],locations_unmonitored:list=[])->list:
    """
    IRL1 network planning algorithm
    Args:
        sensor_placement (sp.SensorPlacement): sensor placement object containing network information
        N (int): total number of network locations
        deployed_network_variance_threshold (float): error variance threshold for network design
        epsilon (float): IRL1 weights update constant
        h_prev (np.ndarray): network locations initialization
        weights (np.ndarray): IRL1 weights initialization
        n_it (int): IRL1 max iterations
        locations_monitored (list, optional): initialization of set of monitored lcoations. Defaults to [].
        locations_unmonitored (list, optional): initialization of set of unmonitored locaitons. Defaults to [].

    Returns:
        locations (list): indices of monitored and unmonitored locations [S,Sc]
    """
    # iterative method
    it = 0
    time_init = time.time()
    new_monitored = []
    new_unmonitored = []
    while len(locations_monitored) + len(locations_unmonitored) != N:
        # solve sensor placement with constraints
        sensor_placement.initialize_problem(Psi,rho=deployed_network_variance_threshold,
                                            w=weights,locations_monitored=locations_monitored,locations_unmonitored=locations_unmonitored)
        sensor_placement.solve()
        print(f'Problem status: {sensor_placement.problem.status}')
        if sensor_placement.problem.status == 'optimal':
            # update sets with new monitored locations
            new_monitored = [i[0] for i in np.argwhere(sensor_placement.h.value >= 1-epsilon) if i[0] not in locations_monitored]
            new_unmonitored = [i[0] for i in np.argwhere(sensor_placement.h.value <= epsilon) if i[0] not in locations_unmonitored]

            locations_monitored += new_monitored
            locations_unmonitored += new_unmonitored
            # check convergence
            if np.linalg.norm(sensor_placement.h.value - h_prev)<=epsilon or it==n_it:
                locations_monitored += [[i for i in np.argsort(sensor_placement.h.value)[::-1] if i not in locations_monitored][0]]
                it = 0
            h_prev = sensor_placement.h.value
            weights = 1/(h_prev + epsilon)
            it +=1
        else:
            # solver fails at iteration
            #locations_monitored = locations_monitored[:-len(new_monitored)]
            if len(new_unmonitored) != 0:
                locations_unmonitored = locations_unmonitored[:-len(new_unmonitored)]
            it+=1

        print(f'{len(locations_monitored)} Locations monitored: {locations_monitored}\n{len(locations_unmonitored)} Locations unmonitored: {locations_unmonitored}\n')
    time_end = time.time()
    locations = [locations_monitored,locations_unmonitored]
    print(f'IRL1 algorithm finished in {time_end-time_init:.2f}s.')
    return locations

# dataset
class Dataset():
    def __init__(self,pollutant:str='O3',N:int=44,start_date:str='2011-01-01',end_date:str='2022-12-31',files_path:str='',synthetic_dataset:bool=False):
        self.pollutant = pollutant
        self.N = N
        self.start_date = start_date
        self.end_date = end_date
        self.files_path = files_path
        self.synthetic_dataset = synthetic_dataset
    
    def load_dataset(self):
        if self.synthetic_dataset:
            fname = f'{self.files_path}SyntheticData_{self.start_date}_{self.end_date}.csv'
        else:
            fname = f'{self.files_path}{self.pollutant}_catalonia_clean_N{self.N}_{self.start_date}_{self.end_date}.csv'
            self.coordinates = pd.read_csv(f'{self.files_path}coordinates.csv',index_col=0)
            self.coordinates_distances = pd.DataFrame([],index=self.coordinates.index,columns=self.coordinates.index)
            for i in range(self.coordinates.shape[0]):
                for j in range(self.coordinates.shape[0]):
                    self.coordinates_distances.iloc[i,j] = geopy.distance.geodesic(self.coordinates.iloc[i,:],self.coordinates.iloc[j,:]).km

        print(f'Loading dataset from {fname}')
        self.ds = pd.read_csv(fname,sep=',',index_col=0)
        self.ds.index = pd.to_datetime(self.ds.index)
        

    def check_dataset(self):
        print(f'Checking missing values in dataset')
        print(f'Percentage of missing values per location:\n{100*self.ds.isna().sum()/self.ds.shape[0]}')
        print(f'Dataset has {self.ds.shape[0]} measurements for {self.ds.shape[1]} locations.\n{self.ds.head()}')

# figures
class Figures():
    def __init__(self,save_path,figx=3.5,figy=2.5,fs_title=10,fs_label=10,fs_ticks=10,fs_legend=10,marker_size=3,dpi=300,use_grid=False,show_plots=False):
        self.figx = figx
        self.figy = figy
        self.fs_title = fs_title
        self.fs_label = fs_label
        self.fs_ticks = fs_ticks
        self.fs_legend = fs_legend
        self.marker_size = marker_size
        self.dpi = dpi
        self.save_path = save_path
        if show_plots:
            self.backend = 'Qt5Agg'
        else:
            self.backend = 'Agg'
        
        print('Setting mpl rcparams')
        
        font = {'weight':'normal',
                'size':str(self.fs_label),
                }
        
        lines = {'markersize':self.marker_size}
        
        fig = {'figsize':[self.figx,self.figy],
               'dpi':self.dpi
               }
        
        ticks={'labelsize':self.fs_ticks
            }
        axes={'labelsize':self.fs_ticks,
              'grid':False,
              'titlesize':self.fs_title
            }
        if use_grid:
            grid = {'alpha':0.5}
            mpl.rc('grid',**grid)
        
        mathtext={'default':'regular'}
        legend = {'fontsize':self.fs_legend}
        
        mpl.rc('font',**font)
        mpl.rc('figure',**fig)
        mpl.rc('xtick',**ticks)
        mpl.rc('ytick',**ticks)
        mpl.rc('axes',**axes)
        mpl.rc('legend',**legend)
        mpl.rc('mathtext',**mathtext)
        mpl.rc('lines',**lines)        
        mpl.use(self.backend)

    def curve_timeseries_singlestation(self,X:pd.DataFrame,station_name:str,date_init:str='2020-01-20',date_end:str='2021-10-27'):
        date_range = pd.date_range(start=date_init,end=date_end,freq='H')
        date_idx = [i for i in date_range if i in X.index]
        data = X.loc[date_idx,[station_name]]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(data)
        ax.set_xlabel('date')
        ax.set_ylabel('Concentration ($\mu$g/$m^3$)')
        fig.tight_layout()

    def curve_timeseries_allstations(self,X:pd.DataFrame,date_init:str='2020-01-20',date_end:str='2021-10-27',save_fig=True):
        date_range = pd.date_range(start=date_init,end=date_end,freq='H')
        date_idx = [i for i in date_range if i in X.index]
        data = X.loc[date_idx]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.fill_between(x=data.index,y1=np.percentile(X,axis=1,q=25),y2=np.percentile(X,axis=1,q=75))
        ax.set_xlabel('date')
        ax.set_ylabel('O$_3$ ($\mu$g/$m^3$)')
        fig.tight_layout()

        if save_fig:
            fname = self.save_path+'timeseries_Allstations.png'
            fig.savefig(fname,dpi=300,format='png',bbox_inches='tight')
            print(f'Figure saved at {fname}')

    
    def curve_timeseries_dailypattern_singlestation(self,X:pd.DataFrame,station_name:str):
        X_ = X.loc[:,station_name].copy()
        data = X_.groupby(X_.index.hour).median()
        q1,q3 = X_.groupby(X_.index.hour).quantile(q=0.25),X_.groupby(X_.index.hour).quantile(q=0.75)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(data)
        ax.fill_between(x=data.index,y1=q1,y2=q3,alpha=0.5)
        ax.set_xlabel('hour')
        yrange = np.arange(0,110,10)
        ax.set_yticks(yrange)
        ax.set_yticklabels([i for i in ax.get_yticks()])
        ax.set_ylabel('O$_3$ ($\mu$g/$m^3$)')
        ax.set_ylim(0,100)
        fig.tight_layout()
    
    def curve_timeseries_dailypattern_multiplestations(self,X:pd.DataFrame,stations_locs:list=[0,1,2,3],save_fig:bool=False):
        stations_names = [i for i in X.columns[stations_locs]]
        colors = ['#1a5276','orange','#117864','#943126']
        X_ = X.iloc[:,stations_locs].copy()
        data = X_.groupby(X_.index.hour).median()
        q1,q3 = X_.groupby(X_.index.hour).quantile(q=0.25),X_.groupby(X_.index.hour).quantile(q=0.75)

        
        fig = plt.figure()
        curves = {}
        for i in range(len(stations_locs)):
            ax = fig.add_subplot(221+i)
            curves[i] = ax.plot(data.iloc[:,i],label=stations_names[i],color=colors[i])
            ax.fill_between(x=data.index,y1=q1.iloc[:,i],y2=q3.iloc[:,i],alpha=0.5,color=colors[i])
            yrange = np.arange(0,110,10)
            ax.set_yticks(yrange)
            ax.set_yticklabels([i for i in ax.get_yticks()])    
            if (221+i)%2 == 1:
                ax.set_ylabel('O$_3$ ($\mu$g/$m^3$)')
            ax.set_ylim(0,100)
            if i in [2,3]:
                ax.set_xlabel('hour')

        handles = [curves[i][0] for i in curves.keys()]
        fig.legend(handles=[i for i in handles],ncol=2,bbox_to_anchor=(0.95,1.15),framealpha=1)
        fig.tight_layout()

        if save_fig:
            fname = f'{self.save_path}Curve_TimeSeriesHourly_ManyStations.png'
            fig.savefig(fname,dpi=300,format='png',bbox_inches='tight')
            print(f'Figure saved into {fname}')
        
    def curve_timeseries_dailypattern_allstations(self,X:pd.DataFrame):
        X_ = pd.DataFrame()
        for c in X.columns:
            X_ = pd.concat((X_,X.loc[:,c]),axis=0)
        X_ = X_.loc[:,0]
        data = X_.groupby(X_.index.hour).median()
        q1,q3 = X_.groupby(X_.index.hour).quantile(q=0.25),X_.groupby(X_.index.hour).quantile(q=0.75)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(data)
        ax.fill_between(x=data.index,y1=q1,y2=q3,alpha=0.5)
        ax.set_xlabel('hour')
        yrange = np.arange(0,110,10)
        ax.set_yticks(yrange)
        ax.set_yticklabels([i for i in ax.get_yticks()])
        ax.set_ylabel('O$_3$ ($\mu$g/$m^3$)')
        ax.set_ylim(0,100)
        fig.tight_layout()

    def boxplot_measurements(self,X,save_fig):
        n = X.shape[1]
        yrange = np.arange(0.0,300,50)
        xrange = np.arange(1,n+1,1)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        bp = ax.boxplot(x=X,notch=False,vert=True,
                   whis=1.5,bootstrap = None,
                   positions=[i for i in range(len(xrange))],widths=0.5,labels=[str(i) for i in xrange],
                   flierprops={'marker':'.','markersize':1},
                   patch_artist=True)
        
        ax.set_yticks(yrange)
        ax.set_yticklabels([np.round(i,2) for i in ax.get_yticks()])
        ax.set_ylabel('O$_3$ ($\mu$g/$m^3$)')
        
        xrange = [i-1 for i in xrange if i%5==0]
        ax.set_xticks(xrange)
        ax.set_xticklabels([int(i+1) for i in xrange],rotation=0)
        ax.set_xlabel('Location index')
        fig.tight_layout()
        if save_fig:
            fname = self.save_path+'boxplot_concentration_allStations.png'
            fig.savefig(fname,dpi=300,format='png',bbox_inches='tight')
            print(f'Figure saved at {fname}')

    def geographical_network_visualization(self,map_path:str,df_coordinates:pd.DataFrame,locations_monitored:np.array=[],roi_idx:dict={},show_legend:bool=False,save_fig:bool=False)->plt.figure:
        """
        Figure showing the geographical area where sensors are deployed along with coordinates of reference stations

        Args:
            map_path (str): path to map file
            df_coordinates (pd.DataFrame): dataframe containing coordiantes(Latitude,Longitude) of each reference stations
            locations_monitored (np.array, optional): indices of monitored locations. Defaults to [].
            roi_idx (dict): dictionary indicating indices that belong to each region of interest (ROI) in case of heterogeneous design. The keys correspond to parameter used for separating ROIs.
            show_legend (bool, optional): Show legend indicating monitored and unmonitored locations. Defaults to False.
            save_fig (bool, optional): save generated figure. Defaults to False.

        Returns:
            plt.figure: Figure with map and stations 
        """
        
        if len(locations_monitored)!=0:
            df_coords_monitored = df_coordinates.iloc[locations_monitored]
            df_coords_unmonitored = df_coordinates.iloc[[i for i in range(df_coordinates.shape[0]) if i not in locations_monitored]]
            geometry_monitored = [Point(xy) for xy in zip(df_coords_monitored['Longitude'], df_coords_monitored['Latitude'])]
            geometry_unmonitored = [Point(xy) for xy in zip(df_coords_unmonitored['Longitude'], df_coords_unmonitored['Latitude'])]
            gdf_monitored = GeoDataFrame(df_coords_monitored, geometry=geometry_monitored)
            gdf_unmonitored = GeoDataFrame(df_coords_unmonitored, geometry=geometry_unmonitored)

        else:
            df_coords_monitored = df_coordinates.copy()
            geometry_monitored = [Point(xy) for xy in zip(df_coords_monitored['Longitude'], df_coords_monitored['Latitude'])]
            gdf_monitored = GeoDataFrame(df_coords_monitored, geometry=geometry_monitored)
        
        spain = gpd.read_file(f'{map_path}ll_autonomicas_inspire_peninbal_etrs89.shp')
        catalonia = spain.loc[spain.NAME_BOUND.str.contains('Catalunya')]
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        geo_map = catalonia.plot(ax=ax,color='#117a65')
        
        try:
            if len(roi_idx)!=0:
                markers = ['o','^']
                for idx,m in zip(roi_idx.values(),markers):
                    
                    locations_monitored_roi = np.array(locations_monitored)[np.isin(locations_monitored,idx)]

                    df_coords_monitored = df_coordinates.iloc[[i for i in range(df_coordinates.shape[0]) if i in locations_monitored_roi]]
                    geometry_monitored = [Point(xy) for xy in zip(df_coords_monitored['Longitude'], df_coords_monitored['Latitude'])]
                    gdf_monitored = GeoDataFrame(df_coords_monitored, geometry=geometry_monitored)
                    gdf_monitored.plot(ax=geo_map, marker=m, color='#943126', markersize=6,label=f'Monitoring node')
                    
                    df_coords_unmonitored = df_coordinates.iloc[[i for i in range(df_coordinates.shape[0]) if i in idx and i not in locations_monitored_roi]]
                    geometry_unmonitored = [Point(xy) for xy in zip(df_coords_unmonitored['Longitude'], df_coords_unmonitored['Latitude'])]
                    gdf_unmonitored = GeoDataFrame(df_coords_unmonitored, geometry=geometry_unmonitored)
                    gdf_unmonitored.plot(ax=geo_map, marker=m, color='k', markersize=6,label=f'Unmonitored locations')    
                
            else:
                gdf_monitored.plot(ax=geo_map, marker='o', color='#943126', markersize=6,label=f'Monitoring node')
                gdf_unmonitored.plot(ax=geo_map, marker='o', color='k', markersize=6,label=f'Unmonitored locations')
        except:
            print('No unmonitored locations')
        ax.set_xlim(0.0,3.5)
        ax.set_ylim(40.5,43)
        
        ax.set_ylabel('Latitude (degrees)')
        ax.set_xlabel('Longitude (degrees)')

        if show_legend:
            ax.legend(loc='center',ncol=2,framealpha=1,bbox_to_anchor=(0.5,1.2))
        ax.tick_params(axis='both', which='major')
        fig.tight_layout()
        
        if save_fig:
            fname = self.save_path+f'Map_PotentialLocations_MonitoredLocations{df_coords_monitored.shape[0]}.png'
            fig.savefig(fname,dpi=300,format='png',bbox_inches='tight')
            print(f'Figure saved at {fname}')
        return fig
        

    # Low-rank plots
    def singular_values_cumulative_energy(self,sing_vals,n,synthetic_dataset=False,save_fig=False):
        """
        Plot sorted singular values ratio and cumulative energy

        Parameters
        ----------
        sing_vals : numpy array
            singular values
        n : int
            network size
        save_fig : bool, optional
            save generated figures. The default is False.

        Returns
        -------
        None.

        """
        cumulative_energy = np.cumsum(sing_vals)/np.sum(sing_vals)
        xrange = np.arange(0,sing_vals.shape[0],1)
        fig1 = plt.figure()
        ax = fig1.add_subplot(111)
        ax.plot(xrange,cumulative_energy,color='#1f618d',marker='o')
        ax.set_xticks(np.concatenate(([0.0],np.arange(xrange[9],xrange[-1]+1,10))))
        ax.set_xticklabels([int(i+1) for i in ax.get_xticks()])
        ax.set_xlabel('$i$th singular value')
        
        #yrange = np.arange(0.5,1.05,0.05)
        yrange = np.arange(0.,1.2,0.2)
        ax.set_yticks(yrange)
        ax.set_yticklabels([np.round(i,2) for i in ax.get_yticks()])
        ax.set_ylabel('Cumulative energy')
        if synthetic_dataset:
            ax.set_yscale('log')
        fig1.tight_layout()
        
        fig2 = plt.figure()
        ax = fig2.add_subplot(111)
        ax.plot(xrange, sing_vals / np.max(sing_vals),color='#1f618d',marker='o')
        ax.set_xticks(np.concatenate(([0.0],np.arange(xrange[9],xrange[-1]+1,10))))
        ax.set_xticklabels([int(i+1) for i in ax.get_xticks()],rotation=0)
        ax.set_xlabel('$i$th singular value')

        yrange = np.logspace(-4,0,5)
        ax.set_yticks(yrange)
        ax.set_ylabel('Normalized singular values')
        ax.set_ylim(1e-2,1)
        ax.set_yscale('log')
        if synthetic_dataset:
            ax.set_yscale('log')
        fig2.tight_layout()
        
        if save_fig:
            fname = self.save_path+f'Curve_sparsity_cumulativeEnergy_N{n}.png'
            fig1.savefig(fname,dpi=300,format='png')
            print(f'Figure saved at: {fname}')

            fname = self.save_path+f'Curve_sparsity_singularValues_N{n}.png'
            fig2.savefig(fname,dpi=300,format='png')
            print(f'Figure saved at: {fname}')
         
    def boxplot_validation_rmse_svd(self,rmse_sparsity,n,max_sparsity_show=10,synthetic_dataset=False,save_fig=False) -> plt.figure:
        yrange = np.arange(0.0,35,5)
        xrange = rmse_sparsity.columns[:max_sparsity_show]
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        bp = ax.boxplot(x=rmse_sparsity.iloc[:,:max_sparsity_show],notch=False,vert=True,
                   whis=1.5,bootstrap = None,
                   positions=[i for i in range(len(xrange))],widths=0.5,labels=[str(i) for i in xrange],
                   flierprops={'marker':'.','markersize':1},
                   patch_artist=True)
        
        ax.set_yticks(yrange)
        ax.set_yticklabels([np.round(i,2) for i in ax.get_yticks()])
        if synthetic_dataset:
            ax.set_yscale('log')
            ax.set_ylim(1e-2,1e1)
        else:
            ax.set_ylim(0,30)
        ax.set_ylabel('RMSE ($\mu$g/$m^3$)')
        xrange = np.array([i-1 for i in xrange if i%5==0])
        ax.set_xticks(xrange)
        ax.set_xticklabels([int(i+1) for i in xrange],rotation=0)
        ax.set_xlabel('Sparsity level')
        fig.tight_layout()

        if save_fig:
            fname = self.save_path+f'boxplot_RMSE_SVDreconstruction_validationSet_Smin{xrange.min()}_Smax{xrange.max()}_N{n}.png'
            fig.savefig(fname,dpi=300,format='png')
            print(f'Figure saved in {fname}')
    
        return fig
    
    def boxplot_rmse_comparison(self,rmse_method1:pd.DataFrame,rmse_method2:pd.DataFrame,maxerror:bool=False,save_fig:bool=False)->plt.figure:
        """
        Boxplot comparing validation set RMSE using 2 different numbers of deployed senors.
        E.g: compare fully monitored vs reduced

        Args:
            rmse_method1 (pd.DataFrame): rmse for certain number of sensors
            rmse_method2 (pd.DataFrame): rmse for different number of sensors (for example fully monitored)
            maxerror (bool, optional): dataframes contain maximum reconstruction error instead of RMSE. Defaults to False.
            save_fig (bool, optional): Save generqated figure. Defaults to False.

        Returns:
            plt.figure: Figure
        """
        n_sensors_1 = rmse_method1.columns[0]
        n_sensors_2 = rmse_method2.columns[0]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        bp1 = ax.boxplot(x=rmse_method1,notch=False,vert=True,
                   whis=1.5,bootstrap = None,
                   positions=[0],widths=0.5,labels=[n_sensors_1],
                   flierprops={'marker':'.','markersize':1},
                   patch_artist=True)
        
        bp2 = ax.boxplot(x=rmse_method2,notch=False,vert=True,
                   whis=1.5,bootstrap = None,
                   positions=[1],widths=0.5,labels=[n_sensors_2],
                   flierprops={'marker':'.','markersize':1},
                   patch_artist=True)
        bp1['boxes'][0].set_facecolor('lightgreen')
        bp2['boxes'][0].set_facecolor('#1a5276')
        
        if maxerror:
            yrange = np.arange(0.,55.,5)
            ax.set_ylim(0,50)
        else:
            yrange = np.arange(0.,22.,2)
            ax.set_ylim(0,20)
        ax.set_yticks(yrange)
        ax.set_yticklabels([np.round(i,1) for i in ax.get_yticks()])

        if maxerror:
            ax.set_ylabel('Max error ($\mu$g/$m^3$)')        
        else:
            ax.set_ylabel('RMSE ($\mu$g/$m^3$)')        
        ax.set_xlabel('Number of deployed sensors')
        fig.tight_layout()

        if save_fig:
            if maxerror:
                fname = f'{self.save_path}Maxerrorcomparison_NsensorsTotal_N1{n_sensors_1}_N2{n_sensors_2}.png'
            else:
                fname = f'{self.save_path}RMSEcomparison_NsensorsTotal_N1{n_sensors_1}_N2{n_sensors_2}.png'
            fig.savefig(fname,dpi=300,format='png')
    
        return fig
    
    def boxplot_errorratio(self,df_error1:pd.DataFrame,df_error2:pd.DataFrame,save_fig:bool=False)->plt.figure:
        n_sensors1 = df_error1.columns[0]
        n_sensors2 = df_error2.columns[0]
        df_ratio = df_error1.to_numpy() / df_error2.to_numpy()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        bp = ax.boxplot(x=df_ratio,notch=False,vert=True,
                   whis=1.5,bootstrap = None,
                   positions=[0],widths=0.5,labels=[f'{n_sensors1} sensors vs {n_sensors2} senors'],
                   flierprops={'marker':'.','markersize':1},
                   patch_artist=True)
        
        
        bp['boxes'][0].set_facecolor('#1a5276')
        
        yrange = np.arange(0.,3.5,0.5)
        ax.set_ylim(0,3)
        ax.set_yticks(yrange)
        ax.set_yticklabels([np.round(i,1) for i in ax.get_yticks()])

        ax.set_ylabel('Reconstruction errors ratio')        
        ax.set_xlabel('')
        fig.tight_layout()

        if save_fig:
            fname = f'{self.save_path}ErrorRatio_NsensorsTotal_N1{n_sensors1}_N2{n_sensors2}.png'
            fig.savefig(fname,dpi=300,format='png')
    
        return fig
    
    def hist_worsterror(self,errormax_fullymonitored,errormax_reconstruction,n_sensors,save_fig=False):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(x=errormax_fullymonitored,bins=np.arange(0.,5.1,0.1),density=True,cumulative=False,color='#1a5276',label='Fully monitored network')
        ax.vlines(x=errormax_fullymonitored.mean(),ymin=0.0,ymax=1.0,colors='#1a5276',linestyles='--')
        ax.hist(x=errormax_reconstruction,bins=np.arange(0.,5.1,0.1),density=True,cumulative=False,color='orange',label=f'Reconstruction with {n_sensors} sensors',alpha=0.5)
        ax.vlines(x=errormax_reconstruction.mean(),ymin=0.0,ymax=1.0,colors='orange',linestyles='--')
        ax.set_xlabel('Maximum reconstruction error')
        ax.set_ylabel('Probability density')
        ax.legend(loc='upper left',ncol=1,framealpha=0.5)
        ax.set_xlim(0,5)
        ax.set_ylim(0,1)
        fig.tight_layout()
        if save_fig:
            fname = f'{self.save_path}Histogram_error_fullymonitored_vs_reconstruction_Nsensors{n_sensors}.png'
            fig.savefig(fname,dpi=300,format='png')
            print(f'Figure saved at {fname}')

    def hist_errorratio(self,errormax_fullymonitored,errormax_reconstruction,n_sensors,save_fig=False):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(x=errormax_reconstruction.to_numpy()/errormax_fullymonitored.to_numpy(),bins=np.arange(0,3.1,0.1),density=True,cumulative=False,color='#1a5276')
        ax.set_xlabel('Maximum error ratio')
        ax.set_ylabel('Probability density')
        ax.set_xlim(0,3)
        fig.tight_layout()
        if save_fig:
            fname = f'{self.save_path}Histogram_errorRatio_Nsensors{n_sensors}.png'
            fig.savefig(fname,dpi=300,format='png')
            print(f'Figure saved at {fname}')
    
    def curve_errorvariance_comparison(self,errorvar_fullymonitored:list,errorvar_reconstruction:list,variance_threshold_ratio:float,worst_coordinate_variance_fullymonitored:float,n:int,n_sensors:int,errorvar_reconstruction_Dopt:list=[],roi_idx:dict={},n_sensors_Dopt:int=0,save_fig:bool=False) -> plt.figure:
        """
        Show error variance over a testing set at each network location. 
        The error variance is obtained after reconstructing the signal from p measurements.
        The p measurement locations are obtained from network design algorithm or D-optimality criteria.
        It also shows the threshold line which the network design algorithm used.
        Another algorithm can be shown for comparison.

        Args:
            errorvar_fullymonitored (list): error variance at each network location obtained with a fully monitored network. This corresponds to the lowest error variance possible.
            errorvar_reconstruction (list): error variance at each network locations obtained with a network with a reduced number of deployed sensors.
            variance_threshold_ratio (float): variance threshold ratio used for design algorithm. It is a multiple of the worst_coordinate_variance_fullymonitored.
            worst_coordinate_variance_fullymonitored (float): fully-monitored network worst coordinate error variance
            n (int): total number of network points
            n_sensors (int): number of deployed sensors
            errorvar_reconstruction_Dopt (list): error variance at each network location obtained by D-optimality (or other) criteria. Defaults to [].
            roi_idx (dict): dictionary containing indices of locations that belong to each ROI. The keys indicate the threshold used to separate the network.
            save_fig (bool, optional): Save generated figure. Defaults to False.

        Returns:
            plt.figure: Figure with error variance curves
        """
        if type(variance_threshold_ratio) is float:
            variance_threshold = variance_threshold_ratio*worst_coordinate_variance_fullymonitored
        
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(errorvar_fullymonitored,color='#1d8348',label='Fully monitored network')
            if len(errorvar_reconstruction_Dopt) !=0:
                ax.plot(errorvar_reconstruction_Dopt,color='orange',label=f'Joshi-Boyd solution',alpha=0.8)
            ax.plot(errorvar_reconstruction,color='#1a5276',label=f'Network design solution')
            ax.hlines(y=variance_threshold,xmin=0,xmax=n+1,color='k',linestyles='--',label=rf'Design threshold $\rho$={variance_threshold_ratio:.2f}$\rho_n$')
            xrange = np.arange(-1,n,10)
            xrange[0] = 0
            ax.set_xticks(xrange)
            ax.set_xticklabels([i+1 for i in ax.get_xticks()])
            ax.set_xlim(0,n)
            ax.set_xlabel('Location index')
            yrange = np.arange(0,1.75,0.25)
            ax.set_yticks(yrange)
            ax.set_yticklabels([np.round(i,2) for i in ax.get_yticks()])
            ax.set_ylim(0,1.5)
            ax.set_ylabel('Error variance')
            ax.legend(loc='center',ncol=2,framealpha=0.5,bbox_to_anchor=(0.5,1.1))
            fig.tight_layout()
            if save_fig:
                fname = f'{self.save_path}Curve_errorVariance_Threshold{variance_threshold_ratio:.2f}_Nsensors{n_sensors}.png'
                fig.savefig(fname,dpi=300,format='png')
                print(f'Figure saved at {fname}')


        else:
            variance_threshold = [t*w for t,w in zip(variance_threshold_ratio,worst_coordinate_variance_fullymonitored)]
            # sort coordinate error variance such that the ROIs are shown in order
            coordinate_error_variance_fully_monitored_sorted = np.concatenate([errorvar_fullymonitored[i] for i in roi_idx.values()])
            coordinate_error_variance_design_sorted = np.concatenate([errorvar_reconstruction[i] for i in roi_idx.values()])

            fig = plt.figure()
            ax = fig.add_subplot(111)
            # coordinate error variance at each location
            ax.plot(coordinate_error_variance_fully_monitored_sorted,color='#1d8348',label='Fully monitored network')
            if len(errorvar_reconstruction_Dopt) !=0:
                coordinate_error_variance_Dopt_sorted = np.concatenate([errorvar_reconstruction_Dopt[i] for i in roi_idx.values()])
                ax.plot(coordinate_error_variance_Dopt_sorted,color='orange',label=f'Joshi-Boyd {n_sensors_Dopt} sensors',alpha=0.8)
            ax.plot(coordinate_error_variance_design_sorted,color='#1a5276',label=f'Network design {n_sensors} sensors')
            
            # horizontal lines showing threshold design
            n_roi = np.concatenate([[0],[len(i) for i in roi_idx.values()]])
            n_roi_cumsum = np.cumsum(n_roi)
            for v,l in zip(variance_threshold,range(len(n_roi_cumsum))):
                if l==0:
                    ax.hlines(y=v,xmin=n_roi_cumsum[l]-1,xmax=n_roi_cumsum[l+1]-1,color='k',linestyles='--',label='Design threshold')
                else:
                    ax.hlines(y=v,xmin=n_roi_cumsum[l],xmax=n_roi_cumsum[l+1]-1,color='k',linestyles='--')
            xrange = np.arange(-1,n,10)
            xrange[0] = 0
            ax.set_xticks(xrange)
            ax.set_xticklabels([i+1 for i in ax.get_xticks()])
            ax.set_xlim(-0.5,n)
            ax.set_xlabel('Location index')
            yrange = np.arange(0,2.75,0.5)
            ax.set_yticks(yrange)
            ax.set_yticklabels([np.round(i,2) for i in ax.get_yticks()])
            ax.set_ylim(0,2.5+0.1)
            ax.set_ylabel('Coordinate error variance')
            ax.legend(loc='center',ncol=2,framealpha=0.5,bbox_to_anchor=(0.5,1.2))
            fig.tight_layout()
            if save_fig:
                #fname = f'{self.save_path}Curve_errorVariance_Threshold{variance_threshold_ratio}_Nsensors{n_sensors}_NsensorsDopt{n_sensors_Dopt}_NsensorsROIDopt_{n_sensors_roi}.png'
                fname = f'{self.save_path}Curve_errorVariance_VarThreshold{variance_threshold_ratio}_Nsensors{n_sensors}_NsensorsDopt{n_sensors_Dopt}.png'
                fig.savefig(fname,dpi=300,format='png')
                print(f'Figure saved at {fname}')




    def curve_rmse_hourly(self,rmse_time,month=0,save_fig=False):
        hours = [i for i in rmse_time.keys()]
        median = [rmse_time[i].median().to_numpy()[0] for i in hours]
        q1,q3 = [rmse_time[i].quantile(q=0.25).to_numpy()[0] for i in hours], [rmse_time[i].quantile(q=0.75).to_numpy()[0] for i in hours]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(median,color='#1a5276')
        ax.fill_between(x=hours,y1=q1,y2=q3,color='#1a5276',alpha=0.5)
        ax.set_xticks(hours[::4])
        ax.set_xticklabels([i for i in ax.get_xticks()])
        ax.set_xlabel('Hour')
        yrange = np.arange(0,12.,2.)
        ax.set_yticks(yrange)
        ax.set_yticklabels([np.round(i,1) for i in ax.get_yticks()])
        ax.set_ylabel('RMSE ($\mu$g/$m^3$)')
        ax.set_ylim(yrange[0],yrange[-1])
        fig.tight_layout()
        if save_fig:
            fname = f'{self.save_path}deploy_sensors_hourly_month{month}.png'
            fig.savefig(fname,dpi=300,format='png')
        return fig
    

if __name__ == '__main__':
    """ load dataset to use """
    abs_path = os.path.dirname(os.path.realpath(__file__))
    files_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/files/catalonia/'
    results_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/test/'

    synthetic_dataset = False
    pollutant = 'O3'
    start_date = '2011-01-01'
    end_date = '2022-12-31'
    N=48
    if synthetic_dataset:
        files_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/files/synthetic/'
    else:
        dataset = Dataset(pollutant,N,start_date,end_date,files_path,synthetic_dataset)
        dataset.load_dataset()
        dataset.check_dataset()
    """ sort stations based on distance"""
    dataset.distances = dataset.coordinates_distances.loc['Ciutadella']
    dataset.distances.sort_values(ascending=True,inplace=True)
    dataset.ds = dataset.ds.loc[:,[f'O3_{i}' for i in dataset.distances.index if f'O3_{i}' in dataset.ds.columns]]
    print(f'Order of dataset locations: {dataset.ds.columns}')

    """ snapshots matrix and low-rank decomposition """
    # train/val/test split
    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10
    X_train, X_test = train_test_split(dataset.ds, test_size= 1 - train_ratio,shuffle=False,random_state=92)
    X_val, X_test = train_test_split(X_test, test_size=test_ratio/(test_ratio + validation_ratio),shuffle=False,random_state=92) 
    print(f'Dataset matrix summary:\n {train_ratio} of dataset for training set with {X_train.shape[0]} measurements from {X_train.index[0]} until {X_train.index[-1]}\n {validation_ratio} of dataset for validation set with {X_val.shape[0]} measurements from {X_val.index[0]} until {X_val.index[-1]}\n {test_ratio} of measuerements for testing set with {X_test.shape[0]} measurements from {X_test.index[0]} until {X_test.index[-1]}')
    
    """ Exploratory analysis of signal behavior over time"""
    explore_time_series = False
    if explore_time_series:
        plots = Figures(save_path=results_path,marker_size=1,
                        fs_label=12,fs_ticks=7,fs_legend=6,fs_title=10,
                        show_plots=True)
        print('Checking full data set')
        plots.curve_timeseries_singlestation(X=X_train,station_name='0',date_init=X_train.index[0],date_end = X_train.index[-1])
        plots.curve_timeseries_allstations(X=X_train,date_init=X_train.index[0],date_end=X_train.index[-1],save_fig=False)
        
        print('Checking hourly patter over whole period')
        plots.curve_timeseries_dailypattern_singlestation(X=X_train,station_name='0')
        plots.curve_timeseries_dailypattern_multiplestations(X=X_train,stations_locs=[1,5,24,43],save_fig=False)
        plots.curve_timeseries_dailypattern_allstations(X=X_train)
        plots.boxplot_measurements(X_train,save_fig=False)
        if synthetic_dataset:
            pass
        else:
            plots.geographical_network_visualization(map_path=f'{files_path}ll_autonomicas_inspire_peninbal_etrs89/',coords_path=files_path,show_legend=False,save_fig=False)
        plt.show()
        sys.exit()

    """ Get signal sparsity via SVD decomposition"""
    determine_sparsity = False
    if determine_sparsity:
        # low-rank decomposition
        snapshots_matrix_train = X_train.to_numpy().T
        snapshots_matrix_val = X_val.to_numpy().T
        snapshots_matrix_test = X_test.to_numpy().T
        snapshots_matrix_train_centered = snapshots_matrix_train - snapshots_matrix_train.mean(axis=1)[:,None]
        snapshots_matrix_val_centered = snapshots_matrix_val - snapshots_matrix_train.mean(axis=1)[:,None]
        snapshots_matrix_test_centered = snapshots_matrix_test - snapshots_matrix_train.mean(axis=1)[:,None]
        U,sing_vals,Vt = np.linalg.svd(snapshots_matrix_train_centered,full_matrices=False)
        print(f'Training snapshots matrix has dimensions {snapshots_matrix_train_centered.shape}.\nLeft singular vectors matrix has dimensions {U.shape}\nRight singular vectors matrix has dimensions {Vt.shape}\nNumber of singular values: {sing_vals.shape}')

        print('\nDetermine signal sparsity from SVD decomposition.\nUse singular values ratios, cumulative energy, or reconstruction error for validation set.')
        s_range = np.arange(1,sing_vals.shape[0]+1,1)
        rmse_sparsity_train = signal_reconstruction_svd(U,snapshots_matrix_train,snapshots_matrix_train_centered,X_train,s_range)
        rmse_sparsity_val = signal_reconstruction_svd(U,snapshots_matrix_train,snapshots_matrix_val_centered,X_val,s_range)
        rmse_threshold = 5
        signal_sparsity = np.argwhere(rmse_sparsity_val.median(axis=0).to_numpy()<=rmse_threshold)[0][0] + 1
        print(f'Reconstruction error is lower than specified threshold {rmse_threshold} in validation set at sparsity of {signal_sparsity}.\nTraining set error of {rmse_sparsity_train.median(axis=0)[signal_sparsity]:.2f}\nValidation set error of {rmse_sparsity_val.median(axis=0)[signal_sparsity]:.2f}\nSingular value ratio: {sing_vals[signal_sparsity]/sing_vals[0]:.2f}\nCumulative energy: {(sing_vals.cumsum()/sing_vals.sum())[signal_sparsity]:.2f}')        
        cumulative_energy = np.cumsum(sing_vals)/np.sum(sing_vals)
        energy_threshold = 0.9
        signal_sparsity_energy = np.where(cumulative_energy>=energy_threshold)[0][0]
        print(f'Energy threshold of {energy_threshold} reached at singular at singular value index: {signal_sparsity_energy}')
        # dataset and sparsity figures
        plots = Figures(save_path=results_path,marker_size=1,
                        fs_label=12,fs_ticks=7,fs_legend=6,fs_title=10,
                        show_plots=True)
        plots.singular_values_cumulative_energy(sing_vals,n = X_train.shape[1],synthetic_dataset=synthetic_dataset,save_fig=True)
        #fig_rmse_sparsity_train = plots.boxplot_validation_rmse_svd(rmse_sparsity_train,max_sparsity_show=sing_vals.shape[0],save_fig=False)
        fig_rmse_sparsity_val = plots.boxplot_validation_rmse_svd(rmse_sparsity_val,n=X_train.shape[1],max_sparsity_show=sing_vals.shape[0],synthetic_dataset=synthetic_dataset,save_fig=True)
        plt.show()
        sys.exit()
    
    """
        Network planning algorithm
            - deploy sensors susch that the reconstructed signal variance is constrained to be lower than specified threshold
            - deploy single class of senors (called reference stations)
            - the number of deployed sensors is minimized and unknown a priori
            - The user-defined threshold design is the same over all locations
    """
    # network design algorithm with homogeneous design threshold
    deploy_sensors_homogeneous = False
    if deploy_sensors_homogeneous:
        # low-rank decomposition
        snapshots_matrix_train = X_train.to_numpy().T
        snapshots_matrix_val = X_val.to_numpy().T
        snapshots_matrix_test = X_test.to_numpy().T
        snapshots_matrix_train_centered = snapshots_matrix_train - snapshots_matrix_train.mean(axis=1)[:,None]
        snapshots_matrix_val_centered = snapshots_matrix_val - snapshots_matrix_train.mean(axis=1)[:,None]
        snapshots_matrix_test_centered = snapshots_matrix_test - snapshots_matrix_train.mean(axis=1)[:,None]
        U,sing_vals,Vt = np.linalg.svd(snapshots_matrix_train_centered,full_matrices=False)
        print(f'Training snapshots matrix has dimensions {snapshots_matrix_train_centered.shape}.\nLeft singular vectors matrix has dimensions {U.shape}\nRight singular vectors matrix has dimensions [{Vt.shape}]\nNumber of singular values: {sing_vals.shape}')
        # specify signal sparsity
        n = U.shape[0]
        if n==44:
            signal_sparsity = 28#[28,34] # 28(30) for RMSE < 5 // 34 for energy>0.9
        elif n==48:
            signal_sparsity = 36#[30,36] # 30 for RMSE < 0.5 // 36 for energy>0.9
        else:
            raise ValueError(f'No network with potential number of locations n:{n}')
        Psi = U[:,:signal_sparsity]
        
        print(f'Low-rank decomposition. Basis shape: {Psi.shape}')
        # initialize algorithm
        fully_monitored_network_max_variance = np.diag(Psi@np.linalg.inv(Psi.T@Psi)@Psi.T).max()
        variance_threshold_ratio = 1.5
        deployed_network_variance_threshold = variance_threshold_ratio*fully_monitored_network_max_variance
        algorithm = 'NetworkPlanning_iterative'
        sensor_placement = sp.SensorPlacement(algorithm, n, signal_sparsity,
                                              n_refst=n,n_lcs=0,n_unmonitored=0)
        # algorithm parameters
        """
        Mind the parameters. epsilon=1e-1 can make the problem infeasible at next iteration as some locations are assigned to be unmonitored.
        As epsilon increases less stations are assigned to be monitored, or it is easier to assign an unmonitored locations.
        """
        epsilon = 1e-1
        n_it = 20
        h_prev = np.zeros(n)
        weights = 1/(h_prev+epsilon)
        locations_monitored = []
        locations_unmonitored = []
        input(f'Iterative network planning algorithm.\n Parameters:\n -Max variance threshold ratio: {variance_threshold_ratio:.2f}\n -epsilon: {epsilon:.1e}\n -number of convergence iterations: {n_it}\nPress Enter to continue...')
        locations = networkPlanning_iterative(sensor_placement,n,Psi,deployed_network_variance_threshold,
                                              epsilon=epsilon,h_prev=h_prev,weights=weights,n_it=n_it,
                                              locations_monitored=locations_monitored,locations_unmonitored=locations_unmonitored)
        
        # deploy sensors and compute variance
        sensor_placement.locations = [[],np.sort(locations[0]),np.sort(locations[1])]
        sensor_placement.C_matrix()
        worst_coordinate_variance = np.diag(Psi@np.linalg.inv(Psi.T@sensor_placement.C[1].T@sensor_placement.C[1]@Psi)@Psi.T).max()
        n_locations_monitored = len(locations[0])
        n_locations_unmonitored = len(locations[1])
        print(f'Network planning results:\n- Total number of potential locations: {n}\n- basis sparsity: {signal_sparsity}\n- Fully monitored basis max variance: {fully_monitored_network_max_variance:.3f}\n- Max variance threshold: {deployed_network_variance_threshold:.3f}\n- Deployed network max variance: {worst_coordinate_variance:.3f}\n- Number of monitored locations: {n_locations_monitored}\n- Number of unmonitored locations: {n_locations_unmonitored}\n')
        if worst_coordinate_variance > deployed_network_variance_threshold:
            warnings.warn('Deployed network does not fullfill threshold design!')
        # save results
        fname = f'{results_path}SensorsLocations_N{n}_S{signal_sparsity}_VarThreshold{variance_threshold_ratio:.2f}_nSensors{n_locations_monitored}.pkl'
        with open(fname,'wb') as f:
            pickle.dump(locations[0],f,protocol=pickle.HIGHEST_PROTOCOL)
        print(f'File saved in {fname}')
        sys.exit()

    # network design algorithm with heterogeneous design threshold
    deploy_sensors_heterogeneous = False
    if deploy_sensors_heterogeneous:
        print(f'\n\nAlgorithm for heterogeneous constraints. Different design threshold will be applied to different Regions of Interest (ROIs)')
        # low-rank decomposition
        snapshots_matrix_train = X_train.to_numpy().T
        snapshots_matrix_val = X_val.to_numpy().T
        snapshots_matrix_test = X_test.to_numpy().T
        snapshots_matrix_train_centered = snapshots_matrix_train - snapshots_matrix_train.mean(axis=1)[:,None]
        snapshots_matrix_val_centered = snapshots_matrix_val - snapshots_matrix_train.mean(axis=1)[:,None]
        snapshots_matrix_test_centered = snapshots_matrix_test - snapshots_matrix_train.mean(axis=1)[:,None]
        U,sing_vals,Vt = np.linalg.svd(snapshots_matrix_train_centered,full_matrices=False)
        print(f'Training snapshots matrix has dimensions {snapshots_matrix_train_centered.shape}.\nLeft singular vectors matrix has dimensions {U.shape}\nRight singular vectors matrix has dimensions [{Vt.shape}]\nNumber of singular values: {sing_vals.shape}')
        # specify signal sparsity
        signal_sparsity = 36#[30,36] # 30 for RMSE < 0.5 // 36 for energy>0.9
        Psi = U[:,:signal_sparsity]
        n = Psi.shape[0]
        print(f'Low-rank decomposition. Basis shape: {Psi.shape}')
        # define ROIs with different threshold desings
        distance_based_ROIs = False
        if distance_based_ROIs:
            print('Distance based ROIs')
            distance_thresholds = [0,10] #km
            roi_threshold =  distance_thresholds
            n_regions = len(distance_thresholds)
            roi_idx = define_rois_distance(dataset.distances,distance_thresholds,n_regions)
        variance_based_ROIs = True
        if variance_based_ROIs:
            print('Variance based ROIs')
            coordinate_error_variance_fullymonitored = np.diag(Psi@np.linalg.inv(Psi.T@Psi)@Psi.T)
            variance_thresholds = [0,0.75]
            roi_threshold = variance_thresholds
            n_regions = len(variance_thresholds)
            roi_idx = define_rois_variance(coordinate_error_variance_fullymonitored,variance_thresholds,n_regions)
        
        # initialize algorithm
        Psi_new = np.concatenate([Psi[i,:] for i in roi_idx.values()])
        fully_monitored_network_max_variance = np.diag(Psi@np.linalg.inv(Psi.T@Psi)@Psi.T).max()
        fully_monitored_network_max_variance_ROI = [np.max(np.diag(Psi@np.linalg.inv(Psi.T@Psi)@Psi.T)[i]) for i in roi_idx.values()]
        variance_threshold_ratio = [1.5,2.0]
        if len(variance_threshold_ratio) != n_regions:
            raise ValueError(f'Number of user-defined thresholds ({len(variance_threshold_ratio)}) mismatch number of ROIs ({n_regions})')
        
        print(f'- Number of ROIs: {n_regions}.\n- Thresholds: {roi_threshold}\n- Number of elements in each ROI: {[len(i) for i in roi_idx.values()]}\n- max variance threshold ratio: {variance_threshold_ratio}')
        algorithm = 'NetworkPlanning_iterative'
        sensor_placement = sp.SensorPlacement(algorithm, n, signal_sparsity,
                                              n_refst=n,n_lcs=0,n_unmonitored=0)
        # algorithm parameters
        """
        Mind the parameters. epsilon=1e-1 can make the problem infeasible at next iteration as some locations are assigned to be unmonitored.
        As epsilon increases less stations are assigned to be monitored, or it is easier to assign an unmonitored locations.
        
        epsilon=1e-1 tends to fail at multiple thresholds as threshold <1.1
        """
        
        epsilon = 2e-1
        n_it = 20
        h_prev = np.zeros(n)
        weights = 1/(h_prev+epsilon)
        locations_monitored = []
        locations_unmonitored = []
        
        deployed_network_variance_threshold = np.zeros(shape=n)
        for f,v,idx in zip(variance_threshold_ratio,fully_monitored_network_max_variance_ROI,roi_idx.values()):
            np.put(deployed_network_variance_threshold,idx,f*v)
        #deployed_network_variance_threshold = np.concatenate([np.tile(f*v,len(roi_idx[d])) for f,v,d in zip(variance_threshold_ratio,fully_monitored_network_max_variance_ROI,roi_threshold)])
        input(f'Iterative network planning design.\n Parameters:\n -Max variance threshold ratio: {variance_threshold_ratio}\n -epsilon: {epsilon:.1e}\n -number of convergence iterations: {n_it}\nPress Enter to continue...')
        locations = networkPlanning_iterative(sensor_placement,n,Psi,deployed_network_variance_threshold,
                                              epsilon=epsilon,h_prev=h_prev,weights=weights,n_it=n_it,
                                              locations_monitored=locations_monitored,locations_unmonitored=locations_unmonitored)
        
        # deploy sensors and compute variance
        sensor_placement.locations = [[],np.sort(locations[0]),np.sort(locations[1])]
        sensor_placement.C_matrix()
        worst_coordinate_variance = np.diag(Psi@np.linalg.inv(Psi.T@sensor_placement.C[1].T@sensor_placement.C[1]@Psi)@Psi.T).max()
        worst_coordinate_variance_ROIs = np.diag(Psi@np.linalg.inv(Psi.T@sensor_placement.C[1].T@sensor_placement.C[1]@Psi)@Psi.T)
        n_locations_monitored = len(locations[0])
        n_locations_unmonitored = len(locations[1])
        print(f'Network design results:\n- Total number of potential locations: {n}\n- basis sparsity: {signal_sparsity}\n- Number of monitored locations: {n_locations_monitored}\n- Number of unmonitored locations: {n_locations_unmonitored}')
        print(f'- Overall fully monitored max variance: {fully_monitored_network_max_variance}\n- Fully monitored max variance per ROI: {fully_monitored_network_max_variance_ROI}\n- Max variance design threshold per ROI: {[np.unique(deployed_network_variance_threshold)]}\n- Overall deployed network max variance: {worst_coordinate_variance}\n- Deployed network max variance per ROI: {[np.max(worst_coordinate_variance_ROIs[i]) for i in roi_idx.values()]}')
        # save results
        fname = f'{results_path}SensorsLocations_N{n}_S{signal_sparsity}_VarThreshold{variance_threshold_ratio}_nSensors{n_locations_monitored}.pkl'
        with open(fname,'wb') as f:
            pickle.dump(locations[0],f,protocol=pickle.HIGHEST_PROTOCOL)
        print(f'File saved in {fname}')
        
        sys.exit()

    """ Compare NetworkDesign results for different parameters (epsilon)"""
    validate_epsilon = False
    if validate_epsilon:
        # low-rank decomposition
        snapshots_matrix_train = X_train.to_numpy().T
        snapshots_matrix_val = X_val.to_numpy().T
        snapshots_matrix_test = X_test.to_numpy().T
        snapshots_matrix_train_centered = snapshots_matrix_train - snapshots_matrix_train.mean(axis=1)[:,None]
        snapshots_matrix_val_centered = snapshots_matrix_val - snapshots_matrix_train.mean(axis=1)[:,None]
        snapshots_matrix_test_centered = snapshots_matrix_test - snapshots_matrix_train.mean(axis=1)[:,None]
        U,sing_vals,Vt = np.linalg.svd(snapshots_matrix_train_centered,full_matrices=False)
        print(f'Training snapshots matrix has dimensions {snapshots_matrix_train_centered.shape}.\nLeft singular vectors matrix has dimensions {U.shape}\nRight singular vectors matrix has dimensions [{Vt.shape}]\nNumber of singular values: {sing_vals.shape}')
        # specify signal sparsity and network parameters
        signal_sparsity = 28
        Psi = U[:,:signal_sparsity]
        n = Psi.shape[0]
        In = np.identity(n)
        # load moniteored locations IRL1ND results
        epsilon_range = np.logspace(-3,-1,3)
        variance_ratio_range = [1.01,1.05,1.1,1.2,1.3,1.4,1.5]
        worst_coordinate_variance_epsilon = pd.DataFrame([],columns=variance_ratio_range,index=epsilon_range)
        for var_ratio in variance_ratio_range:
            for epsilon in epsilon_range:
                fname = f'{results_path}NetworkDesign/epsilon{epsilon:.0e}/SensorsLocations_N{n}_S{signal_sparsity}_VarThreshold{var_ratio:.2f}.pkl'
                try:
                    with open(fname,'rb') as f:
                        locations_monitored = np.sort(pickle.load(f))
                    locations_unmonitored = [i for i in np.arange(n) if i not in locations_monitored]
                    C = In[locations_monitored,:]
                    worst_coordinate_variance_epsilon.loc[epsilon,var_ratio] = np.diag(Psi@np.linalg.inv(Psi.T@C.T@C@Psi)@Psi.T).max()
                except:
                    print(f'No file for error variance ratio {var_ratio:.2f} and epsilon {epsilon:.1e}')
        print(f'Analytical worst coordinate error variance for different IRL1ND parameter\n{worst_coordinate_variance_epsilon}')
        sys.exit()


    """ Reconstruct signal using measurements at certain locations and compare with actual values """
    reconstruct_signal = True
    if reconstruct_signal:
        print('\nReconstructing signal from sparse measurements.\nTwo algorithms are used:\n- Network design\n- Joshi-Boyd (D-optimal) sensor selection.')
        
        """ Set homogeneous/heterogeneous threshold applied for designing network """
        homogeneous_threshold = False
        if homogeneous_threshold:
            variance_threshold_ratio = 1.3
            n_sensors = 41
        else:
            variance_threshold_ratio = [2.0,2.5]
            n_sensors = 40
            n_sensors_Dopt = 40
            #n_sensorsROI = [5,35]

        # low-rank decomposition
        snapshots_matrix_train = X_train.to_numpy().T
        snapshots_matrix_val = X_val.to_numpy().T
        snapshots_matrix_test = X_test.to_numpy().T
        snapshots_matrix_train_centered = snapshots_matrix_train - snapshots_matrix_train.mean(axis=1)[:,None]
        snapshots_matrix_val_centered = snapshots_matrix_val - snapshots_matrix_train.mean(axis=1)[:,None]
        snapshots_matrix_test_centered = snapshots_matrix_test - snapshots_matrix_train.mean(axis=1)[:,None]
        U,sing_vals,Vt = np.linalg.svd(snapshots_matrix_train_centered,full_matrices=False)
        n = U.shape[0]
        print(f'Training snapshots matrix has dimensions {snapshots_matrix_train_centered.shape}.\nLeft singular vectors matrix has dimensions {U.shape}\nRight singular vectors matrix has dimensions [{Vt.shape}]\nNumber of singular values: {sing_vals.shape}')
        # specify signal sparsity and network parameters
        if n==44:
            signal_sparsity = 28#[28,34] # 28(30) for RMSE < 5 // 34 for energy>0.9
        elif n==48:
            signal_sparsity =36#[30,36] # 30 for RMSE < 0.5 // 36 for energy>0.9
        else:
            raise ValueError(f'No network with potential number of locations n:{n}')

        Psi = U[:,:signal_sparsity]
        
        # fully monitored network
        error_variance_fullymonitored = Psi@np.linalg.inv(Psi.T@Psi)@Psi.T
        if homogeneous_threshold:
            worst_variance_fullymonitored = np.diag(error_variance_fullymonitored).max()
            rmse_fullymonitored = np.sqrt(np.trace(error_variance_fullymonitored)/n)
        else:
            distance_based_ROIs = False
            if distance_based_ROIs:
                print('Distance based ROIs')
                distance_thresholds = [0,10] #km
                roi_threshold =  distance_thresholds
                n_regions = len(distance_thresholds)
                roi_idx = define_rois_distance(dataset.distances,distance_thresholds,n_regions)
            variance_based_ROIs = True
            if variance_based_ROIs:
                print('Variance based ROIs')
                coordinate_error_variance_fullymonitored = np.diag(Psi@np.linalg.inv(Psi.T@Psi)@Psi.T)
                variance_thresholds = [0,0.75]
                roi_threshold = variance_thresholds
                n_regions = len(variance_thresholds)
                roi_idx = define_rois_variance(coordinate_error_variance_fullymonitored,variance_thresholds,n_regions)

            error_variance_fullymonitored_roi = [(Psi@np.linalg.inv(Psi.T@Psi)@Psi.T)[i] for i in roi_idx.values()]
            worst_variance_fullymonitored_roi = [np.max(np.diag(Psi@np.linalg.inv(Psi.T@Psi)@Psi.T)[i]) for i in roi_idx.values()]
            

        print(f'Basis shape: {Psi.shape}.')
        # load monitored locations indices
        if homogeneous_threshold:
            fname = f'{results_path}NetworkDesign/homogeneous/SensorsLocations_N{n}_S{signal_sparsity}_VarThreshold{variance_threshold_ratio:.2f}_nSensors{n_sensors}.pkl'
        else:
            if distance_based_ROIs:
                fname = f'{results_path}NetworkDesign/heterogeneous/distance_based/SensorsLocations_N{n}_S{signal_sparsity}_VarThreshold{variance_threshold_ratio}_nSensors{n_sensors}.pkl'
            elif variance_based_ROIs:
                fname = f'{results_path}NetworkDesign/heterogeneous/variance_based/SensorsLocations_N{n}_S{signal_sparsity}_VarThreshold{variance_threshold_ratio}_nSensors{n_sensors}.pkl'
        with open(fname,'rb') as f:
            locations_monitored = np.sort(pickle.load(f))
        locations_unmonitored = [i for i in np.arange(n) if i not in locations_monitored]
        n_locations_monitored = len(locations_monitored)
        n_locations_unmonitored = len(locations_unmonitored)
        print(f'Loading indices of monitored locations from: {fname}\n- Total number of potential locations: {n}\n- Number of monitored locations: {len(locations_monitored)}\n- Number of unmonitored locations: {len(locations_unmonitored)}')
        for idx in roi_idx.values():
            print(f'ROI with {len(idx)} potential locations. Network design algorithm has {np.isin(idx,locations_monitored).sum()} sensors in ROI.')
        # get worst coordinate error variance analytically (from formula)
        In = np.identity(n)
        C = In[locations_monitored,:]
        error_variance_design = Psi@np.linalg.inv(Psi.T@C.T@C@Psi)@Psi.T
        worst_variance_design = np.diag(error_variance_design).max()
        rmse_design = np.sqrt(np.trace(Psi@np.linalg.inv(Psi.T@C.T@C@Psi)@Psi.T)/n)
        
        # get worst coordinate error variance from empirical signal reconstruction
        project_signal = True
        if project_signal:
            print('Reconstructed signal results.')
            X_test_proj = (Psi@Psi.T@X_test.T).T
            X_test_proj.columns = X_test.columns
            X_test_proj.index = X_test.index
            X_test_proj_noisy = add_noise_signal(X_test_proj,seed=42,var=1.0)
            rmse_design_reconstruction,errorvar_design_reconstruction = signal_reconstruction_regression(Psi,locations_monitored,X_test=X_test_proj,X_test_measurements=X_test_proj_noisy,projected_signal=True)
            rmse_fullymonitored_reconstruction,errorvar_fullymonitored_reconstruction = signal_reconstruction_regression(Psi,np.arange(n),X_test=X_test_proj,X_test_measurements=X_test_proj_noisy,projected_signal=True)
            
            
            # reconstruction using alternative method: Joshi-Boyd (D-optimality)
            try:
                print(f'Loading alternative sensor placement locations obtained with Dopt method.')
                if homogeneous_threshold:
                    fname = f'{results_path}Dopt/homogeneous/SensorsLocations_N{n}_S{signal_sparsity}_nSensors{n_locations_monitored}.pkl'
                else:
                    if distance_based_ROIs:
                        fname = f'{results_path}Dopt/heterogeneous/distance_based/SensorsLocations_N{n}_S{signal_sparsity}_nSensors{n_sensors_Dopt}_nSensorsROI{n_sensorsROI}.pkl'
                    else:
                        fname = f'{results_path}Dopt/heterogeneous/variance_based/SensorsLocations_Boyd_N{n}_S{signal_sparsity}_VarThreshold{variance_threshold_ratio}_nSensors{n_sensors_Dopt}.pkl'
                print(f'Loading D-opt solution from {fname}')
                with open(fname,'rb') as f:
                    locations_monitored_Dopt = np.sort(pickle.load(f))
                locations_unmonitored_Dopt = [i for i in np.arange(n) if i not in locations_monitored_Dopt]
                C_Dopt = In[locations_monitored_Dopt,:]
                error_variance_Dopt = Psi@np.linalg.inv(Psi.T@C_Dopt.T@C_Dopt@Psi)@Psi.T
                worst_variance_Dopt = np.diag(error_variance_Dopt).max()
                rmse_Dopt = np.sqrt(np.trace(error_variance_Dopt)/n)
                rmse_Dopt_reconstruction,errorvar_Dopt_reconstruction = signal_reconstruction_regression(Psi,locations_monitored_Dopt,X_test=X_test_proj,X_test_measurements=X_test_proj_noisy,projected_signal=True)

            except:
                    print(f'No Dopt sensor placement file for worse error variance threshold {variance_threshold_ratio} and num sensors {n_locations_monitored}')
                    errorvar_reconstruction_Dopt = []

        else:
            # fix before running!!
            rmse_reconstruction,errormax_reconstruction = signal_reconstruction_regression(Psi,locations_monitored,
                                                                                           snapshots_matrix_train,snapshots_matrix_test_centered,X_test)
            rmse_fullymonitored,errormax_fullymonitored = signal_reconstruction_regression(Psi,np.arange(n),
                                                                                           snapshots_matrix_train,snapshots_matrix_test_centered,X_test)
        if homogeneous_threshold:
            print('Results of variance-covariance matrices comparison')
            print(f'Network designed with threshold ratio: {variance_threshold_ratio:2f}')
            print(f'Fully monitored network\n- worst coordinate error variance: {worst_variance_fullymonitored:.3f}\n- RMSE: {rmse_fullymonitored:.3f}')
            print(f'Designed network\n- worst coordinate error variance: {worst_variance_design:.3f}\n- RMSE: {rmse_design:.3f}')
            print(f'D-optimality\n- worst coordinate error variance: {worst_variance_Dopt:.3f}\n- RMSE: {rmse_Dopt:.3f}')
            print(f'Network design and D-opt methods share {np.isin(locations_monitored,locations_monitored_Dopt).sum()} locations out of {len(locations_monitored)} monitored locations')

            print('\nResults from dataset comparison')
            print(f'Fully monitored network\n- worst coordinate error variance: {errorvar_fullymonitored_reconstruction.max():.3f}\n- RMSE: {rmse_fullymonitored_reconstruction.mean()}')
            print(f'Designed network\n- worst coordinate error variance: {errorvar_design_reconstruction.max():.3f}\n- RMSE: {rmse_design_reconstruction.mean()}')
            print(f'D-optimality\n- worst coordinate error variance: {errorvar_Dopt_reconstruction.max():.3f}\n- RMSE: {rmse_Dopt_reconstruction.mean()}')
        else:
            print('Results of variance-covariance matrix comparison\nROIs with different threshold design')
            print(f'Number of ROIs: {n_regions}.\nROIs thresholds: {roi_threshold}\nNumber of elements in each ROI: {[len(i) for i in roi_idx.values()]}\nNetwork designed with threshold ratios: {variance_threshold_ratio}')
            print(f'Fully monitored network:\n- worst coordinate error variance per ROI: {[np.round(i,3) for i in worst_variance_fullymonitored_roi]}')
            print(f'Maximum variance tolerated in each ROI: {[np.round(w*t,3) for w,t in zip(worst_variance_fullymonitored_roi,variance_threshold_ratio)]}')
            print(f'Designed network\n- worst coordinate error variance: {[np.round(np.max(np.diag(error_variance_design)[i]),3) for i in roi_idx.values()]}')
            print(f'D-optimality\n- worst coordinate error variance: {[np.round(np.max(np.diag(error_variance_Dopt)[i]),3) for i in roi_idx.values()]}')
            print(f'Network design and D-opt methods share {np.isin(locations_monitored,locations_monitored_Dopt).sum()} locations out of {len(locations_monitored)} monitored locations')

        # visualize        
        plots = Figures(save_path=results_path,marker_size=1,
                        fs_label=14,fs_ticks=8,fs_legend=6,fs_title=10,
                        show_plots=True)
            
        if homogeneous_threshold:
            # map of monitored and unmonitored locaitons
            plots.geographical_network_visualization(map_path=f'{files_path}ll_autonomicas_inspire_peninbal_etrs89/',df_coordinates=dataset.coordinates.reindex(dataset.distances.index),
                                                 locations_monitored=locations_monitored,show_legend=True,save_fig=False)
            plots.geographical_network_visualization(map_path=f'{files_path}ll_autonomicas_inspire_peninbal_etrs89/',df_coordinates=dataset.coordinates.reindex(dataset.distances.index),
                                                 locations_monitored=locations_monitored_Dopt,show_legend=True,save_fig=False)
            
            # curve of error variance for every location (from measurements)
            # plots.curve_errorvariance_comparison(errorvar_fullymonitored_reconstruction,errorvar_design_reconstruction,variance_threshold_ratio,errorvar_fullymonitored_reconstruction.max(),n,n_locations_monitored,errorvar_Dopt_reconstruction,save_fig=False)
            # error variance from analytical matrix
            plots.curve_errorvariance_comparison(np.diag(error_variance_fullymonitored),np.diag(error_variance_design),
                                                 variance_threshold_ratio,worst_variance_fullymonitored,
                                                 n,n_locations_monitored,
                                                 np.diag(error_variance_Dopt),save_fig=False)
        else:
            plots.geographical_network_visualization(map_path=f'{files_path}ll_autonomicas_inspire_peninbal_etrs89/',df_coordinates=dataset.coordinates.reindex(dataset.distances.index),
                                        locations_monitored=locations_monitored_Dopt,roi_idx=roi_idx,show_legend=True,save_fig=False)

            plots.curve_errorvariance_comparison(np.diag(error_variance_fullymonitored),np.diag(error_variance_design),
                                                 variance_threshold_ratio,worst_variance_fullymonitored_roi,
                                                 n,n_locations_monitored,
                                                 np.diag(error_variance_Dopt),roi_idx,n_sensors_Dopt,save_fig=True)
                    
        plt.show()
        sys.exit()