#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 17:21:04 2023

@author: jparedes
"""
import os
import time
import pandas as pd
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
    
    error_max = pd.DataFrame(np.abs(error).max(axis=1),columns=[n_sensors_reconstruction],index=X_test.index)
    error_var = np.zeros(shape = error.shape)
    for i in range(error.shape[0]):
        error_var[i,:] = np.diag(error.iloc[i,:].to_numpy()[:,None]@error.iloc[i,:].to_numpy()[:,None].T)
    error_var = pd.DataFrame(error_var,index=X_test.index,columns=X_test.columns)
    
    return rmse, error_var.mean()

def sensorplacement()->list:
    pass
# dataset
class Dataset():
    def __init__(self,pollutant:str='O3',start_date:str='2011-01-01',end_date:str='2022-12-31',files_path:str=''):
        self.pollutant = pollutant
        self.start_date = start_date
        self.end_date = end_date
        self.files_path = files_path
    
    def load_dataset(self):        
        fname = f'{self.files_path}{self.pollutant}_catalonia_clean_{self.start_date}_{self.end_date}.csv'
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
        date_range = pd.date_range(start=start_date,end=end_date,freq='H')
        date_idx = [i for i in date_range if i in X.index]
        data = X.loc[date_idx,[station_name]]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(data)
        ax.set_xlabel('date')
        ax.set_ylabel('Concentration ($\mu$g/$m^3$)')
        fig.tight_layout()

    def curve_timeseries_allstations(self,X:pd.DataFrame,date_init:str='2020-01-20',date_end:str='2021-10-27',save_fig=True):
        date_range = pd.date_range(start=start_date,end=end_date,freq='H')
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

    def geographical_network_visualization(self,map_path:str,coords_path:str,locations_monitored:np.array=[],show_legend:bool=False,save_fig:bool=False):
        
        df_coords = pd.read_csv(f'{coords_path}coordinates.csv',index_col=0)
        if len(locations_monitored)!=0:
            df_coords_monitored = df_coords.iloc[locations_monitored]
            df_coords_unmonitored = df_coords.iloc[[i for i in range(df_coords.shape[0]) if i not in locations_monitored]]
            geometry_monitored = [Point(xy) for xy in zip(df_coords_monitored['Longitude'], df_coords_monitored['Latitude'])]
            geometry_unmonitored = [Point(xy) for xy in zip(df_coords_unmonitored['Longitude'], df_coords_unmonitored['Latitude'])]
            gdf_monitored = GeoDataFrame(df_coords_monitored, geometry=geometry_monitored)
            gdf_unmonitored = GeoDataFrame(df_coords_unmonitored, geometry=geometry_unmonitored)

        else:
            df_coords_monitored = df_coords.copy()
            geometry_monitored = [Point(xy) for xy in zip(df_coords_monitored['Longitude'], df_coords_monitored['Latitude'])]
            gdf_monitored = GeoDataFrame(df_coords_monitored, geometry=geometry_monitored)
        
        spain = gpd.read_file(f'{map_path}ll_autonomicas_inspire_peninbal_etrs89.shp')
        catalonia = spain.loc[spain.NAME_BOUND.str.contains('Catalunya')]
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        geo_map = catalonia.plot(ax=ax,color='#117a65')
        gdf_monitored.plot(ax=geo_map, marker='o', color='#943126', markersize=6,label=f'Monitoring station')
        try:
            gdf_unmonitored.plot(ax=geo_map, marker='o', color='k', markersize=6,label=f'Unmonitored location')
        except:
            print('No unmonitored locations')
        ax.set_xlim(0.0,3.5)
        ax.set_ylim(40.5,43)
        
        ax.set_ylabel('Latitude (degrees)')
        ax.set_xlabel('Longitude (degrees)')

        if show_legend:
            ax.legend(loc='center',ncol=2,framealpha=1,bbox_to_anchor=(0.5,1.1))
        ax.tick_params(axis='both', which='major')
        fig.tight_layout()
        
        if save_fig:
            fname = self.save_path+f'Map_PotentialLocations_MonitoredLocations{df_coords_monitored.shape[0]}.png'
            fig.savefig(fname,dpi=300,format='png',bbox_inches='tight')
            print(f'Figure saved at {fname}')
        return fig
        

    # Low-rank plots
    def singular_values_cumulative_energy(self,sing_vals,n,save_fig=False):
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
        fig1.tight_layout()
        
        
        
        fig2 = plt.figure()
        ax = fig2.add_subplot(111)
        ax.plot(xrange, sing_vals / np.max(sing_vals),color='#1f618d',marker='o')
        ax.set_xticks(np.concatenate(([0.0],np.arange(xrange[9],xrange[-1]+1,10))))
        ax.set_xticklabels([int(i+1) for i in ax.get_xticks()],rotation=0)
        ax.set_xlabel('$i$th singular value')
        
        ax.set_yscale('log')
        yrange = np.logspace(-4,0,5)
        ax.set_yticks(yrange)
        ax.set_ylabel('Normalized singular values')
        ax.set_ylim(1e-2,1)
        fig2.tight_layout()
        
        if save_fig:
            fname = self.save_path+f'Curve_sparsity_cumulativeEnergy_N{n}.png'
            fig1.savefig(fname,dpi=300,format='png')
            print(f'Figure saved at: {fname}')

            fname = self.save_path+f'Curve_sparsity_singularValues_N{n}.png'
            fig2.savefig(fname,dpi=300,format='png')
            print(f'Figure saved at: {fname}')
         
    def boxplot_validation_rmse_svd(self,rmse_sparsity,max_sparsity_show=10,save_fig=False) -> plt.figure:
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
        ax.set_ylim(0,30)
        ax.set_ylabel('RMSE ($\mu$g/$m^3$)')
        xrange = np.array([i-1 for i in xrange if i%5==0])
        ax.set_xticks(xrange)
        ax.set_xticklabels([int(i+1) for i in xrange],rotation=0)
        ax.set_xlabel('Sparsity level')
        fig.tight_layout()

        if save_fig:
            fname = self.save_path+f'boxplot_RMSE_SVDreconstruction_validationSet_Smin{xrange.min()}_Smax{xrange.max()}.png'
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
    
    def curve_errorvariance_comparison(self,errorvar_fullymonitored:list,errorvar_reconstruction:list,variance_threshold:float,n:int,n_sensors:int,save_fig=False) -> plt.figure:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(errorvar_fullymonitored,color='#1a5276',label='Fully monitored network')
        ax.plot(errorvar_reconstruction,color='orange',label=f'Reconstruction with {n_sensors} sensors')
        ax.hlines(y=variance_threshold,xmin=0,xmax=n+1,color='k',linestyles='--',label=rf'Design threshold $\rho$={variance_threshold:.2f}')
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
        ax.legend(loc='upper left',ncol=1,framealpha=0.5)
        fig.tight_layout()
        if save_fig:
            fname = f'{self.save_path}Curve_errorVariance_Threshold{variance_threshold:.2f}_Nsensors{n_sensors}.png'
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

    pollutant = 'O3'
    start_date = '2011-01-01'
    end_date = '2022-12-31'

    dataset = Dataset(pollutant,start_date,end_date,files_path)
    dataset.load_dataset()
    dataset.check_dataset()

    """ snapshots matrix and low-rank decomposition """
    # train/val/test split
    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10
    X_train, X_test = train_test_split(dataset.ds, test_size= 1 - train_ratio,shuffle=False,random_state=92)
    X_val, X_test = train_test_split(X_test, test_size=test_ratio/(test_ratio + validation_ratio),shuffle=False,random_state=92) 
    print(f'Dataset matrix summary:\n {train_ratio} of dataset for training set with {X_train.shape[0]} measurements from {X_train.index[0]} until {X_train.index[-1]}\n {validation_ratio} of dataset for validation set with {X_val.shape[0]} measurements from {X_val.index[0]} until {X_val.index[-1]}\n {test_ratio} of measuerements for testing set with {X_test.shape[0]} measurements from {X_test.index[0]} until {X_test.index[-1]}')
    
    """
        D-optimality sensor placement criteria:
            - number of sensors specified a priori (from network design)
    """
    deploy_sensors = True
    if deploy_sensors:
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
        signal_sparsity = 28
        Psi = U[:,:signal_sparsity]
        n = Psi.shape[0]
        n_sensors = 31
        # initialize algorithm
        algorithm = 'JB'
        sensor_placement = sp.SensorPlacement(algorithm, n, signal_sparsity,
                                              n_refst=n_sensors,n_lcs=0,n_unmonitored=n-n_sensors)
        sensor_placement.initialize_problem(Psi)
        sensor_placement.solve()
        sensor_placement.discretize_solution()
        locations = [sensor_placement.locations[1],[i for i in np.arange(n) if i not in sensor_placement.locations[1]]]
        sensor_placement.C_matrix()
        # deploy sensors and compute variance
        worst_coordinate_variance = np.diag(Psi@np.linalg.inv(Psi.T@sensor_placement.C[1].T@sensor_placement.C[1]@Psi)@Psi.T).max()
        locations_monitored = sensor_placement.locations[1]
        n_locations_monitored = len(locations[0])
        n_locations_unmonitored = len(locations[1])
        print(f'Network planning results:\n- Total number of potential locations: {n}\n- basis sparsity: {signal_sparsity}\n- Deployed network max variance: {worst_coordinate_variance:.2f}\n- Number of monitored locations: {n_locations_monitored}\n- Number of unmonitored locations: {n_locations_unmonitored}\n')
        # save results
        fname = f'{results_path}SensorsLocations_N{n}_S{signal_sparsity}_nSensors{n_locations_monitored}.pkl'
        with open(fname,'wb') as f:
            pickle.dump(locations[0],f,protocol=pickle.HIGHEST_PROTOCOL)
        print(f'File saved in {fname}')
        sys.exit()

    """ Reconstruct signal using measurements at certain locations and compare with actual values """
    reconstruct_signal = True
    if reconstruct_signal:
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
        variance_threshold_ratio = 1.5
        n_locations_monitored = 30
        fully_monitored_network_max_variance = np.diag(Psi@np.linalg.inv(Psi.T@Psi)@Psi.T).max()
        deployed_network_variance_threshold = variance_threshold_ratio*fully_monitored_network_max_variance
        # load monitored locations indices
        fname = f'{results_path}Dopt/SensorsLocations_N{n}_S{signal_sparsity}_VarThreshold{variance_threshold_ratio:.1f}_nSensors{n_locations_monitored}.pkl'
        with open(fname,'rb') as f:
            locations_monitored = np.sort(pickle.load(f))
        locations_unmonitored = [i for i in np.arange(n) if i not in locations_monitored]
        print(f'Loading indices of monitored locations from: {fname}\n- Total number of potential locations: {n}\n- Number of monitored locations: {len(locations_monitored)}\n- Number of unmonitoreed locations: {len(locations_unmonitored)}')
        # get worst variance analytically
        In = np.identity(n)
        C = In[locations_monitored,:]
        worst_coordinate_variance = np.diag(Psi@np.linalg.inv(Psi.T@C.T@C@Psi)@Psi.T).max()
        print(f'Worst coordinate variance threshold: {variance_threshold_ratio}\nAnalytical worst coordinate variance achieved: {worst_coordinate_variance:.3f}')
        # empirical signal reconstruction
        project_signal = True
        if project_signal:
            X_test_proj = (Psi@Psi.T@X_test.T).T
            X_test_proj.columns = X_test.columns
            X_test_proj.index = X_test.index
            X_test_proj_noisy = add_noise_signal(X_test_proj,seed=0,var=1.0)
            rmse_reconstruction,errorvar_reconstruction = signal_reconstruction_regression(Psi,locations_monitored,X_test=X_test_proj,X_test_measurements=X_test_proj_noisy,projected_signal=True)
            rmse_fullymonitored,errorvar_fullymonitored = signal_reconstruction_regression(Psi,np.arange(n),X_test=X_test_proj,X_test_measurements=X_test_proj_noisy,projected_signal=True)
            
        else:
            # fix before running
            rmse_reconstruction,errormax_reconstruction = signal_reconstruction_regression(Psi,locations_monitored,
                                                                                           snapshots_matrix_train,snapshots_matrix_test_centered,X_test)
            rmse_fullymonitored,errormax_fullymonitored = signal_reconstruction_regression(Psi,np.arange(n),
                                                                                           snapshots_matrix_train,snapshots_matrix_test_centered,X_test)
        # visualize        
        plots = Figures(save_path=results_path,marker_size=1,
            fs_label=12,fs_ticks=7,fs_legend=6,fs_title=10,
            show_plots=True)
        plots.geographical_network_visualization(map_path=f'{files_path}ll_autonomicas_inspire_peninbal_etrs89/',coords_path=files_path,locations_monitored=locations_monitored,show_legend=True,save_fig=False)
        plots.curve_errorvariance_comparison(errorvar_fullymonitored,errorvar_reconstruction,deployed_network_variance_threshold,n,n_locations_monitored,save_fig=False)
        plt.show()
        sys.exit()