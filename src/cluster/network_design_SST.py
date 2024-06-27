#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 17:21:04 2023

@author: jparedes
"""
import os
import sys
import time
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import sys
import pickle
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import dask
import dask.array as da
import sensor_placement as sp

#%% Script parameters
parser = argparse.ArgumentParser(prog='IRNet-sensorPlacement',
                                 description='Iterative Reweighted Network design.',
                                 epilog='---')
# action to perform
parser.add_argument('--determine_sparsity',help='determine signal saprsity via SVD',action='store_true')
parser.add_argument('--design_network',help='Perform IRNet algorithm',action='store_true')
parser.add_argument('--design_large_networks',help='IRNet algorithm for monitoring network of large size',action='store_true')
parser.add_argument('-n','--n_locations',help='Number of potential locations. Network size.',type=int,default=691150,required=False)
parser.add_argument('-s','--signal_sparsity',help='Signal sparsity for basis cutoff',type=int,default=150,required=False)

# network design parameters
parser.add_argument('-e','--epsilon',help='Reweighting epsilon parameter',type=float,default=1e-1,required=False)
parser.add_argument('-n_it','--num_it',help='Number of iterations for updating monitored set',type=int,default=20,required=False)
parser.add_argument('-vtr','--variance_threshold_ratio',help='Maximum error variance threshold for network design',type=float,default=1.1,required=False)
args = parser.parse_args()



# recover map
def recover_map(y:np.array,idx_nan:np.array,n_rows:int,n_cols:int)->np.ndarray:
    """
    Recovers a snapshot image from vector of measurements and array with nan indices (indicating earth)
    
    Args:
        y (np.array): array of measurements
        idx_nan (np.array): array with indices where entry is nan
        n_rows (int): number of rows of snapshot figure
        n_cols (int): number of columns of snapshot figure

    Returns:
        X_mat (np.ndarray): recovered snapshot figure
    """
    
    # initalize array with nan values
    X_vect = np.full(len(y) + len(idx_nan),np.nan,dtype=np.float32)
    #print(f'Length of non-NaNs array: {len(y)}. Length of array of NaN indices: {len(idx_nan)}. Length of recovered array: {len(X_vect)}.\nFirst NaN idx: {idx_nan.values[0][0]}. Last NaN idx: {idx_nan.values[-1][0]}')
    # recover value at ith location
    idx = 0
    for i in range(len(X_vect)):
        #if i not in idx_nan.values:
        if i not in idx_nan.values:
            X_vect[i] = y[idx]
            idx += 1
    # reshape to matrix form
    X_mat = np.reshape(X_vect,newshape=(n_rows,n_cols),order='F')
    return X_mat

def grid_mask_map(image_height:int=720, image_width:int = 1440, block_size:int = 15):
    assert image_height % block_size == 0, "Image height must be a multiple of the block size."
    assert image_width % block_size == 0, "Image width must be a multiple of the block size."
    mask_grid = np.zeros((image_height, image_width), dtype=np.float32)
    # Iterate over each block
    for i in range(0, image_height, block_size):
        for j in range(0, image_width, block_size):
            # Calculate the center of each block
            center_i = i + block_size // 2
            center_j = j + block_size // 2
            # Mark the center pixel
            mask_grid[center_i, center_j] = 1 
    return mask_grid


def window_image(data:np.ndarray,lat_range:np.array,lon_range:np.array,lon_left:float,lon_right:float,lat_down:float,lat_up:float)->np.ndarray:
    idx_window_lon = np.where(np.logical_and(lon_range>=lon_left,lon_range<=lon_right))[0]
    idx_window_lat = np.where(np.logical_and(lat_range>=lat_down,lat_range<=lat_up))[0]
    data_new = data[idx_window_lat[0]:idx_window_lat[-1]+1,idx_window_lon[0]:idx_window_lon[-1]+1]
    mask_new = np.isnan(data_new)
    return data_new,mask_new, idx_window_lat,idx_window_lon

def define_ROI(dataset,lon_min:float=0.0,lon_max:float=180.,lat_min:float=-90.,lat_max:float=0.):
    """
    Defines a ROI from the dataset indices of pixels with measurements and specified coordinates for windowing.
    Latitude in interval [-90,90]
    Longitude in interval [0,360].
    The indices can be used on the dataset.df for obtaining the respective snapshots matrix at those locations (dataset.df[:,idx_ROI_measurements])

    Args:
        dataset (_type_): dataset class containing dataset indices and coordinates
        lon_min (float, optional): Minimum longitude of ROI. Defaults to 0.0.
        lon_max (float, optional): Maximum longitude of ROI. Defaults to 180..
        lat_min (float, optional): Minimum latitude of ROI. Defaults to -90..
        lat_max (float, optional): Maximum latitude of ROI. Defaults to 0.
    """
    # check
    if lat_min < -90:
        raise ValueError(f'Wrong value for minimum latitude of ROI (-90): {lat_min}')
    if lat_max > 90:
        raise ValueError(f'Wrong value for maximum latitude of ROI (90): {lat_max}')
    if lon_min < 0:
        raise ValueError(f'Wrong value for minimum longitude of ROI (0): {lon_min}')
    if lon_max > 360.:
        raise ValueError(f'Wrong value for maximum longitude of ROI (360): {lon_max}')
    if lat_min > lat_max :
        raise ValueError(f'Wrong order of maximum ({lat_max}) and minimum ({lat_min}) latitudes')
    if lon_min > lon_max :
        raise ValueError(f'Wrong order of maximum ({lon_max}) and minimum ({lon_min}) longitudes')

    idx_measurements = np.array([i[0] for i in dataset.idx_measurements.values],dtype=int)
    mesh = np.meshgrid(dataset.lon_range,dataset.lat_range)
    mesh_lon_array = np.reshape(mesh[0],newshape=mesh[0].shape[0]*mesh[0].shape[1],order='F')
    mesh_lat_array = np.reshape(mesh[1],newshape=mesh[1].shape[0]*mesh[1].shape[1],order='F')
    # window latitude and longitude to be within ROI borders
    lon_array_ROI = np.where(np.logical_and(mesh_lon_array>=lon_min,mesh_lon_array<=lon_max))[0]
    lat_array_ROI = np.where(np.logical_and(mesh_lat_array>=lat_min,mesh_lat_array<=lat_max))[0]
    # get indices that are within the window on both coordinates
    idx_ROI = lon_array_ROI[np.isin(lon_array_ROI,lat_array_ROI)]
    # retrieve the indices in basis that correspond to ROI indices in map
    #idx_ROI_measurements = np.sort(idx_measurements[np.isin(idx_measurements,idx_ROI)])
    idx_ROI_measurements = np.sort(np.where(np.isin(idx_measurements,idx_ROI))[0])
    return idx_ROI_measurements



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
def signal_reconstruction_svd(U:np.ndarray,mean_values:np.ndarray,snapshots_matrix_val:np.ndarray,s_range:np.ndarray,centered_data:bool=True) -> pd.DataFrame:
    """
    Decompose signal keeping s-first singular vectors using training set data
    and reconstruct validation set.

    Args:
        U (numpy array): left singular vectors matrix
        mean_values (numpy array): average value for each location obtained from training set snapshots matrix
        snapshots_matrix_val (numpy array): non-centererd snapshots matrix of validation set data
        s_range (numpy array): list of sparsity values to test
        centered_data (bool): True for PCA computed on centered data.

    Returns:
        rmse_sparsity: dataframe containing reconstruction errors at different times for each sparsity threshold in the range
    """
    print(f'Determining signal sparsity by decomposing training set and reconstructing validation set.\nRange of sparsity levels: {s_range}')
    rmse_sparsity = []
    for s in s_range:
        if centered_data:
            snapshots_matrix_val_pred_svd = da.matmul(U[:,:s],da.matmul(U[:,:s].T,snapshots_matrix_val - mean_values)) + mean_values
        else:
            snapshots_matrix_val_pred_svd = da.matmul(U[:,:s],da.matmul(U[:,:s].T,snapshots_matrix_val))

        error = snapshots_matrix_val - snapshots_matrix_val_pred_svd
        rmse = da.sqrt((error**2).mean(axis=0))
        rmse_sparsity.append(rmse)
    rmse_sparsity = np.array(dask.compute(*rmse_sparsity)).T
    rmse_sparsity = pd.DataFrame(rmse_sparsity,columns = s_range)

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

def networkPlanning_iterative(sensor_placement:sp.SensorPlacement,Psi:np.ndarray,deployed_network_variance_threshold:float,epsilon:float,h_prev:np.ndarray,weights:np.ndarray,n_it:int,locations_monitored:list=[],locations_unmonitored:list=[])->list:
    """
    IRNet network planning algorithm
    Args:
        sensor_placement (sp.SensorPlacement): sensor placement object containing network information
        Psi (np.ndarray): low-rank basis with (n_rows) locations and (n_cols) vectors/dimension
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
    epsilon_zero = epsilon/10
    primal_start = {'x':[],'sl':[],'ss':[]}
    it = 0
    time_init = time.time()
    
    while len(locations_monitored) + len(locations_unmonitored) != sensor_placement.n:
        # solve sensor placement with constraints
        sensor_placement.initialize_problem(Psi,rho=deployed_network_variance_threshold,w=weights,epsilon=epsilon,
                                        locations_monitored=locations_monitored,locations_unmonitored = locations_unmonitored,
                                        primal_start=primal_start)
        
        if sensor_placement.problem['status'] == 'optimal':
            # get solution
            primal_start['x'] = sensor_placement.problem['x']
            primal_start['sl'] = sensor_placement.problem['sl']
            primal_start['ss'] = sensor_placement.problem['ss']
            # update sets
            new_monitored = [int(i[0]) for i in np.argwhere(sensor_placement.h >= 1-epsilon) if i[0] not in locations_monitored]
            new_unmonitored = [int(i[0]) for i in np.argwhere(sensor_placement.h <= epsilon_zero) if i[0] not in locations_unmonitored]
            locations_monitored += new_monitored
            locations_unmonitored += new_unmonitored
            # check convergence
            if np.linalg.norm(sensor_placement.h - h_prev)<=epsilon or it==n_it:
                locations_monitored += [[int(i[0]) for i in np.argsort(sensor_placement.h,axis=0)[::-1] if i not in locations_monitored][0]]
                it = 0        
            h_prev = sensor_placement.h
            weights = 1/(h_prev + epsilon)
            it +=1
            print(f'Iteration results\n ->Primal objective: {sensor_placement.problem["primal objective"]:.6f}\n ->{len(locations_monitored) + len(locations_unmonitored)} locations assigned\n ->{len(locations_monitored)} monitored locations\n ->{len(locations_unmonitored)} unmonitored locations\n')
        
        else:
            # solver fails at iteration
            locations_monitored = locations_monitored[:-len(new_monitored)]
            locations_unmonitored = locations_unmonitored[:-len(new_unmonitored)]
            it+=1


    time_end = time.time()
    locations = [locations_monitored,locations_unmonitored]
    print(f'IRL1 algorithm finished in {time_end-time_init:.2f}s.')
    return locations


def networkPlanning_iterative_ROIs(dataset,Psi:np.ndarray,variance_threshold_ratio:float,epsilon:float,n_it:int,locaitons_forbidden:list=[])->list:
    """
    IRNet network planning algorithm for large monitoring network using Regions of Interest (ROIs) and forbidden locations
    Args:
        dataset : loaded dataset containing coordinates and snapshots matrix
        Psi (np.ndarray): low-rank basis with (n_rows) locations and (n_cols) vectors/dimension
        variance_threshold_ratio (float): percentage of worsening of coordinate error variance with respect to fully monitored network
        epsilon (float): IRL1 weights update constant
        n_it (int): IRL1 max iterations
        locations_forbidden (list,optional): set of indices of locations where a sensor cannot be deployed

    Returns:
        locations (list): indices of monitored and unmonitored locations [S,Sc]
    """
    algorithm = 'IRNet_ROI'#['IRNet_ROI','NetworkPlanning_iterative_LMI','IRL1ND']
    # Define ROIs size
    lat_min,lat_max = -90,90
    lon_min,lon_max = 0,360
    delta_lat = 5
    delta_lon = 5
    
    locations_monitored = []
    locations_unmonitored = []
    locations_monitored_roi = []
    locations_unmonitored_roi = []
    
    time_init = time.time()
    idx_roi = np.array([],dtype=int)
    for latitude in np.arange(lat_min,lat_max,delta_lat):
        for longitude in np.arange(lon_min,lon_max,delta_lon):
            # get entries of original (large) basis matrix that belong to ROI
            idx_roi_new = define_ROI(dataset,lat_min=latitude,lat_max=latitude+delta_lat,lon_min=longitude,lon_max=longitude+delta_lon)
            print(f'Longitude: {longitude} - Latitude: {latitude}')
            print(f'Number of new elements in ROI: {len(idx_roi_new)}')
            if len(idx_roi_new) == 0:
                continue
            # joint new indices to previous ROI indices
            idx_roi = np.sort(np.unique(np.concatenate((idx_roi,idx_roi_new),axis=0,dtype=np.int64)))
            Psi_roi = Psi[idx_roi,:]
            fully_monitored_network_max_variance_roi = da.diagonal(da.matmul(Psi_roi,Psi_roi.T)).max()
            fully_monitored_network_max_variance_roi = fully_monitored_network_max_variance_roi.compute()
            deployed_network_variance_threshold_roi = variance_threshold_ratio*fully_monitored_network_max_variance_roi
            n_roi = Psi_roi.shape[0]
            print(f'Size of ROI: {n_roi}')
            sys.stdout.flush()
            Psi_roi = Psi_roi.compute()

            # IRNet method parameters
            sensor_placement = sp.SensorPlacement(algorithm,n_roi,args.signal_sparsity,n_refst=n_roi,n_lcs=0,n_unmonitored=0)
            epsilon_zero = epsilon/10
            primal_start_roi = {'x':[],'sl':[],'ss':[]}
            it_roi = 0
            h_prev_roi = np.zeros(n_roi)
            weights_roi = 1/(h_prev+args.epsilon)
            # carry monitored locations from previous ROI step
            if len(locations_monitored)!=0:
                locations_monitored_roi = np.where(np.isin(idx_roi,locations_monitored))[0]
            else:
                locations_monitored_roi = []
            if len(locations_unmonitored)!=0:
                locations_unmonitored_roi = np.where(np.isin(idx_roi,locations_unmonitored_roi))[0]
            else:
                locations_unmonitored_roi = []
            
            new_monitored_roi = []
            new_unmonitored_roi = []

            # begin IRNet iterations
            while len(locations_monitored_roi) + len(locations_unmonitored_roi) != n_roi:
                # solve sensor placement with constraints
                sensor_placement.initialize_problem(Psi_roi,rho=deployed_network_variance_threshold_roi,w=weights_roi,epsilon=epsilon,
                                                locations_monitored=locations_monitored_roi,locations_unmonitored = locations_unmonitored_roi,
                                                primal_start=primal_start_roi)
                
                if sensor_placement.problem['status'] == 'optimal':
                    # get solution dictionary
                    primal_start_roi['x'] = sensor_placement.problem['x']
                    primal_start_roi['sl'] = sensor_placement.problem['sl']
                    primal_start_roi['ss'] = sensor_placement.problem['ss']
                    # update sets: get entries (from constrained basis) of monitored and unmonitored locations
                    new_monitored_roi = [int(i[0]) for i in np.argwhere(sensor_placement.h >= 1-epsilon) if i[0] not in locations_monitored_roi]
                    new_unmonitored_roi = [int(i[0]) for i in np.argwhere(sensor_placement.h <= epsilon_zero) if i[0] not in locations_unmonitored_roi]
                    locations_monitored_roi += new_monitored_roi
                    locations_unmonitored_roi += new_unmonitored_roi
                    # check convergence: update entries of monitored locations
                    if np.linalg.norm(sensor_placement.h - h_prev_roi)<=epsilon or it_roi==n_it:
                        locations_monitored_roi += [[int(i[0]) for i in np.argsort(sensor_placement.h,axis=0)[::-1] if i not in locations_monitored_roi][0]]
                        it_roi = 0        
                    h_prev_roi = sensor_placement.h
                    weights_roi = 1/(h_prev_roi + epsilon)
                    it_roi +=1
                    print(f'Iteration results\n ->Primal objective: {sensor_placement.problem["primal objective"]:.6f}\n ->{len(locations_monitored_roi) + len(locations_unmonitored_roi)} locations assigned\n ->{len(locations_monitored_roi)} monitored locations\n ->{len(locations_unmonitored_roi)} unmonitored locations\n')
                    sys.stdout.flush()
                
                else:
                    # solver fails at iteration
                    #locations_monitored = locations_monitored[:-len(new_monitored)]
                    locations_unmonitored_roi = locations_unmonitored_roi[:-len(new_unmonitored_roi)]
                    it_roi+=1
            
            # add monitored locations from ROI to list of overall monitored locations of original basis entries
            locations_monitored += list(idx_roi[locations_monitored_roi])
            locations_unmonitored += list(idx_roi[locations_unmonitored_roi])
    time_end = time.time()

    locations = [locations_monitored,locations_unmonitored]
    print(f'IRNet algorithm finished in {time_end-time_init:.2f}s.')
    return locations

# dataset
class Dataset():
    def __init__(self,files_path:str='',fname:str='SST_month.parquet',n_locations:int=6624):
        self.files_path = files_path
        self.fname = fname
        self.n_locations = n_locations
    
    def load_dataset(self):
        print(f'Loading dataset from {self.files_path}{self.fname}')
        
        #self.df = pd.read_parquet(f'{self.files_path}{self.fname}')
        self.df = np.load(f'{self.files_path}{self.fname}').T
        self.idx_land = pd.read_csv(f'{self.files_path}idx_land_N{self.n_locations}.csv',header=None,index_col=None)
        self.idx_measurements = pd.read_csv(f'{self.files_path}idx_measurements_N{self.n_locations}.csv',header=None,index_col=None)
        self.lat_range = np.load(f'{self.files_path}latitudes_range.npy')
        self.lon_range = np.load(f'{self.files_path}longitudes_range.npy')
        print(f'Dataset Loaded.\n -num measurements: {self.df.shape[0]}\n -number of locations: {self.df.shape[1]}')

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


    def SST_map(self,data,vmin=0,vmax=30,show_coords=False,show_cbar=True,save_fig=False,save_fig_fname='SST_map.png'):
        gmap = Basemap(lon_0=180, projection="kav7", resolution='c')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cmap = mpl.colormaps['Spectral'].resampled(64).reversed()
        cmap.set_bad('w')
        cmap_trunc = mpl.colors.LinearSegmentedColormap.from_list('trunc_cmap',cmap(np.linspace(0.,30.,100)),N=64)
        im = gmap.imshow(data,cmap=cmap,origin='lower',vmin=vmin,vmax=vmax)
        
        xrange = np.arange(0,1600,200)
        if show_coords:
            gmap.drawmeridians(np.arange(0, 360.01, 90), ax=ax,linewidth=0.5,labels=[1,0,1,0])
            gmap.drawparallels(np.arange(-90,90.1,45), ax=ax,linewidth=0.5,labels=[1,0,0,1])
        else:
            gmap.drawmeridians(np.arange(0, 360.01, 90), ax=ax,linewidth=0.5)
            gmap.drawparallels(np.arange(-90,90.1,45), ax=ax,linewidth=0.5)        
        
        # color bar in same figure
        """ 
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(im,cax=cax,orientation='vertical')
        cbar.set_ticks(np.arange(0,34,5))
        cbar.set_ticklabels([np.round(i,1) for i in cbar.get_ticks()])
        cbar.set_label('Temperature (ºC)')
        """
        if show_cbar:
            fig2, ax2 = plt.subplots(figsize=(6, 1), layout='constrained')
            cbar = fig2.colorbar(mpl.cm.ScalarMappable(cmap=cmap,norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)),cax=ax2,
                                orientation='horizontal', label=f'Temperature (ºC)',
                                location='top',extend='both')

        fig.tight_layout()
        if save_fig:
            fname = f'{self.save_path}{save_fig_fname}'
            fig.savefig(fname,dpi=300,format='png',bbox_inches='tight')
            print(f'Figure saved at {fname}')
            if show_cbar:
                fname= f'{self.save_path}SST_map_colorbar.png'
                fig2.savefig(fname,dpi=300,format='png')

        return fig
    

    def curve_IQR_measurements(self,snapshots_matrix,save_fig):
        n = snapshots_matrix.shape[0]
        yrange = np.arange(-5,40,5)
        xrange = np.arange(0,n,1)
        median = np.median(snapshots_matrix,axis=1)
        q1,q3 = np.percentile(snapshots_matrix,axis=1,q=[25,75])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xrange,median,color='#1a5276')
        ax.fill_between(x=xrange,y1=q1,y2=q3,color='#1a5276',alpha=0.5)
        ax.set_yticks(yrange)
        ax.set_yticklabels([np.round(i,1) for i in ax.get_yticks()])
        ax.set_ylabel('Temp (ºC)')
        
        xrange = np.where(xrange%10000==0)[0]
        ax.set_xticks(xrange)
        xrange[0]=1
        ax.set_xticklabels([int(i+1) for i in xrange],rotation=0)
        ax.set_xlabel('Location index')
        fig.tight_layout()
        if save_fig:
            fname = self.save_path+'curve_Temp_allLocations.png'
            fig.savefig(fname,dpi=300,format='png',bbox_inches='tight')
            print(f'Figure saved at {fname}')

    # Low-rank plots
    def singular_values_cumulative_energy(self,sing_vals,save_fig=False):
        """
        Plot sorted singular values ratio and cumulative energy

        Parameters
        ----------
        sing_vals : numpy array
            singular values
        save_fig : bool, optional
            save generated figures. The default is False.

        Returns
        -------
        None.

        """
        cumulative_energy = np.cumsum(sing_vals)/np.sum(sing_vals)
        sing_vals_normalized = sing_vals / np.max(sing_vals)
        xrange = np.arange(0,sing_vals.shape[0],1)
        fig1 = plt.figure()
        ax = fig1.add_subplot(111)
        ax.plot(xrange,cumulative_energy,color='#1f618d',marker='o')
        ax.set_xticks(np.concatenate(([0.0],np.arange(xrange[49],xrange[-1]+1,50))))
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
        ax.plot(xrange, sing_vals_normalized,color='#1f618d',marker='o')
        ax.set_xticks(np.concatenate(([0.0],np.arange(xrange[49],xrange[-1]+1,50))))
        ax.set_xticklabels([int(i+1) for i in ax.get_xticks()],rotation=0)
        ax.set_xlabel('$i$th singular value')

        yrange = np.logspace(-4,0,5)
        ax.set_yticks(yrange)
        ax.set_ylabel('Normalized singular values')
        ax.set_yscale('log')
        ax.set_ylim(sing_vals_normalized[-2],1e0)
        fig2.tight_layout()
        
        if save_fig:
            fname = self.save_path+f'Curve_sparsity_cumulativeEnergy.png'
            fig1.savefig(fname,dpi=300,format='png')
            print(f'Figure saved at: {fname}')

            fname = self.save_path+f'Curve_sparsity_singularValues.png'
            fig2.savefig(fname,dpi=300,format='png')
            print(f'Figure saved at: {fname}')
         
    def boxplot_validation_rmse_svd(self,rmse_sparsity_val:pd.DataFrame,rmse_sparsity_train:pd.DataFrame=pd.DataFrame(),max_sparsity_show:int=10,save_fig:bool=False) -> plt.figure:
        yrange = np.arange(0.0,1.4,0.2)
        xrange = rmse_sparsity_val.columns[:max_sparsity_show].astype(int)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        bp_val = ax.boxplot(x=rmse_sparsity_val.iloc[:,:max_sparsity_show],notch=False,vert=True,
                   whis=1.5,bootstrap = None,
                   positions=[i for i in range(len(xrange))],widths=0.5,labels=[str(i) for i in xrange],
                   flierprops={'marker':'.','markersize':1},
                   patch_artist=True)
        for i in range(len(xrange)):
            bp_val['boxes'][i].set_facecolor('#1a5276')


        if rmse_sparsity_train.shape[0] !=0:
            bp_train = ax.boxplot(x=rmse_sparsity_train.iloc[:,:max_sparsity_show],notch=False,vert=True,
                    whis=1.5,bootstrap = None,
                    positions=[i for i in range(len(xrange))],widths=0.5,labels=[str(i) for i in xrange],
                    flierprops={'marker':'.','markersize':1},
                    patch_artist=True)
            for i in range(len(xrange)):
                bp_train['boxes'][i].set_facecolor('lightgreen')

        
        ax.set_yticks(yrange)
        ax.set_yticklabels([np.round(i,2) for i in ax.get_yticks()])
        ax.set_ylim(0,1.2)
        ax.set_ylabel('RMSE (ºC)')
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
    
    def curve_errorvariance_comparison(self,errorvar_fullymonitored:list,errorvar_reconstruction:list,variance_threshold_ratio:float,worst_coordinate_variance_fullymonitored:float,n:int,n_sensors:int,errorvar_reconstruction_Dopt:list=[],save_fig:bool=False) -> plt.figure:
        """
        Show error variance over a testing set vs network locations (n). 
        The error variance is obtained after reconstructing the signal from p measurements.
        The p measurement locations are obtained from IRL1ND algorithm.
        It also shows the threshold line which the IRL1ND algorithm used.
        Another algorithm can be shown for comparison.

        Args:
            errorvar_fullymonitored (list): error variance at each network location obtained with a fully monitored network
            errorvar_reconstruction (list): error variance at each network locations obtained with a network with a reduced number of deployed sensors
            variance_threshold_ratio (float): variance threshold ratio used for design algorithm
            worst_coordinate_variance_fullymonitored (float): fully-monitored network worst coordinate error variance
            n (int): total number of network points
            n_sensors (int): number of deployed sensors
            save_fig (bool, optional): Save generated figure. Defaults to False.

        Returns:
            plt.figure: Figure with error variance curves
        """
        variance_threshold = variance_threshold_ratio*worst_coordinate_variance_fullymonitored
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(errorvar_fullymonitored,color='#1d8348',label='Fully monitored network')
        if len(errorvar_reconstruction_Dopt) !=0:
            ax.plot(errorvar_reconstruction_Dopt,color='orange',label=f'logdet solution',alpha=0.8)
        ax.plot(errorvar_reconstruction,color='#1a5276',label=f'IRL1ND solution')
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
    files_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/files/'
    results_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/test/'    
    dataset = Dataset(files_path,fname=f'SST_month_N{args.n_locations}.npy',n_locations=args.n_locations)
    dataset.load_dataset()
    map_recovered = recover_map(dataset.df[0,:],dataset.idx_land,n_rows=720,n_cols=1440)
    plots = Figures(save_path=results_path,marker_size=1,
                fs_label=12,fs_ticks=7,fs_legend=6,fs_title=10,
                show_plots=False)

    plots.SST_map(map_recovered,vmin=0,vmax=30,show_coords=True,show_cbar=False,save_fig=False)
    
    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10

    """ Get signal sparsity via SVD decomposition"""
    
    if args.determine_sparsity:
        # low-rank decomposition of snapshots matrix
        print('Preparing snapshots matrix')
        X_train, X_test = train_test_split(dataset.df, test_size= 1 - train_ratio,shuffle=False,random_state=92)
        X_val, X_test = train_test_split(X_test, test_size=test_ratio/(test_ratio + validation_ratio),shuffle=False,random_state=92) 
        #snapshots_matrix_train = da.array(X_train.to_numpy().T)
        snapshots_matrix_train = da.array(X_train.T)
        mean_values = snapshots_matrix_train.mean(axis=1)[:,None]
        U,sing_vals,Vt = da.linalg.svd(snapshots_matrix_train - mean_values)
        print(f'Training snapshots matrix has dimensions {snapshots_matrix_train.shape}.\nLeft singular vectors matrix has dimensions {U.shape}\nRight singular vectors matrix has dimensions [{Vt.shape}]\nNumber of singular values: {sing_vals.shape}')
        
        # singular values figure
        plots = Figures(save_path=results_path,marker_size=1,
                fs_label=12,fs_ticks=7,fs_legend=6,fs_title=10,
                show_plots=False)
        sing_vals = sing_vals.compute()
        plots.singular_values_cumulative_energy(sing_vals,save_fig=True)
        
        # eigenmodes figure: takes time!
        show_eigenmode_map = False
        if show_eigenmode_map:
            print('Recovering map offirst eigenmode')
            if args.n_locations == 1039:
                eigenmode_matrix = recover_map(U[:,0],dataset.idx_land,n_rows=40,n_cols=40)
            elif args.n_locations == 6624:
                eigenmode_matrix = recover_map(U[:,0],dataset.idx_land,n_rows=100,n_cols=100)
            elif args.n_locations == 691150:
                eigenmode_matrix = recover_map(U[:,0],dataset.idx_land,n_rows=720,n_cols=1440)
            plots.SST_map(eigenmode_matrix,vmin=np.nanmin(eigenmode_matrix),vmax=np.nanmax(eigenmode_matrix),show_coords=False,show_cbar=False,save_fig=True,save_fig_fname='SST_map_SVD_mode1.png')
        
        """ signal reconstruction at different sparsity levels """
        print('\nDetermine signal sparsity from SVD decomposition.\nUse singular values ratios, cumulative energy, or reconstruction error for validation set.')
        s_range = np.array([1,50,100,150,200,250,300,384])
        del X_train, Vt
        snapshots_matrix_val = da.array(X_val.T)
        rmse_sparsity_val = signal_reconstruction_svd(U,mean_values,snapshots_matrix_val,s_range,centered_data=True)
        print(f'Sparsity reconstruction over validation set\n{rmse_sparsity_val.median(axis=0)}')

        rmse_threshold = 0.5
        signal_sparsity = rmse_sparsity_val.columns[np.argwhere(rmse_sparsity_val.median(axis=0).to_numpy()<=rmse_threshold)[0][0]]
        print(f'Reconstruction error is lower than specified threshold {rmse_threshold} in validation set at sparsity of {signal_sparsity}.\nSingular value ratio: {sing_vals[int(signal_sparsity)]/sing_vals[0]:.2e}\nCumulative energy: {(sing_vals.cumsum()/sing_vals.sum())[int(signal_sparsity)]:.2f}')        
    
        # figure of reconstruction errors on validation set 
        plots.boxplot_validation_rmse_svd(rmse_sparsity_val,max_sparsity_show=sing_vals.shape[0],save_fig=True)
        sys.exit()
    
    """
        Network planning algorithm
            - deploy sensors susch that the reconstructed signal variance is minimized
            - deploy single class of senors (called reference stations)
            - the number of deployed sensors is unknown a priori
    """
    if args.design_network:
        # low-rank decomposition
        X_train, X_test = train_test_split(dataset.df, test_size= 1 - train_ratio,shuffle=False,random_state=92)
        snapshots_matrix_train = da.array(X_train.to_numpy().T)
        mean_values = snapshots_matrix_train.mean(axis=1)[:,None]
        U,sing_vals,Vt = da.linalg.svd(snapshots_matrix_train - mean_values)
        # specify signal sparsity
        Psi = U[:,:args.signal_sparsity]        
        n = Psi.shape[0]
        # initialize algorithm
        fully_monitored_network_max_variance = da.diagonal(da.matmul(Psi,Psi.T)).max()
        
        fully_monitored_network_max_variance = fully_monitored_network_max_variance.compute()
        Psi = Psi.compute()
        del U, snapshots_matrix_train, X_train

        deployed_network_variance_threshold = args.variance_threshold_ratio*fully_monitored_network_max_variance
        algorithm = 'IRL1ND'#['NetworkPlanning_iterative_LMI','IRL1ND']
        sensor_placement = sp.SensorPlacement(algorithm, n, args.signal_sparsity,
                                              n_refst=n,n_lcs=0,n_unmonitored=0)
        # algorithm parameters
        h_prev = np.zeros(n)
        w = 1/(h_prev+args.epsilon)
        locations_monitored = []
        locations_unmonitored = []
        print(f'Iterative network planning algorithm.\n Parameters:\n -Basis shape: {Psi.shape}\n -Max variance threshold ratio: {args.variance_threshold_ratio:.2f}\n -epsilon: {args.epsilon:.1e}\n -number of convergence iterations: {args.num_it}')
        locations = networkPlanning_iterative(sensor_placement,Psi,deployed_network_variance_threshold,
                                              epsilon=args.epsilon,h_prev=h_prev,weights=w,n_it=args.num_it,
                                              locations_monitored=locations_monitored,locations_unmonitored=locations_unmonitored)
        
        # deploy sensors and compute variance
        sensor_placement.locations = [[],np.sort(locations[0]),np.sort(locations[1])]
        sensor_placement.C_matrix()
        worst_coordinate_variance = np.diag(Psi@np.linalg.inv(Psi.T@sensor_placement.C[1].T@sensor_placement.C[1]@Psi)@Psi.T).max()
        n_locations_monitored = len(locations[0])
        n_locations_unmonitored = len(locations[1])
        print(f'Network planning results:\n- Total number of potential locations: {n}\n- basis sparsity: {signal_sparsity}\n- Fully monitored basis max variance: {fully_monitored_network_max_variance:.2f}\n- Max variance threshold: {deployed_network_variance_threshold:.2f}\n- Deployed network max variance: {worst_coordinate_variance:.2f}\n- Number of monitored locations: {n_locations_monitored}\n- Number of unmonitored locations: {n_locations_unmonitored}\n')
        # save results
        fname = f'{results_path}SensorsLocations_N{n}_S{args.signal_sparsity}_VarThreshold{args.variance_threshold_ratio:.2f}_nSensors{n_locations_monitored}.pkl'
        with open(fname,'wb') as f:
            pickle.dump(locations[0],f,protocol=pickle.HIGHEST_PROTOCOL)
        print(f'File saved in {fname}')
        sys.exit()

    """ IRNet method adapted for large monitoring networks (SST dataset)"""
    if args.design_large_networks:
        # low-rank decomposition
        X_train, X_test = train_test_split(dataset.df, test_size= 1 - train_ratio,shuffle=False,random_state=92)
        snapshots_matrix_train = da.array(X_train.T)
        mean_values = snapshots_matrix_train.mean(axis=1)[:,None]
        print(f'Training set snapshots matrix shape: {snapshots_matrix_train.shape}')
        sys.stdout.flush()
        U,sing_vals,Vt = da.linalg.svd(snapshots_matrix_train - mean_values)
        # specify signal sparsity
        Psi = U[:,:args.signal_sparsity]        
        n = Psi.shape[0]

        # initialize algorithm
        fully_monitored_network_max_variance = da.diagonal(da.matmul(Psi,Psi.T)).max()
        fully_monitored_network_max_variance = fully_monitored_network_max_variance.compute()
        del U, snapshots_matrix_train, X_train
        deployed_network_variance_threshold = args.variance_threshold_ratio*fully_monitored_network_max_variance
        
        # setup algorithm
        
        """ 
        mask_grid = grid_mask_map()
        mask_grid = np.reshape(mask_grid,newshape=mask_grid.shape[0]*mask_grid.shape[1],order='F')
        mask_grid = np.delete(mask_grid, np.array([int(i[0]) for i in dataset.idx_land.values]))
        locations_forbidden = np.where(mask_grid==0)[0]
        """

        print(f'Iterative network planning algorithm.\n Parameters:\n -Basis shape: {Psi.shape}\n -Max variance threshold ratio: {args.variance_threshold_ratio:.2f}\n -epsilon: {args.epsilon:.1e}\n -number of convergence iterations: {args.num_it}')
        sys.stdout.flush()
        # IRNet for large-scale deployments
        locations = networkPlanning_iterative_ROIs(dataset,Psi,args.variance_threshold_ratio,args.epsilon,args.num_it)
        
        # deploy sensors and compute variance
        sensor_placement.locations = [[],np.sort(locations[0]),np.sort(locations[1])]
        sensor_placement.C_matrix()
        worst_coordinate_variance = np.diag(Psi@np.linalg.inv(Psi.T@sensor_placement.C[1].T@sensor_placement.C[1]@Psi)@Psi.T).max()
        n_locations_monitored = len(locations[0])
        n_locations_unmonitored = len(locations[1])
        print(f'Network planning results:\n- Total number of potential locations: {n}\n- basis sparsity: {signal_sparsity}\n- Fully monitored basis max variance: {fully_monitored_network_max_variance:.2f}\n- Max variance threshold: {deployed_network_variance_threshold:.2f}\n- Deployed network max variance: {worst_coordinate_variance:.2f}\n- Number of monitored locations: {n_locations_monitored}\n- Number of unmonitored locations: {n_locations_unmonitored}\n')
        # save results
        fname = f'{results_path}SensorsLocations_N{n}_S{args.signal_sparsity}_VarThreshold{args.variance_threshold_ratio:.2f}_nSensors{n_locations_monitored}.pkl'
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
    reconstruct_signal = False
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
        epsilon = 1e-2
        variance_threshold_ratio = 1.5
        fully_monitored_network_max_variance = np.diag(Psi@np.linalg.inv(Psi.T@Psi)@Psi.T).max()
        deployed_network_variance_threshold = variance_threshold_ratio*fully_monitored_network_max_variance
        # load monitored locations indices
        fname = f'{results_path}NetworkDesign/epsilon{epsilon:.0e}/SensorsLocations_N{n}_S{signal_sparsity}_VarThreshold{variance_threshold_ratio:.2f}.pkl'
        with open(fname,'rb') as f:
            locations_monitored = np.sort(pickle.load(f))
        locations_unmonitored = [i for i in np.arange(n) if i not in locations_monitored]
        n_locations_monitored = len(locations_monitored)
        n_locations_unmonitored = len(locations_unmonitored)
        print(f'Loading indices of monitored locations from: {fname}\n- Total number of potential locations: {n}\n- Number of monitored locations: {len(locations_monitored)}\n- Number of unmonitoreed locations: {len(locations_unmonitored)}')
        # get worst variance analytically
        In = np.identity(n)
        C = In[locations_monitored,:]
        worst_coordinate_variance_reconstruction = np.diag(Psi@np.linalg.inv(Psi.T@C.T@C@Psi)@Psi.T).max()
        error_variance_reconstruction = np.diag(Psi@np.linalg.inv(Psi.T@C.T@C@Psi)@Psi.T)
        error_variance_fullymonitored = np.diag(Psi@np.linalg.inv(Psi.T@Psi)@Psi.T)
        print(f'Worst coordinate variance threshold: {deployed_network_variance_threshold:.3f}\nAnalytical Fullymonitored worst coordinate variance: {error_variance_fullymonitored.max():.3f}\nAnalytical worst coordinate variance achieved: {worst_coordinate_variance_reconstruction:.3f}')
        # empirical signal reconstruction
        project_signal = True
        if project_signal:
            X_test_proj = (Psi@Psi.T@X_test.T).T
            X_test_proj.columns = X_test.columns
            X_test_proj.index = X_test.index
            X_test_proj_noisy = add_noise_signal(X_test_proj,seed=42,var=1.0)
            rmse_reconstruction,errorvar_reconstruction = signal_reconstruction_regression(Psi,locations_monitored,X_test=X_test_proj,X_test_measurements=X_test_proj_noisy,projected_signal=True)
            rmse_fullymonitored,errorvar_fullymonitored = signal_reconstruction_regression(Psi,np.arange(n),X_test=X_test_proj,X_test_measurements=X_test_proj_noisy,projected_signal=True)
            
            # reconstruction using alternative method
            try:
                fname = f'{results_path}Dopt/SensorsLocations_N{n}_S{signal_sparsity}_nSensors{n_locations_monitored}.pkl'
                with open(fname,'rb') as f:
                    locations_monitored_Dopt = np.sort(pickle.load(f))
                locations_unmonitored_Dopt = [i for i in np.arange(n) if i not in locations_monitored_Dopt]
                C_Dopt = In[locations_monitored_Dopt,:]
                error_variance_Dopt = np.diag(Psi@np.linalg.inv(Psi.T@C_Dopt.T@C_Dopt@Psi)@Psi.T)
                rmse_reconstruction_Dopt,errorvar_reconstruction_Dopt= signal_reconstruction_regression(Psi,locations_monitored_Dopt,X_test=X_test_proj,X_test_measurements=X_test_proj_noisy,projected_signal=True)
                print(f'Loading alternative sensor placement locations obtained with Dopt method.')

            except:
                    print(f'No Dopt sensor placement file for worse error variance threshold {variance_threshold_ratio:.2f} and num sensors {n_locations_monitored}')
                    errorvar_reconstruction_Dopt = []

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
        
        plots.curve_errorvariance_comparison(errorvar_fullymonitored,errorvar_reconstruction,variance_threshold_ratio,errorvar_fullymonitored.max(),n,n_locations_monitored,errorvar_reconstruction_Dopt,save_fig=True)
        plots.curve_errorvariance_comparison(error_variance_fullymonitored,error_variance_reconstruction,variance_threshold_ratio,error_variance_fullymonitored.max(),n,n_locations_monitored,error_variance_Dopt,save_fig=False)


        
        plt.show()
        sys.exit()