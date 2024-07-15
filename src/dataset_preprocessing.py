#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 17:21:04 2023

@author: jparedes
"""
import os
import pandas as pd
import numpy as np

""" Dataset preprocessing"""

class Dataset():
    def __init__(self,pollutant,start_date,end_date,files_path):
        self.pollutant = pollutant
        self.start_date = start_date
        self.end_date = end_date
        self.files_path = files_path
    def load_dataset(self):
        fname = f'{self.files_path}{self.pollutant}_catalonia_{self.start_date}_{self.end_date}.csv'
        self.ds = pd.read_csv(fname,sep=',',index_col=0)
        self.ds.index = pd.to_datetime(self.ds.index)
        print(f'Loaded dataset with {self.ds.shape[0]} measurements for {self.ds.shape[1]} locations.\n{self.ds.head()}')

    def time_window(self,start_date='2018',end_date='2023'):
        """ Time windowing of measurements. Fabra reference stations has measurements from 2018"""
        date_range = pd.date_range(start=start_date,end=end_date,freq='H')
        self.ds = self.ds.loc[np.isin(self.ds.index,date_range)]
        
    def cleanMissingvalues(self,strategy='remove',tol=0.1):
        """
        Remove missing values from data set.
        Three possibilities: 
            1) remove stations with not enough measurements
            2) drop missing values for all stations
            3) interpolate missing values (linear)

        Parameters
        ----------
        strategy : str, optional
            Strategy for dealing with missing values. The default is 'remove'.
        tol : float, optional
            Fraction of missing values for removing the whole station. The default is 0.1.
        
        Returns
        -------
        None.

        """
        print(f'Percentage of missing values:\n{100*self.ds.isna().sum()/self.ds.shape[0]}')
        if strategy=='stations':
            print(f'Removing stations with high percentage of missing values (tol={tol})')
            refstations = self.ds.columns
            mask = self.ds.isna().sum()/self.ds.shape[0]>tol
            idx = [i[0] for i in np.argwhere(mask.values)]
            refst_remove = [refstations[i] for i in idx]
            self.ds = self.ds.drop(columns=refst_remove)
            
        if strategy == 'remove':
            print('Removing missing values')
            self.ds.dropna(inplace=True)
            print(f'Entries with missing values remaining:\n{self.ds.isna().sum()}')
            print(f'{self.ds.shape[0]} remaining measurements')
            
        elif strategy == 'interpolate':
            print('Interpolating missing data')
            self.ds = self.ds.interpolate(method='linear')
            print(f'Entries with missing values remiaining:\n{self.ds.isna().sum()}')
    
    def save_dataset(self):
        fname = f'{self.files_path}{self.pollutant}_catalonia_clean_N{self.ds.shape[1]}_{self.start_date}_{self.end_date}.csv'
        self.ds.to_csv(fname,sep=',')
        print(f'Dataset saved to {fname}')


if __name__ == '__main__':
    abs_path = os.path.dirname(os.path.realpath(__file__))
    files_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/files/catalonia/'
    pollutant = 'O3'
    start_date = '2011-01-01'
    end_date = '2022-12-31'

    dataset = Dataset(pollutant,start_date,end_date,files_path)
    dataset.load_dataset()
    #dataset.time_window(start_date='2018',end_date='2023')
    dataset.cleanMissingvalues(strategy='stations',tol=0.1)
    print(f'New dataset dataset has {dataset.ds.shape[0]} measurements for {dataset.ds.shape[1]} locations.\n{dataset.ds.head()}')
    #dataset.cleanMissingvalues(strategy='interpolate')
    dataset.cleanMissingvalues(strategy='remove')
    print(f'New dataset dataset has {dataset.ds.shape[0]} measurements for {dataset.ds.shape[1]} locations.\n{dataset.ds.head()}')
    dataset.save_dataset()





