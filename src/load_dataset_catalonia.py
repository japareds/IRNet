#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 17:21:04 2023

@author: jparedes
"""
import os
import pandas as pd
import numpy as np


""" Load Catalonia dataset """
class DataSet():
    def __init__(self,pollutant,start_date,end_date,files_path):
        """
        Network reference stations. Each station has an assigned ID

        Parameters
        ----------
        pollutant : str
            pollutant measured by reference stations
        start_date : str
            starting date of measurements in 'YYYY-MM-DD'
        end_date : str
            end date of measurements in 'YYYY-MM-DD'
        files_path: str
            path to files
        """
        self.stations_list = {
                       'Badalona':'08015021',
                       'Eixample':'08019043',
                       'Gracia':'08019044',
                       'Ciutadella':'08019050',
                       'Vall-Hebron':'08019054',
                       'Palau-Reial':'08019057',
                       'Fabra':'08019058',
                       'Berga':'08022006',
                       'Gava':'08089005',
                       'Granollers':'08096014',
                       'Igualada':'08102005',
                       'Manlleu':'08112003',
                       'Manresa':'08113007',
                       'Mataro':'08121013',
                       'Montcada':'08125002',
                       #'Montseny':'08125002', # linearly dependent to other measurements
                       'El-Prat':'08169009',
                       'Rubi':'08184006',
                       'Sabadell':'08187012',
                       'Sant-Adria':'08194008',
                       'Sant-Celoni':'08202001',
                       'Sant-Cugat':'08205002',
                       'Santa-Maria':'08259002',
                       'Sant-Vicenç':'08263001',
                       'Terrassa':'08279011',
                       'Tona':'08283004',
                       'Vic':'08298008',
                       'Viladecans':'08301004',
                       'Vilafranca':'08305006',
                       'Vilanova':'08307012',
                       'Agullana':'17001002',
                       'Begur':'17013001',
                       'Pardines':'17125001',
                       'Santa-Pau':'17184001',
                       'Bellver':'25051001',
                       'Juneda':'25119002',
                       'Lleida':'25120001',
                       'Ponts':'25172001',
                       'Montsec':'25196001',
                       'Sort':'25209001',
                       'Alcover':'43005002',
                       'Amposta':'43014001',
                       'La-Senla':'43044003',
                       'Constanti':'43047001',
                       'Gandesa':'43064001',
                       'Els-Guiamets':'43070001',
                       'Reus':'43123005',
                       'Tarragona':'43148028',
                       'Vilaseca':'43171002'
                       }
        
        self.RefStations = [i for i in self.stations_list.keys()]
        self.pollutant = pollutant # pollutant to load
        self.startDate = start_date 
        self.endDate = end_date 
        self.files_path = files_path
        self.ds = pd.DataFrame()
        
        
    def load_dataSet(self,n_years=12,n_stations=100):
        """
        Load csv files containing reference stations measurements for specified period of time

        Returns
        -------
        self.ds: pandas dataframe
            dataframe containing measurements: [num_dates x num_stations]

        """
        # load real dataset
        for rs in self.RefStations:
            fname = f'{self.files_path}{self.pollutant}_{rs}_{self.startDate}_{self.endDate}.csv'
            print(f'Loading data set {fname}')
            df_ = pd.read_csv(fname,index_col=0,sep=';')
            df_.index = pd.to_datetime(df_.index)
            self.ds = pd.concat([self.ds,df_],axis=1)
        self.ds = self.ds.drop_duplicates(keep='first')
        print(f'All data sets loaded\n{self.ds.shape[0]} measurements for {self.ds.shape[1]} reference stations')

    def save_dataset(self):
        fname = f'{self.files_path}{self.pollutant}_catalonia_{self.startDate}_{self.endDate}.csv'
        self.ds.to_csv(fname,sep=',',index=True)
        print(f'dataset saved to {fname}')

if __name__ == '__main__':
    
    abs_path = os.path.dirname(os.path.realpath(__file__))
    files_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/files/catalonia/'
        
    pollutant = 'O3'
    start_date = '2011-01-01'
    end_date = '2022-12-31'
    
    dataset = DataSet(pollutant, start_date, end_date, files_path)
    dataset.load_dataSet()
    print(f'Merged dataset with {dataset.ds.shape[1]} locations and {dataset.ds.shape[0]} measurements:\n{dataset.ds.head()}')
    dataset.save_dataset()
