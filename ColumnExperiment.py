#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 16:49:39 2022

@author: yli5
"""
#import sys
#sys.path.append(r'/Users/yli5/Box\ Sync/NREL_documents/PythonWorking/PilotPlant')

from isort import file
from pputils import FrenchPress
import statistics as stat
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from pputils import ignition_query, TagNames
from pputils.miscellaneous import dtrange, indexconvert, df_smooth
from pputils.pilot_plant_general import ScadaObject

from pputils.chetools import WaterViscosity
from scipy import constants

import os
import re

from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from scipy.optimize import leastsq
from scipy.integrate import solve_ivp
from scipy.integrate import odeint

"""
Plot preferences
"""
plt.rc('font', family='serif', serif='Times New Roman')
plt.rc('text', usetex=False)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.rc('axes', labelsize=18)
"""
data processing
"""

markerlist = ['o','v','s','h','X','D','P','^','<','>','p','*']
#%%
localdbpath='./datapickles/'
exp_path='./exp/'

import glob
filenames_abs = glob.glob(exp_path+'*.xlsx')
nexp = len(filenames_abs)
## extract basename of files
filenames = [] # basename of files (with extension)
names = [] # names of experiments (no extension)
for i in range(nexp):
    filename = os.path.basename(filenames_abs[i])
    filenames.append(filename)
    names.append(filename.split('.')[0])
    
    
#%%
class ColumnExperiment:
    """
    packed bed flow-through column expeirment with room temperature water
    """
    baseWaterWt=0.651
    waterDensity = 1 #kg/L
    filenames = ['clean_wet_large_03_11.xlsx',
    'dirty_wet_small_04_22.xlsx', 
    'clean_dry_small_04_21.xlsx', 
    'dirty_dry_large_02_25.xlsx', 
    'clean_wet_small_04_08.xlsx',
    'clean_dry_large_03_19.xlsx', # inserted 0319 exp
    'dirty_dry_small_04_05.xlsx', 
    #'clean_dry_large_03_17.xlsx',   # removed 0317 exp because of huge inconsistency in density with other experiments
    'dirty_dry_unfractionated_05_12.xlsx', 
    'dirty_wet_large_03_05.xlsx',
    ]
    cleanPressureExpDict = {filenames[0]: [0,2],
                            filenames[1]: [4], #[4,5],
                            filenames[2]: [0],
                            filenames[3]: [0,1,3], #[0]
                            filenames[4]: [0,2],
                            filenames[5]: [1,3,4],
                            filenames[6]: [0,1,5],
                            #filenames[5]: [0,1,5],
                            #filenames[6]: [1,3,5],
                            filenames[7]: [2],
                            filenames[8]: [1,4],
                            }
    #cleanPressureExpDict = pd.DataFrame(list(cleanPressureExp.items()))
    cleanVelocityExpDict = {filenames[0]: 5,
                            filenames[1]: 0,
                            filenames[2]: 0,
                            filenames[3]: 3,
                            filenames[4]: 0,
                            filenames[5]: 5,
                            filenames[6]: 1,
                            #filenames[5]: 1,
                            #filenames[6]: 5,
                            filenames[7]: 5,
                            filenames[8]: 6
                            }
    #cleanVelocityExpDict = pd.DataFrame(list(cleanVelocityExp.items()))
    """
    filenames = ['clean_wet_large_03_11.xlsx',
    'dirty_wet_small_04_22.xlsx', 
    'clean_dry_small_04_21.xlsx', 
    'dirty_dry_large_02_25.xlsx', 
    'clean_wet_small_04_08.xlsx',
    'dirty_dry_small_04_05.xlsx', 
    'clean_dry_large_03_17.xlsx', 
    'dirty_dry_unfractionated_05_12.xlsx', 
    'dirty_wet_large_03_05.xlsx']
    cleanPressureExpDict = {filenames[0]: [0,2],
                            filenames[1]: [4], #[4,5],
                            filenames[2]: [0],
                            filenames[3]: [0,1,3], #[0]
                            filenames[4]: [0,2],
                            filenames[5]: [0,1,5],
                            filenames[6]: [1,3,5],
                            filenames[7]: [2],
                            filenames[8]: [1,4]
                            }
    cleanVelocityExpDict = {filenames[0]: 5,
                            filenames[1]: 0,
                            filenames[2]: 0,
                            filenames[3]: 3,
                            filenames[4]: 0,
                            filenames[5]: 1,
                            filenames[6]: 5,
                            filenames[7]: 5,
                            filenames[8]: 6
                            }
    
    """
    
    def __init__(self, filename_abs, localdbpath,cache=True):
        self.name = os.path.basename(filename_abs)
        self.filename_abs = filename_abs
        self.localdbpath = localdbpath
        self.cache = cache
        self.data_pickle_path = os.path.join(localdbpath,'exp/')
        if not os.path.exists(self.data_pickle_path):
            os.makedirs(self.data_pickle_path)
        self.picklefile = os.path.join(self.data_pickle_path, self.name.split('.')[0]+'_exp_data.pkl')
        self.temperature = 20 # Celcius , room temperature
        self.viscosity = WaterViscosity(self.temperature).vd
        self.fp, self.df_flow, self.df_manual, self.start, self.end, self.filename = self.load_exp_data()
        self.result, self.result_all, self.Pc, self.epsilon, self.rou, self.N_exp, self.area, self.radius = self.extract_data(self.fp, self.df_manual, self.df_flow)
    
    def load_exp_data(self, flowsheetname='Flow Tests', manualdatasheet='Manual Data'):
        filename = os.path.basename(str(self.filename_abs))
        name = filename.split('.')[0]
        # load process data
        df_flow = pd.read_excel(self.filename_abs,sheet_name = flowsheetname)
        # drop any trailing empyty cell
        df_flow = df_flow.dropna()
        # drop the last 2 rows in data
        """
        this should be done elsewhere. since the porosity calculation depends on the last data.
        """
        ##df_flow = df_flow.drop(df_flow.index[-1])
        ##df_flow = df_flow.drop(df_flow.index[-1])
        ## df_flow = df_flow.drop(df_flow.index[-1])
        date = df_flow['Date'][0].date()
        starttime = df_flow[df_flow.columns[1]][0]
        endtime = df_flow[df_flow.columns[-1]].iloc[-1]
        # testing : caused by trailing nan value in excel
        #print(df_flow[df_flow.columns[-1]])
        #print(f"type: {type(endtime)}")
        ### import process information from pilot plant SCADA server
        start, end = str(datetime.combine(date,starttime)), str(datetime.combine(date,endtime))
        fp = FrenchPress(name=name)
        fp.load_scada(start, end, localdbpath=localdbpath)
        fp.setmeta(colID=1, dp_conns=((1,4),(0,2),(1,3)))
        fp.calc_performance()
        radius=fp.dia_col/2000 # unit: m
        ## load manual data for porosity and surface area calculation
        df_manual = pd.read_excel(self.filename_abs,sheet_name = manualdatasheet)
        # drop any trailing nan ; default not operated in place 
        # seems dropna() will drop all columns if one of it have nan in that row. Not ideal.
        #df_manual = df_manual.dropna()
        
        return fp, df_flow, df_manual, start, end, filename
    
    def extract_data(self,fp,df_manual,df_flow):
        df = self.extract_data_n_pickle(fp, df_manual,df_flow)
        return df['result'],df['result_all'],df['Pc'].to_numpy(),df['epsilon'].to_numpy(),df['rou'].to_numpy(),df['N_exp'],df['area'],df['radius']


    
    def extract_data_n_pickle(self,fp, df_manual, df_flow):
        if self.cache:      
            if os.path.isfile(self.picklefile):
                print("Importing locally cached experiment result data.")
                return pd.read_pickle(self.picklefile)
            else:
                print("Analyze expeirments.")

        # self.result.append(_extract( dftest,col_name[2*i+1],col_name[2*i+2]))
        baseWaterWt=0.651
        waterDensity = 1 #kg/L
        bedWater = df_manual['Water in Bed (kg)']-baseWaterWt #unit kg
        DryDensity = df_manual['Dry Weight(kg)'][0]
        
        area = np.pi * (fp.dia_col / 1000)**2 / 4 # unit: m2
        radius = fp.dia_col/20000 #unit m
        # include the last data of zero
        result_all = []
        # clean result without final zero data
        result = []
        col_name = list(df_flow.columns)
        N_exp = int((len(col_name)-1)/2)
        Pc = [] #compaction in PSI?
        epsilon = [] #porosity
        rou =[] #density
        for i in range(N_exp):
            startStr = col_name[2*i+1]
            endStr = col_name[2*i+2]
            tStart= df_flow[startStr].astype(str)
            tEnd = df_flow[endStr].astype(str)
            n = len(tStart)
            stdev_sfvel = np.zeros(n)
            avg_sfvel = np.zeros(n)
            stdev_pg = np.zeros(n)
            avg_pg = np.zeros(n)
            pg_1= np.zeros(n)
            pg_2= np.zeros(n)
            std_pg_1= np.zeros(n)
            std_pg_2= np.zeros(n)
            PistonHeight=np.zeros(n)
            BedVolumeU=np.zeros(n)
            for j in range(n):
                #print(f"i={i}, j={j}, start={tStart[j]}, end={tEnd[j]}")
                df = dtrange(fp.process, start=tStart[j], end=tEnd[j])
                df = indexconvert(df, 's')
                tag = 'column bulk flux'
                stdev_sfvel[j] = stat.stdev(df[tag]) #mm/s
                avg_sfvel[j] = stat.mean(df[tag]) #mm/s
                stdev_pg[j] = np.std([df[fp.tags_grad[1]],df[fp.tags_grad[2]]]) # new pputils takes care of this(used to need devide 1e6 to get kPa/m)
                avg_pg[j] = np.average([df[fp.tags_grad[1]],df[fp.tags_grad[2]]]) #kPa/m
                pg_1[j] = np.average(df[fp.tags_grad[1]])
                pg_2[j] = np.average(df[fp.tags_grad[2]])
                std_pg_1[j] = np.std(df[fp.tags_grad[1]])
                std_pg_2[j] = np.std(df[fp.tags_grad[2]])
                PistonHeight[j]=np.mean(df['bed volume'])/area
                BedVolumeU[j]=np.mean(df['bed volume'])
            avg_sfvel = (avg_sfvel - avg_sfvel[0]) #mm/s
            # correct pressure measurments
            avg_pg = avg_pg - avg_pg[0] 
            bedVolume = np.mean(df['bed volume']) # porosity was measured at the end of experiment
            #bedVolume = BedVolumeU[0] # use velocity at 0
            bedHeight = bedVolume/(1000*area)
            epsilon_temp= bedWater[i]/(waterDensity*bedVolume) #unit L/L (porosity)
            packingDensity = 1000*DryDensity/bedVolume # density (for compressibility) unit: kg/m3
            # use temp variable
            Pc_temp = np.mean(df['column piston stress']) # unit kPa
            ##
            Pc.append(Pc_temp)
            epsilon.append(epsilon_temp)
            rou.append(packingDensity)
            result_all.append(dict(avg_sfvel = avg_sfvel,stdev_sfvel=stdev_sfvel, avg_pg = avg_pg, 
                                    stdev_pg = stdev_pg,pg_1 = pg_1,std_pg_1=std_pg_1, pg_2 = pg_2,
                                    std_pg_2=std_pg_2,bedVolume = bedVolume,bedHeight=bedHeight, 
                                    PistonHeight=PistonHeight,BedVolumeU=BedVolumeU,epsilon=epsilon,
                                    packingDensity=packingDensity,Pc = Pc_temp))
            result.append(dict(avg_sfvel = avg_sfvel[:-1],stdev_sfvel=stdev_sfvel[:-1], avg_pg = avg_pg[:-1], 
                                    stdev_pg = stdev_pg[:-1],pg_1 = pg_1[:-1],std_pg_1=std_pg_1[:-1], pg_2 = pg_2[:-1],
                                    std_pg_2=std_pg_2[:-1],bedVolume = bedVolume,bedHeight=bedHeight, 
                                    PistonHeight=PistonHeight[:-1],BedVolumeU=BedVolumeU[:-1],epsilon=epsilon,
                                    packingDensity=packingDensity,Pc = Pc_temp))
        Pc = np.array(Pc)
        epsilon = np.array(epsilon)
        rou = np.array(rou)

        #self.area = area # m2
        #self.radius = radius # m

        ## consider pickle result
        data_all = dict(result=result, result_all=result_all,Pc=Pc,epsilon=epsilon,rou=rou,N_exp=N_exp,area=area,radius=radius)
        df_data_all = pd.DataFrame(data=data_all)
        #df.to_pickle(localdbpath,)
        if self.cache:
            # save a local cache where specified (path integrity checks already done above)
            df_data_all.to_pickle(self.picklefile)
            print("Saved a local SCADA data cache at {}".format(self.picklefile))
        return df_data_all
        #return result, result_all, Pc, epsilon, rou, N_exp

    def clean_data(self):
        keys = ["avg_sfvel","stdev_sfvel","avg_pg","stdev_pg","pg_1","std_pg_1", "pg_2","std_pg_2","PistonHeight","BedVolumeU" ]
                                   
        self.result_all_nozero = self.result
        self.Pc_all = self.Pc
        self.epsilon_all = self.epsilon
        self.rou_all = self.rou
        self.N_exp_all = self.N_exp

        result = []
        p_del_idx = ColumnExperiment.cleanPressureExpDict[self.name]
        v_del_num = ColumnExperiment.cleanVelocityExpDict[self.name]
        for i, pressure in enumerate(self.result):
            # remove compaction pressure experiment entries
            if i not in p_del_idx:
                temp = pressure
                if v_del_num !=0:
                    for key in keys:
                        # remove velocity experiment entries
                        temp[key] = pressure[key][:-v_del_num]
                result.append(temp)
        
        self.result = result
        if p_del_idx != [None]:
            self.Pc = np.delete(self.Pc,p_del_idx,0)
            self.epsilon = np.delete(self.epsilon,p_del_idx,0)
            self.rou = np.delete(self.rou,p_del_idx,0)
            self.N_exp = len(self.Pc)

#%%       
if __name__ == '__main__':
       data = ColumnExperiment(filenames_abs[0], localdbpath) 
       print(type(data.result))
       data.clean_data()

