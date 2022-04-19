#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 16:49:39 2022

@author: yli5
"""
# change to Dave's newest repo
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
from engtools import dtrange, indexconvert, df_smooth, WaterViscosity, WaterDensity
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
# localdbpath='/Users/yli5/Box Sync/NREL_documents/PythonWorking/PilotPlant/FCICDataAnalyze/AirClassified/YLAnalysis/datapickles'

# name, start, end = 'F53_biomass_asMillWM09042020', '2020-09-04 10:30', '2020-09-04 17:30'
# fp = FCIC.FrenchPress(start, end, name=name, localdbpath=localdbpath)
# #fp.setmeta(colID=1, dp_conns=((1,4),(2,4),(3,4)))
# fp.setmeta(colID=1, dp_conns=((1,4),(1,3),(2,4)))
# #    fp.importmeta()
# fp.performance()
# plt.close('all')
# fp.plt_time()
# fp.plt_perf()

"""
Clip the data, calculate average, standardDeviations
"""
localdbpath='/Users/yli5/Box Sync/NREL_documents/PythonWorking/PilotPlant/FCICDataAnalyze/AirClassified/YLAnalysis/datapickles/'
exp_path='/Users/yli5/Box Sync/NREL_documents/PythonWorking/PilotPlant/FCICDataAnalyze/AirClassified/YLAnalysis/exp/'

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
def load_exp_data(filename_abs, localdbpath,flowsheetname='Flow Tests', manualdatasheet='Manual Data'):
    filename = os.path.basename(str(filename_abs))
    name = filename.split('.')[0]
    # load process data
    df_flow = pd.read_excel(filename_abs,sheet_name = flowsheetname)
    # drop the last 2 rows in data
    df_flow = df_flow.drop(df_flow.index[-1])
    df_flow = df_flow.drop(df_flow.index[-1])
    # df_flow = df_flow.drop(df_flow.index[-1])
    date = df_flow['Date'][0].date()
    starttime = df_flow[df_flow.columns[1]][0]
    endtime = df_flow[df_flow.columns[-1]].iloc[-1]
    ### import process information from pilot plant SCADA server
    start, end = str(datetime.combine(date,starttime)), str(datetime.combine(date,endtime))
    fp = FrenchPress(name=name)
    fp.load_scada(start, end, localdbpath=localdbpath)
    fp.setmeta(colID=1, dp_conns=((1,4),(0,2),(1,3)))
    fp.calc_performance()
    radius=fp.dia_col/2000 # unit: m
    ## load manual data for porosity and surface area calculation
    df_manual = pd.read_excel(filename_abs,sheet_name = manualdatasheet)
    
    return fp, df_flow, df_manual, start, end, filename, name

fp, df_flow, df_manual, start, end,filename,name = load_exp_data(filenames_abs[3], localdbpath)

#%% test
# col_name = list(df_flow.columns)
# i = 3
# j = 5
# startStr = col_name[2*i+1]
# endStr = col_name[2*i+2]
# tStart= df_flow[startStr].astype(str)
# tEnd = df_flow[endStr].astype(str)
# df_temp = dtrange(fp.process, start=tStart[j], end=tEnd[j])[0]

#%%
def extract_data(fp,df_manual,df_flow):
    # self.result.append(_extract( dftest,col_name[2*i+1],col_name[2*i+2]))
    baseWaterWt=0.651
    waterDensity = 1 #kg/L
    bedWater = df_manual['Water in Bed (kg)']-baseWaterWt #unit kg
    DryDensity = df_manual['Dry Weight(kg)'][0]
    
    area = np.pi * (fp.dia_col / 1000)**2 / 4 # unit: m2
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
            print(f"i={i}, j={j}, start={tStart[j]}, end={tEnd[j]}")
            df = dtrange(fp.process, start=tStart[j], end=tEnd[j])[0]
            df = indexconvert(df, 's')
            tag = 'column bulk flux'
            stdev_sfvel[j] = stat.stdev(df[tag])*1000 #mm/s
            avg_sfvel[j] = stat.mean(df[tag])*1000 #mm/s
            stdev_pg[j] = np.std([df[fp.tags_grad[1]],df[fp.tags_grad[2]]])/1e6 # 1e3 Pa/m # 1e6 kPa/m
            avg_pg[j] = np.average([df[fp.tags_grad[1]],df[fp.tags_grad[2]]])/1e6 #1e3 Pa/m # 1e6 kPa/m
            pg_1[j] = np.average(df[fp.tags_grad[1]])/1e6
            pg_2[j] = np.average(df[fp.tags_grad[2]])/1e6
            std_pg_1[j] = np.std(df[fp.tags_grad[1]])/1e6
            std_pg_2[j] = np.std(df[fp.tags_grad[2]])/1e6
            PistonHeight[j]=np.mean(df['bed volume'])/area
            BedVolumeU[j]=np.mean(df['bed volume'])
        avg_sfvel = (avg_sfvel - avg_sfvel[0]) #mm/s
    #    pg_1 = pg_1 - pg_1[0]
    #    pg_2 = pg_2 - pg_2[0]
    
        # correct pressure measurments
        avg_pg = avg_pg - avg_pg[0] 
        bedVolume = np.mean(df['bed volume']) # porosity was measured at the end of experiment
        bedHeight = bedVolume/(1000*area)
        epsilon_temp= bedWater[i]/(waterDensity*bedVolume) #unit L/L (porosity)
        packingDensity = 1000*DryDensity/bedVolume # density (for compressibility) unit: kg/m3
        # use temp variable
        Pc_temp = np.mean(df['column piston stress']) # unit kPa
        ##
        Pc.append(Pc_temp)
        epsilon.append(epsilon_temp)
        rou.append(packingDensity)
        result.append(dict(avg_sfvel = avg_sfvel,stdev_sfvel=stdev_sfvel, avg_pg = avg_pg, 
                                stdev_pg = stdev_pg,pg_1 = pg_1,std_pg_1=std_pg_1, pg_2 = pg_2,
                                std_pg_2=std_pg_2,bedVolume = bedVolume,bedHeight=bedHeight, 
                                PistonHeight=PistonHeight,BedVolumeU=BedVolumeU,epsilon=epsilon,
                                packingDensity=packingDensity,Pc = Pc))
    Pc = np.array(Pc)
    epsilon = np.array(epsilon)
    rou = np.array(rou)
    
    return result, Pc, epsilon, rou, N_exp

result, Pc, epsilon, rou, N_exp= extract_data(fp, df_manual, df_flow)


#%% Bed information
g = 9.8 # gravitational 
rouf = 1000 # water density unit kg/m3
lamda = 0.5
miu = 0.1
radius = fp.dia_col/2000 # unit: m
Area = np.pi * (fp.dia_col / 1000)**2 / 4 # unit: m2
solids = 0.2

BedHeight = [] #unit: m
# N_exp = int((len(col_name)-1)/2)
for i in range(N_exp):
    BedHeight.append(result[i]['bedVolume']/(1000*Area))     
#%%
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)

for i in range(N_exp):
#    label = 'Pc = ' + str(result[i]['Pc']) + ' kPa'
    label = 'Pc = %0.2f kPa' % (Pc[i])
#    plt.plot(result[i]['avg_sfvel'],result[i]['pg_2'],label=label) #pg_2
    plt.errorbar(result[i]['avg_sfvel'],result[i]['avg_pg'],xerr=result[i]['stdev_sfvel'],yerr=result[i]['stdev_pg'],\
                 color='C'+str(i),markersize = 4, fmt='-s',label=label, capsize= 5, capthick=1)
plt.legend(loc='upper left')
#ax.set_xlim(0, 6)
#ax.set_ylim(0, 35)
#ax.xaxis.set_ticks(np.arange(0, xend, 1))
#fig.savefig('AsMilledCob-DPdz.jpg',dpi=300,bbox_inches='tight')
#
#%% select data for modelling:
idx = [0,1,2,3,4]

# all data:
idx = None
if idx != None:
    Pc_fitting = Pc[idx]
    result_fitting = [result[i] for i in idx]
    epsilon_fitting = epsilon[idx]
else:
    Pc_fitting = Pc
    result_fitting = result
    epsilon_fitting = epsilon

#%%
packingDensity = [result_fitting [i]['packingDensity'] for i in range(len(result_fitting))] # unit: kg/m3

from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from scipy.optimize import leastsq

def funcDensityMini(x,p,y):
    return x[0]+x[1]*pow(p,x[2])-y  #c0+M*pow(p,N)
def funcDensity(x,p):
    return x[0]+x[1]*pow(p,x[2])

def fitDensity(x0,Pdata,ydata,plot=True,save=False):
    #popt, pcov = curve_fit(funcDensityMini1, xdata, ydata)
    res_lsq=least_squares(funcDensityMini,x0, loss='soft_l1', f_scale=50, args=(Pdata,ydata))
    densityCoeff = res_lsq.x
    if plot:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1)

        P_model = np.linspace(0,40,40) 
        plt.plot(Pdata,ydata, 's',linestyle='',markersize = 12, label='Experiment')
        plt.plot(P_model, funcDensity(densityCoeff, P_model), 'C0-', label='Model',linewidth = 3)

        plt.legend(loc = 'upper left',prop={'size': 18, 'weight':'bold'},frameon=False,)
        ax.set_xlabel('$\mathregular{P_c}$ (kPa)')
        ax.set_ylabel('Bulk density ($\mathregular{kg/m^3}$)')
        # ax.set_xlim(0, 40)
        # ax.set_ylim(80, 180)
        ax.tick_params(axis = 'y', which = 'major', direction = 'out', length = 6.0, width = 2.0)
        ax.tick_params(axis = 'x', which = 'major', direction = 'out', length = 6.0, width = 2.0)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.8)
        fig.tight_layout()
        plt.show()
    if save:
        fig.savefig('./asMillWM/asMillWM_WetCompressibility-Pc.jpg',dpi=300,bbox_inches='tight')
    return densityCoeff


x0 = [1,1,1]

Pdata=Pc_fitting #pressure Pa
ydata=packingDensity #kg/m3

densityCoeff = fitDensity(x0,Pdata,ydata)

#%% export data
DataToPlotDensity=np.stack((np.round(Pdata,2),ydata))

#%%
def funcEpsMini2(x,p,y):
    return x[0]/(1+x[1]*p)-y
def funcEps2(x,p):
    return x[0]/(1+x[1]*p)


def fitEpsilon(x0,Pdata,ydata,plot=True,save=False):
    res_lsq=least_squares(funcEpsMini2,x0, loss='soft_l1', f_scale=50, args=(Pdata,ydata))
    EpsCoeff2 = res_lsq.x
    
    PcPlot=Pdata
    epsPlot=ydata
    PcModel = np.linspace(0,40,1000)
    if plot:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1)
        plt.plot(PcPlot,epsPlot,'C1s',markersize = 12,label='Experiment')
        plt.plot(PcModel, funcEps2(EpsCoeff2, PcModel), 'C1-',label='Model',linewidth = 3)
        plt.legend(loc = 'upper right',prop={'size': 18, 'weight':'bold'},frameon=False,)
        ax.set_xlabel('$\mathregular{P_c}$ (kPa)')
        ax.set_ylabel('Effective porosity (-)')
        ax.set_xlim(0, 40)
        ax.set_ylim(0, 0.4)
        ax.tick_params(axis = 'y', which = 'major', direction = 'out', length = 6.0, width = 2.0)
        ax.tick_params(axis = 'x', which = 'major', direction = 'out', length = 6.0, width = 2.0)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.8)
        fig.tight_layout()
        plt.show()
    if save:
        fig.savefig('./asMillWM/asMillWM_PorosityDensityCorrelation.jpg',dpi=300,bbox_inches='tight')
    return EpsCoeff2

x0 = np.ones(2)
Pdata=Pc_fitting #pressure Pa
epsydata=epsilon_fitting #kg/m3
EpsCoeff2 = fitEpsilon(x0,Pdata,epsydata)
#%%
DataToPlotDensityPorosity=np.vstack((DataToPlotDensity,epsydata))

#%%
#Kozeny-Carman equation
#constants
k = 5.55 # Kozeny-factor . 5.55 is for fibrous randomly packed bed
temperature = 20 # oC
viscosity = WaterViscosity(temperature).vd

def funcMini2(x,U,y,eps):
    S=x[0]
#    eps=funcEps2(EpsCoeff2,Pc[i]) #epsilon Pc corelation
#    return k*S*S*U*viscosity*pow((1-eps),2)/pow(eps,3)-y
    return k*S*S*U*viscosity*pow((1-eps),2)/(100*pow(eps,3))-y

def func2(x,U,eps):
    S=x[0]
#    eps=funcEps2(EpsCoeff2,Pc[i]) #epsilon Pc corelation
#    return k*S*S*U*viscosity*pow((1-eps),2)/pow(eps,3)
    return k*S*S*U*viscosity*pow((1-eps),2)/(100*pow(eps,3))

#x0 = np.ones(4)*12

def Sv_single_fit(x0,n_fit,Pc,result,EpsCoeff2,plot=True, save=False):
    eps_nfit=funcEps2(EpsCoeff2,Pc[n_fit])
    Udata=result[n_fit]['avg_sfvel'] #mm/s
    #ydata=result[n_fit]['pg_1']#kPa/m
    ydata=result[n_fit]['avg_pg']#kPa/m
    #ydata=result[n_fit]['pg_2']#kPa/m
    
    res_lsq2=least_squares(funcMini2,x0, loss='soft_l1', f_scale=0.1, args=(Udata,ydata,eps_nfit))
    
    if plot:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1)
        
        # 03-11 data
        for i in range(len(Pc)):
        # for i in range(1,N_exp):
            label = '$\mathregular{P_c}$ = %0.2f kPa' % (Pc[i])
            eps_kc=funcEps2(EpsCoeff2,Pc[i])
            plt.errorbar(result[i]['avg_sfvel'],result[i]['avg_pg'],\
                         xerr=result[i]['stdev_sfvel'],yerr=result[i]['stdev_pg'],\
                         color='C'+str(i),marker = markerlist[i], markersize = 12,\
                         fmt='s',label=label, capsize= 6, capthick=2)
            plt.plot(result[i]['avg_sfvel'], func2(res_lsq2.x, result[i]['avg_sfvel'],eps_kc),'C'+str(i)+'-',linewidth = 3)
        plt.legend(loc = 'upper left',prop={'size': 16, 'weight':'bold'},frameon=False,)
        ax.set_xlabel('Superficial velocity (mm/s)')
        ax.set_ylabel('dP/dz (kPa/m)') 
        # ax.set_xlim(0, 2)
        # ax.set_ylim(0, 10)
        ax.tick_params(axis = 'y', which = 'major', direction = 'out', length = 6.0, width = 2.0)
        ax.tick_params(axis = 'x', which = 'major', direction = 'out', length = 6.0, width = 2.0)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.8)
        fig.tight_layout()
        plt.show()
        
    if save:
        fig.savefig('./asMillWM/traditionalKC_Exp_noMatch_model2.jpg',dpi=300,bbox_inches='tight')
        
    return res_lsq2.x
    
x0 = np.ones(1)*0.01
Sv_single = Sv_single_fit(x0,0,Pc_fitting,result_fitting,EpsCoeff2)
#%% fit Specific surface area    This is compressible Kozeny-Carman equation
#####second form non-exponential
def funcSvMini2(x,p,y):
    return x[0]/(1+x[1]*p)-y
def funcSv2(x,p):
    return x[0]/(1+x[1]*p)

def fitSv(x0,Pdata,result,EpsCoeff,plot=True,save=False):
    # Pdata should skip first experiment
    # fit for Sv @ each P
    N_Svfit=len(Pdata)
    specificSurfaceArea=np.zeros(N_Svfit)
    for i in range(N_Svfit):
    #    n_fit = i
        eps_kc=funcEps2(EpsCoeff,Pdata[i])
        x_Sv = np.ones(1)*0.01
        # Udata=result[i+1]['avg_sfvel'] #mm/s  # here i+1 to skip first experiment
        # ydata=result[i+1]['avg_pg']#Pa/m
        Udata=result[i]['avg_sfvel'] #mm/s  # here i+1 to skip first experiment
        ydata=result[i]['avg_pg']#Pa/m
        
        res_lsq=least_squares(funcMini2,x_Sv, loss='soft_l1', f_scale=0.1, args=(Udata,ydata,eps_kc))
        specificSurfaceArea[i]=res_lsq.x[0]
    
    ydata=specificSurfaceArea
    PcPlot=np.linspace(0,40,1000)
    
    res_lsqSv2=least_squares(funcSvMini2,x0, loss='soft_l1', f_scale=0.1, args=(Pdata,ydata))
    SvCoeff2 = res_lsqSv2.x
    
    print(ydata)
    
    if plot:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1)
        
        plt.plot(Pdata,ydata,'C3s',markersize = 12,label='Experiment')
        plt.plot(PcPlot, funcSv2(SvCoeff2, PcPlot), 'C3-',label='Model',linewidth = 3)
        
        plt.legend(loc = 'upper right',prop={'size': 18, 'weight':'bold'},frameon=False,)
        ax.set_xlabel('$\mathregular{P_c}$ (kPa)')
        ax.set_ylabel('Effective specific surface area ($\mathregular{cm^2 cm^{-3}}$)')
        #ax.set_xlim(0, 40)
        #ax.set_ylim(20, 80)
        ax.tick_params(axis = 'y', which = 'major', direction = 'out', length = 6.0, width = 2.0)
        ax.tick_params(axis = 'x', which = 'major', direction = 'out', length = 6.0, width = 2.0)      
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.8)
        fig.tight_layout()
        plt.show()
    if save:
        fig.savefig('./asMillWM/asMillWM_SvEpsModel2_correct.jpg',dpi=300,bbox_inches='tight')
    return SvCoeff2

#x0 = np.ones(2)*1000
#x0 = [50,0.1]
x0 = np.ones(2)*0.0


Pcdata=Pc_fitting #unit Pa

SvCoeff2= fitSv(x0,Pcdata,result_fitting,EpsCoeff2)
#%%
DataToPlotSv=funcSv2(SvCoeff2, Pcdata)
DataToPlotDensPoroSv=np.vstack((DataToPlotDensityPorosity,DataToPlotSv))
#%%
dfDataToPlotDensPoroSv = pd.DataFrame(DataToPlotDensPoroSv,index=['Pc','Density(kg/m3)','Porosity','EffectiveSpecificSurfaceArea(cm2/cm3)'])
dfDataToPlotDensPoroSv = dfDataToPlotDensPoroSv.transpose()
#dfDataToPlotDensPoroSv.to_csv('Data_asMillWM_BedProperties.csv',sep=' ')
#%%

def funcFinal2(U,i):
    S=funcSv2(SvCoeff2,Pc[i])
    eps=funcEps2(EpsCoeff2,Pc[i])
    return k*S*S*U*viscosity*pow((1-eps),2)/(100*pow(eps,3))

def modifiedKCValidation(EpsCoeff,SvCoeff,Pdata,result, save=False):
    def funcFinal(EpsCoeff,SvCoeff,U,i):
        # S=funcSv2(SvCoeff,Pdata[i-1])
        # eps=funcEps2(EpsCoeff,Pdata[i-1])
        S=funcSv2(SvCoeff,Pdata[i])
        eps=funcEps2(EpsCoeff,Pdata[i])
        return k*S*S*U*viscosity*pow((1-eps),2)/(100*pow(eps,3))
    
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    handles, labels = ax.get_legend_handles_labels()
    N_exp=len(Pdata)
    for i in range(N_exp):
        label = '$\mathregular{P_c}$ = %0.2f kPa' % (Pdata[i])
    #    plt.plot(result[i]['avg_sfvel'],result[i]['pg_1'])
        plt.errorbar(result[i]['avg_sfvel'],result[i]['avg_pg'],\
                     xerr=result[i]['stdev_sfvel'],yerr=result[i]['stdev_pg'],\
                     color='C'+str(i),marker=markerlist[i],markersize = 12, fmt='s',label=label, capsize= 6, capthick=2)
    #    plt.plot(result[i]['avg_sfvel'],result[i]['pg_2'])
        # plt.plot(result[i]['avg_sfvel'], funcFinal2(result[i]['avg_sfvel'],i), 'C'+str(i)+'-',linewidth = 3)
        plt.plot(result[i]['avg_sfvel'], funcFinal(EpsCoeff,SvCoeff,result[i]['avg_sfvel'],i), 'C'+str(i)+'-',linewidth = 3)
              
    plt.legend(loc = 'upper left',prop={'size': 16, 'weight':'bold'},frameon=False,)
    ax.set_xlabel('Superficial velocity (mm/s)')
    ax.set_ylabel('dP/dz (kPa/m)') 
    # ax.set_xlim(0, 2)
    # ax.set_ylim(0, 80)
    ax.tick_params(axis = 'y', which = 'major', direction = 'out', length = 6.0, width = 2.0)
    ax.tick_params(axis = 'x', which = 'major', direction = 'out', length = 6.0, width = 2.0)
    
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.8)
    
    fig.tight_layout()
    plt.show()
    if save:
        fig.savefig('./asMillWM/asMillWM_modifiedKC_svModel2_correct.jpg',dpi=300,bbox_inches='tight')

Pcdata=Pc_fitting
modifiedKCValidation(EpsCoeff2,SvCoeff2,Pcdata,result_fitting)

#%% Export data
#DataToPlot=[]
#indexD=[]
#for i in range(1,N_exp):
#    DataToPlot.append(result[i]['avg_sfvel'])
#    DataToPlot.append(result[i]['avg_pg'])
#    DataToPlot.append(funcFinal2(result[i]['avg_sfvel'],i))
#    indexD.append(round(result[i]['Pc'],2))
#    indexD.append(round(result[i]['Pc'],2))
#    indexD.append(round(result[i]['Pc'],2))
#
#
#dfDataToPlot = pd.DataFrame(DataToPlot,index=indexD)
#dfDataToPlot = dfDataToPlot.transpose()
#dfDataToPlot.to_csv('Data_asMillWM.csv',sep=' ')
#with pd.ExcelWriter("Data_asMillWM.xlsx", engine="openpyxl",mode='a') as writer: 
#    dfDataToPlot.to_excel(writer,sheet_name='Test') 

#%%
### not in use
A = np.pi * (fp.dia_col / 1000)**2 / 4 # unit: m2 , column cross-section area
H = 1 # Height: unit meters
Volume = A*H # m3
#####
def getBedProperties(frictionCoeff,U,BedHeight,radius,yinit,DensityCoeff = densityCoeff,\
                     epsCoeff=EpsCoeff2,svCoeff=SvCoeff2,scaleUp = False,\
                     solids = 0.17,N=1000,g=9.8,rouf = 1000.0,viscosity=0.00089,lamda = 0.5,plot = False):
    miu = frictionCoeff
    U_f = U # unit: mm/s
    Height = BedHeight # unit m
    if scaleUp:
        tspan=np.linspace(0,Height,N)
    else:
        tspan=np.linspace(Height,0,N)
    c=[]

    def funcDpDz2(U,sigma):
        y=sigma
        S=funcSv2(svCoeff,y)
        eps=funcEps2(epsCoeff,y)
        return k*S*S*U*viscosity*pow((1-eps),2)/(100*pow(eps,3))
    
    def s2(t,y,c): # y (sigma) in kPa
        yA=y
        density=funcDensity(densityCoeff, yA)
        eps=funcEps2(epsCoeff,yA)
        m=solids*(1-eps)
        DGBouyantDh=m*(density/m-rouf)*g/1000# converted to kPa
        DfsDh = funcDpDz2(U_f,yA)
        DfwDh = (2*lamda*miu*yA/radius) # yA has unit of kPa
        if scaleUp:
            DsigmaDh = DfsDh - DGBouyantDh - DfwDh
        else:
            DsigmaDh = DfsDh - DGBouyantDh + DfwDh
        return DsigmaDh
    def s3(y,t): # y (sigma) in kPa
        yA=y
        density=funcDensity(densityCoeff, yA)
        eps=funcEps2(epsCoeff,yA)
        
        m=solids*(1-eps)
        DGBouyantDh=m*(density/m-rouf)*g/1000# converted to kPa
        DfsDh = funcDpDz2(U_f,yA)
        DfwDh = (2*lamda*miu*yA/radius) # yA has unit of kPa
        if scaleUp:
            DsigmaDh = DfsDh - DGBouyantDh - DfwDh
        else:
            DsigmaDh = DfsDh - DGBouyantDh + DfwDh
        
        return DsigmaDh
  
#    sol2 = solve_ivp(lambda t, y: s2(t, y, c),
#                    [tspan[0], tspan[-1]], yinit,method='Radau', t_eval=tspan,first_step=Height/(5*N),max_step=Height/(3*N),rtol=1e-9,atol=1e-12)
#    sol2 = solve_ivp(lambda t, y: s2(t, y, c),
#                    [tspan[0], tspan[-1]], yinit, t_eval=tspan)
    sol2 = odeint(s3,yinit,tspan,hmax=Height/(1*N))
    sol2 = np.array(sol2).flatten()
    
#    sigmaAbs = abs(sol.y[0])
#    densityProfile=funcDensity(DensityCoeff, sigmaAbs)
#    epsilonProfile=funcEps(epsCoeff, sigmaAbs)
#    SvProfile = funcSv(svCoeff,sigmaAbs)
#    dpdz = funcDpDz(U_f,sigmaAbs)
#    deltaP = np.cumsum(dpdz*Height/N)
#    DfsDh = funcDpDz(U_f,sigmaAbs)
#    DGBouyantDh=solids*(1-epsilonProfile)*(densityProfile/(solids*(1-epsilonProfile))-rouf)*g/1000 
#    DfwDh = (2*lamda*miu*sigmaAbs/radius)
    
#    sigmaAbs2 = abs(sol2.y[0])
#    sigmaAbs2 = sol2.y[0]
    sigmaAbs2 = sol2
    avgStress = np.sum(sigmaAbs2*Height/N)/Height
    densityProfile2=funcDensity(DensityCoeff, sigmaAbs2)
    densityAvg2 = np.sum(densityProfile2*Height/N)/Height
    epsilonProfile2=funcEps2(epsCoeff, sigmaAbs2)
    epsilonAvg2 = np.sum(epsilonProfile2*Height/N)/Height
    SvProfile2 = funcSv2(svCoeff,sigmaAbs2)
#    SvProfile4=funcSv4(svCoeff,epsilonProfile2,epsCoeff[0])
    dpdz2 = funcDpDz2(U_f,sigmaAbs2)
    if scaleUp:
        revDpDz=dpdz2[::-1]
        revdeltaP2 = np.cumsum(revDpDz*Height/N)
        deltaP2=revdeltaP2[::-1]
    else:
        deltaP2 = np.cumsum(dpdz2*Height/N)
    DfsDh2 = funcDpDz2(U_f,sigmaAbs2)
#    DGBouyantDh2=solids*(1-epsilonProfile2)*(densityProfile2/(solids*(1-epsilonProfile2))-rouf)*g/1000 
    DGBouyantDh2=solids*(1-epsilonProfile2)*(densityProfile2/(solids*(1-epsilonProfile2))-rouf)*g
    DfwDh2 = (2*lamda*miu*sigmaAbs2/radius)
    
    label = ["stress","density","epsilon","Sv","dpdz","deltaP","fs","GBouyant","fw"]
#    PlotData = dict(stress =sol.y[0], density = densityProfile,\
#                epsilon=epsilonProfile,Sv = SvProfile,dpdz = dpdz,deltaP = deltaP,\
#                fs=DfsDh,GBouyant=-DGBouyantDh,fw=-DfwDh)
#    if plot:
#        for i in range(9):
#            fig = plt.figure(figsize=(8, 6))
#            ax = fig.add_subplot(1, 1, 1)
#            ax.plot(sol.t, PlotData[label[i]], 'k--', label=label[i]) 
#            ax.set_xlabel('Height (m)')
#            ax.legend(loc='best')
            
    return dict(height=tspan,stress =sigmaAbs2,avgStress=avgStress, density = densityProfile2,avgDensity=densityAvg2,\
                    epsilon=epsilonProfile2,avgEps=epsilonAvg2,Sv=SvProfile2,dpdz = dpdz2,deltaP = deltaP2,\
                    fs=DfsDh2,GBouyant=DGBouyantDh2,fw=DfwDh2,label = label)



#%%
#fp.setmeta(colID=1, dp_conns=((1,4),(1,3),(2,4)))
# dp probe #2:
def getProbLocIdx(N,Height):
    tspan=np.linspace(Height,0,N)
    dp2_low = fp.ele_ports[0]/1000 #unit: m
    dp2_high = fp.ele_ports[2]/1000
    idx_dp2low = np.where(tspan>dp2_low)[-1][-1]
    idx_dp2high = np.where(tspan<dp2_high)[0][0]
    dp3_low = fp.ele_ports[1]/1000 #unit: m
    dp3_high = fp.ele_ports[3]/1000
    idx_dp3low = np.where(tspan>dp3_low)[-1][-1]
    idx_dp3high = np.where(tspan<dp3_high)[0][0]
    return dp2_low,idx_dp2low,dp2_high,idx_dp2high,dp3_low,idx_dp3low,dp3_high,idx_dp3high
def getModelDP(DP,N,Height):
    deltaP = DP
    dp2_low,idx_dp2low,dp2_high,idx_dp2high,dp3_low,idx_dp3low,dp3_high,idx_dp3high = getProbLocIdx(N,Height)
    Model_dp2 = (deltaP[idx_dp2low] - deltaP[idx_dp2high])/(dp2_high-dp2_low)
    Model_dp3 = (deltaP[idx_dp3low] - deltaP[idx_dp3high])/(dp3_high-dp3_low)
    Model_avgDP = np.average([Model_dp2,Model_dp3])
    Model_stdDP = np.std([Model_dp2,Model_dp3])
    return Model_dp2, Model_dp3, Model_avgDP,Model_stdDP
def getModelProbeLocStress(Stress,N,Height):
    dp2_low,idx_dp2low,dp2_high,idx_dp2high,dp3_low,idx_dp3low,dp3_high,idx_dp3high = getProbLocIdx(N,Height)
#    avgStress_dp2 = (Stress[idx_dp2low] + Stress[idx_dp2high])/(dp2_high-dp2_low)
#    avgStress_dp3 = (Stress[idx_dp3low] + Stress[idx_dp3high])/(dp3_high-dp3_low)
#    avgStress_dp2 = (Stress[idx_dp2low] + Stress[idx_dp2high])/2
#    avgStress_dp3 = (Stress[idx_dp3low] + Stress[idx_dp3high])/2
    avgStress_dp2 = np.mean(Stress[idx_dp2high:idx_dp2low+1])
    stdStress_dp2 = np.std(Stress[idx_dp2high:idx_dp2low+1])
    avgStress_dp3 = np.mean(Stress[idx_dp3high:idx_dp3low+1])
    stdStress_dp3 = np.std(Stress[idx_dp3high:idx_dp3low+1])
    avgStress_avgDP = np.average([avgStress_dp2,avgStress_dp3])
    stdStress_avgDP = np.std([avgStress_dp2,avgStress_dp3])
    return avgStress_dp2,stdStress_dp2,avgStress_dp3,stdStress_dp3,avgStress_avgDP,stdStress_avgDP

def compareModelExp(result,frictCoeff,Height,yinit,i_exp,rouCoeff=densityCoeff,EpsCoeff=EpsCoeff2,\
                    SvCoeff = SvCoeff2,rouf = 1000.0,viscosity=0.00089,plot = False):
    frictionCoeff = frictCoeff
#    i_exp = 5
    radius = fp.dia_col/2000 # unit: m
#    Height = BedHeight[i_exp] # unit m
    
    DP_result = []
    N_Uf = len(result[i_exp]['avg_sfvel'])
    Model_dp2 = np.zeros(N_Uf)
    Model_dp3 = np.zeros(N_Uf)
    Model_avgDP = np.zeros(N_Uf)
    Model_stdDP = np.zeros(N_Uf)
    avgStress_dp2= np.zeros(N_Uf)
    stdStress_dp2= np.zeros(N_Uf)
    avgStress_dp3= np.zeros(N_Uf)
    stdStress_dp3= np.zeros(N_Uf)
    avgStress_avgDP= np.zeros(N_Uf)
    stdStress_avgDP= np.zeros(N_Uf)
    for i in range(N_Uf):
        U_f = result[i_exp]['avg_sfvel'][i] # unit: mm/s
        DP_result.append(getBedProperties(frictionCoeff,U_f,Height,radius,yinit,\
                                          DensityCoeff=rouCoeff,epsCoeff=EpsCoeff,\
                                          svCoeff = SvCoeff,rouf = rouf,viscosity=viscosity))
        DP = DP_result[i]['deltaP']
        Stress = DP_result[i]['stress']
        avgStress_dp2[i],stdStress_dp2[i],avgStress_dp3[i],stdStress_dp3[i],\
        avgStress_avgDP[i],stdStress_avgDP[i] = getModelProbeLocStress(Stress,1000,Height)
        Model_dp2[i], Model_dp3[i],Model_avgDP[i],Model_stdDP[i] = getModelDP(DP,1000,Height)
#    U_f = result[-1]['avg_sfvel'][-1]
#    DP_result = getBedProperties(frictionCoeff,U_f,Height,radius)
#    DP = DP_result['deltaP']
#    Model_dp2, Model_dp3 = getModelDP(DP)
    exp_dp2 = result[i_exp]['pg_1']
    exp_std_dp2 = result[i_exp]['std_pg_1']
    exp_dp3 = result[i_exp]['pg_2']
    exp_std_dp3 = result[i_exp]['std_pg_2']
    exp_dp = np.concatenate((exp_dp2,exp_std_dp2,exp_dp3,exp_std_dp3))
#    model_dp = np.concatenate((Model_dp2,Model_dp3))
#    RMSE = np.sqrt(sum(np.power((exp_dp - model_dp),2))/len(exp_dp))
#    if plot:
##        fig = plt.figure(figsize=(8, 6))
##        ax = fig.add_subplot(1, 1, 1)
#        plt.errorbar(result[i_exp]['avg_sfvel'],result[i_exp]['avg_pg'],xerr=result[i_exp]['stdev_sfvel'],yerr=result[i_exp]['stdev_pg'],\
#                 color='C1',markersize = 4, fmt='-s',label='average', capsize= 6, capthick=2)
#        plt.errorbar(result[i_exp]['avg_sfvel'],Model_avgDP,xerr=result[i_exp]['stdev_sfvel'],yerr=Model_stdDP,\
#                 color='C1',markersize = 4, fmt='-s',label='average', capsize= 5, capthick=1)
#        ax.plot(result[i_exp]['avg_sfvel'],exp_dp2,label='expDP2')
#        ax.plot(result[i_exp]['avg_sfvel'],exp_dp3,label='expDP3')
#        ax.plot(result[i_exp]['avg_sfvel'],Model_dp2,label='ModelDP2')
#        ax.plot(result[i_exp]['avg_sfvel'],Model_dp3,label='ModelDP3')
#        ax.set_xlabel('Surperficial Velocity (mm/s)')
#        ax.legend(loc='best')
#    RMSE = np.sqrt(sum(np.power((exp_dp - model_dp),2))/len(exp_dp))
#    return exp_dp, model_dp, Model_avgDP,Model_stdDP, RMSE,DP_result
    return dict(avgStressP1 = avgStress_dp2,stdStressP1 = stdStress_dp2,\
                avgStressP2=avgStress_dp3,stdStressP2 = stdStress_dp3,avgStressDP=avgStress_avgDP,\
                stdStressDP=stdStress_avgDP,Model_dpP1=Model_dp2,Model_dpP2=Model_dp3,\
                Model_avgDP=Model_avgDP,Model_stdDP=Model_stdDP,BPData_U=DP_result)
#%%
def CPBModelExpCompare(Pc,result,BedHeight,istart,rouCoefficient,EpsCoefficient,SvCoefficient,\
                       rouf = 1000.0,viscosity=0.00089,frictCoeff=0.10): #plot=True,save=False
    BP_ModelExp = []
    Model_dpP1 = []
    Model_dpP2 = []
    Model_avgDP = []
    Model_stdDP = []
    avgStressP1=[]
    avgStressP2=[]
    avgStressDP=[]
    stdStressDP=[]
    stdStressP1=[]
    stdStressP2=[]
    N_exp=len(Pc)
    for i_exp in range(istart,N_exp):
        yinit = [Pc[i_exp]]
        frictCoeff=0.10
        Height = BedHeight[i_exp]
        BP_ModelExp_U = compareModelExp(result,frictCoeff,Height,yinit,i_exp,rouCoeff=rouCoefficient,\
                                        EpsCoeff=EpsCoefficient,SvCoeff = SvCoefficient,rouf = rouf,viscosity=viscosity)
        BP_ModelExp.append(BP_ModelExp_U)
        Model_dpP1.append(BP_ModelExp_U['Model_dpP1'])
        Model_dpP2.append(BP_ModelExp_U['Model_dpP2'])
        Model_avgDP.append(BP_ModelExp_U['Model_avgDP'])
        Model_stdDP.append(BP_ModelExp_U['Model_stdDP'])
        avgStressP1.append(BP_ModelExp_U['avgStressP1'])
        stdStressP1.append(BP_ModelExp_U['stdStressP1'])
        avgStressP2.append(BP_ModelExp_U['avgStressP2'])
        stdStressP2.append(BP_ModelExp_U['stdStressP2'])
        avgStressDP.append(BP_ModelExp_U['avgStressDP'])
        stdStressDP.append(BP_ModelExp_U['stdStressDP'])
    
    return dict(i_init=istart,totalNo=N_exp,Model_dpP1=Model_dpP1,Model_dpP2=Model_dpP2,\
                Model_avgDP=Model_avgDP,Model_stdDP=Model_stdDP,\
                avgStressP1=avgStressP1,stdStressP1=stdStressP1,\
                avgStressP2=avgStressP2,stdStressP2=stdStressP2,\
                avgStressDP=avgStressDP,stdStressDP=stdStressDP,BP_ModelExp=BP_ModelExp) 

def plotCPBModelExpDPCompare(CPB_Data,Pc, result,fn_str='',useMarker=True,markersize = 6, fmt='--s',\
                             capsize= 5,alpha=1, capthick=1,linewidth = 3,\
                             plot=True,save=False):
    # Plot data from CPBModelExpCompare return;
    # compare CPB model with expeirment
    istart=CPB_Data['i_init']
    N_exp=CPB_Data['totalNo']

    if plot:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1)
        for i_exp in range(istart,N_exp):
            label = '$\mathregular{P_c}$ = %0.2f kPa' % (Pc[i_exp])
            plt.errorbar(result[i_exp]['avg_sfvel'],result[i_exp]['avg_pg'],\
                         xerr=result[i_exp]['stdev_sfvel'],yerr=result[i_exp]['stdev_pg'],\
                         color='C'+str(i_exp),marker=markerlist[i_exp],markersize = 12, fmt='s',label=label, capsize= 6, capthick=2)
        for i_exp in range(istart,N_exp):
            Model2_avgDP = CPB_Data['Model_avgDP'][i_exp-istart]
            Model2_stdDP = CPB_Data['Model_stdDP'][i_exp-istart]
            if useMarker:
                plt.errorbar(result[i_exp]['avg_sfvel'],Model2_avgDP,\
                             yerr=Model2_stdDP,\
                             color='C'+str(i_exp),marker=markerlist[i_exp],\
                             markersize = markersize, fmt=fmt,capsize= capsize, \
                             alpha=alpha,capthick=capthick,linewidth = linewidth)
            else:
                plt.errorbar(result[i_exp]['avg_sfvel'],Model2_avgDP,\
                             yerr=Model2_stdDP,\
                             color='C'+str(i_exp),\
                             fmt=fmt,capsize= capsize, \
                             alpha=alpha,capthick=capthick,linewidth = linewidth)
                    
        plt.legend(loc = 'upper left',prop={'size': 16, 'weight':'bold'},frameon=False,)
        ax.set_xlabel('Superficial velocity (mm/s)')
        ax.set_ylabel('dP/dz (kPa/m)') 
        # ax.set_xlim(0, 5)
        # ax.set_ylim(0, 80)
        ax.tick_params(axis = 'y', which = 'major', direction = 'out', length = 6.0, width = 2.0)
        ax.tick_params(axis = 'x', which = 'major', direction = 'out', length = 6.0, width = 2.0)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.8)
        fig.tight_layout()
        plt.show()
    if save:
        fig.savefig('./asMillWM/asMillWM_BedModelPredictionValidation_Parameters_'+fn_str+'.jpg',dpi=300,bbox_inches='tight')
    

#%%
BedHeight = [result_fitting[i]['bedHeight'] for i in range(len(result_fitting))]
radius = fp.dia_col/2000

def get_avgPs(Pc,result,BedHeight,packingDensity,epsilon,frictCoeff = 0.10):
    DP_U0 = []
    N_exp = len(Pc)
    avgPs=np.zeros(len(Pc))
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    for i_exp in range(1,N_exp):
        yinit = [Pc[i_exp]]
        # frictCoeff=0.10
        Height = BedHeight[i_exp]
        U_f = 0
        DP_U0.append(getBedProperties(frictCoeff,U_f,Height,radius,yinit,\
                                              DensityCoeff=densityCoeff,epsCoeff=EpsCoeff2,\
                                              svCoeff = SvCoeff2))
        avgPs[i_exp]=DP_U0[i_exp-1]['avgStress']
        labelstr = '%0.2f kPa avgStress=%0.2f' % (Pc[i_exp],DP_U0[i_exp-1]['avgStress'])
        ax.plot(DP_U0[i_exp-1]['height'],DP_U0[i_exp-1]['stress'],label=labelstr)
    ax.legend(loc='best')
    

    PsData=avgPs#pressure kPa
    rouData=packingDensity #kg/m3
    x0_rou =  [1,1,1]
    densityCoeffPs = fitDensity(x0_rou,PsData,rouData)
    eps_x0 = [1,1]
    epsData=epsilon #kg/m3
    EpsCoeffPs = fitEpsilon(eps_x0,PsData,epsData)
    x0_Sv = np.ones(2)*10
    SvCoeffPs= fitSv(x0_Sv,PsData,result,EpsCoeffPs)
    modifiedKCValidation(EpsCoeffPs,SvCoeffPs,PsData,result)
    
    return DP_U0,avgPs,densityCoeffPs,EpsCoeffPs,SvCoeffPs

DP_U0,avgPs,densityCoeffPs,EpsCoeffPs,SvCoeffPs = get_avgPs(Pc_fitting,result_fitting,BedHeight,packingDensity,epsilon_fitting)
#%%
istart=0
CPB_Pc=CPBModelExpCompare(Pc_fitting,result_fitting,BedHeight,istart,densityCoeff,EpsCoeff2,SvCoeff2)

plotCPBModelExpDPCompare(CPB_Pc,Pc_fitting,result_fitting,fn_str='PcCorelation')
#%%
def fitSvFromCPBModel(CPB_data,result,plot=True,save=False):
#    Model_DP_pb = CPB_data['BP_ModelExp']
    avgStressDp2=CPB_data['avgStressP1']
    avgStressDp3=CPB_data['avgStressP2']
#    avgStressDp=CPB_data['avgStressDP']
#    stdStressDp=CPB_data['stdStressDP']
#    Model2_avgDP=CPB_data['Model_avgDP']
#    Model2_stdDP=CPB_data['Model_stdDP']
    N_Uf_total = 0
    istart=CPB_data['i_init']
    N_exp=CPB_data['totalNo']
    for i_exp in range(istart,N_exp):
        Udata=result[i_exp]['avg_sfvel'][1:] # Highest Pc,first zero velocity
        N_Uf_total = N_Uf_total+ len(Udata)
    ESSAModelData = []  
    Ps_Uf_AllData = []   
    #fig = plt.figure(figsize=(8, 6))
    #ax = fig.add_subplot(1, 1, 1)
    for i_exp in range(istart,N_exp):
        Udata=result[i_exp]['avg_sfvel'][1:] # Highest Pc,first zero velocity
        N_Uf =  len(Udata)    
        specificSurfaceArea1=np.zeros(N_Uf)
        specificSurfaceArea2=np.zeros(N_Uf)
        Ps_Uf_Data1 = np.zeros(N_Uf)
        Ps_Uf_Data2 = np.zeros(N_Uf)
        for i in range(1,N_Uf+1):
            Ps_Uf_Data1[i-1]=avgStressDp2[i_exp-1][i]
            Ps_Uf_Data2[i-1]=avgStressDp3[i_exp-1][i]
            eps_kc1=funcEps2(EpsCoeffPs,Ps_Uf_Data1[i-1])
            eps_kc2=funcEps2(EpsCoeffPs,Ps_Uf_Data2[i-1])
            Udata=result[i_exp]['avg_sfvel'][i] #mm/s  # here i+1 to skip first experiment
            ydata1=result[i_exp]['pg_1'][i]#kPa/m
            ydata2=result[i_exp]['pg_2'][i]#kPa/m
            x_Sv = np.ones(1)*0.01
            res_lsq1=least_squares(funcMini2,x_Sv, loss='soft_l1', f_scale=0.1, args=(Udata,ydata1,eps_kc1))
            specificSurfaceArea1[i-1]=res_lsq1.x[0]
            print(specificSurfaceArea1)
            x_Sv = np.ones(1)*0.01
            res_lsq2=least_squares(funcMini2,x_Sv, loss='soft_l1', f_scale=0.1, args=(Udata,ydata2,eps_kc2))
            specificSurfaceArea2[i-1]=res_lsq2.x[0]
            
        ySvData1=specificSurfaceArea1
        ySvData2=specificSurfaceArea2
        ESSAModelData = np.concatenate((ESSAModelData,ySvData1,ySvData2))
        Ps_Uf_AllData = np.concatenate((Ps_Uf_AllData,Ps_Uf_Data1,Ps_Uf_Data2))

    PcPlot=np.linspace(0,40,1000)
    x0_Sv = np.ones(2)*0.001
    #
    res_lsqSvPs2=least_squares(funcSvMini2,x0_Sv, loss='soft_l1', f_scale=0.1, args=(Ps_Uf_AllData,ESSAModelData))
    SvCoeffPs_Modeldp23 = res_lsqSvPs2.x
    
    if plot:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1)
        
        plt.plot(Ps_Uf_AllData,ESSAModelData,'C3s',markersize = 12,label='Experiment dp2')
#        plt.plot(PcPlot, funcSv2(SvCoeffPs_Modeldp23+[1.5,0.0090], PcPlot), 'C3-',label='Model',linewidth = 3)
        plt.plot(PcPlot, funcSv2(SvCoeffPs_Modeldp23+[13,0.009], PcPlot), 'C3-',label='Model',linewidth = 3)
        
        plt.legend(loc = 'upper right',prop={'size': 18, 'weight':'bold'},frameon=False,)
        ax.set_xlabel('$\mathregular{P_c}$ (kPa)')
        ax.set_ylabel('Effective specific surface area ($\mathregular{cm^2 cm^{-3}}$)')
        ax.tick_params(axis = 'y', which = 'major', direction = 'out', length = 6.0, width = 2.0)
        ax.tick_params(axis = 'x', which = 'major', direction = 'out', length = 6.0, width = 2.0)      
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.8)
        fig.tight_layout()
        plt.show()
    if save:
        fig.savefig('./asMillWM/asMillWM_SvEpsModel2_correct.jpg',dpi=300,bbox_inches='tight')
    return SvCoeffPs_Modeldp23

SvCoeffPs_Modeldp23 = fitSvFromCPBModel(CPB_Pc,result_fitting)

#%%
SvCoeffPs_Modeldp23_fit =SvCoeffPs_Modeldp23 #+[15,0.009]#+[1.5,0.0090]# [7.547,0.00985]
CPB_Ps=CPBModelExpCompare(Pc_fitting,result_fitting,BedHeight,istart,densityCoeffPs,EpsCoeffPs,SvCoeffPs_Modeldp23_fit,\
                          viscosity=0.00089,frictCoeff=0.1)
plotCPBModelExpDPCompare(CPB_Ps,Pc_fitting,result_fitting,fn_str='PsCorelation',\
                         fmt=':',capsize= 9, alpha=0.7,capthick=3,linewidth = 3)
#%% Plot result based on Ps correlated model parameters
#plotCPBModelExpDPCompare(CPB_Ps,fn_str='PsCorelation',useMarker=False,\
#                         fmt=':',capsize= 9, alpha=0.7,capthick=3,linewidth = 3)
#%%
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)

#for i in range(N_exp):
#    label = 'Pc = ' + str(result[i]['Pc']) + ' kPa'
#    label = 'Pc = %0.2f kPa' % (result[i]['Pc'])
#    plt.plot(result[i]['avg_sfvel'],result[i]['pg_2'],label=label) #pg_2
i=1
label = 'Pc = ' + str(Pc_fitting[i]) + ' kPa'
#plt.plot(result[i]['avg_sfvel'],result[i]['PistonHeight'],'s',label=label)
plt.plot(result[i]['avg_sfvel'],result[i]['BedVolumeU'],'s',label=label)
plt.legend(loc='upper left')

#%%
#%% U=0
def plot_solidP_Ps_fitting(CPB_Ps,Pc):
    BPmodelExp=CPB_Ps['BP_ModelExp']
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    for i_exp in range(len(Pc)):
        label = '$\mathregular{P_c}$ = %0.2f kPa' % (Pc[i_exp])
        ax.plot(BPmodelExp[i_exp]['BPData_U'][0]['height'],BPmodelExp[i_exp]['BPData_U'][0]['stress'],\
                color='C'+str(i_exp),marker=markerlist[i_exp],markevery=50,markersize = 7, label=label) 
    ax.set_xlabel('z (m)')
    ax.set_ylabel('$\mathregular{P_s}$ (kPa)')
    #plt.legend(loc = 'upper right',prop={'size': 14, 'weight':'bold'},frameon=False,)
    # ax.set_xlim(0, 0.32)
    # ax.set_ylim(2.5, 25)
    ax.tick_params(axis = 'y', which = 'major', direction = 'out', length = 6.0, width = 2.0)
    ax.tick_params(axis = 'x', which = 'major', direction = 'out', length = 6.0, width = 2.0)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.8)
    fig.tight_layout()
    plt.show()
    #fig.savefig('./asMillWM/asMillWM_BedModel_solidP_ParaFit.jpg',dpi=300,bbox_inches='tight')

    # BPmodelExp=CPB_Ps['BP_ModelExp']
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    for i_exp in range(len(Pc)):
        label = '$\mathregular{P_c}$ = %0.2f kPa' % (Pc[i_exp])
        ax.plot(BPmodelExp[i_exp]['BPData_U'][0]['height'],BPmodelExp[i_exp]['BPData_U'][0]['epsilon'],\
                color='C'+str(i_exp),marker=markerlist[i_exp],markevery=50,markersize = 7, label=label) 
    ax.set_xlabel('z (m)')
    ax.set_ylabel('Effective porosity (-)')
    #plt.legend(bbox_to_anchor=(1.7,1), borderaxespad=0,loc = 'upper right',prop={'size': 14, 'weight':'bold'},frameon=False,)
    # ax.set_xlim(0, 0.32)
    # ax.set_ylim(0.14, 0.32)
    ax.tick_params(axis = 'y', which = 'major', direction = 'out', length = 6.0, width = 2.0)
    ax.tick_params(axis = 'x', which = 'major', direction = 'out', length = 6.0, width = 2.0)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.8)
    fig.tight_layout()
    plt.show()
    #fig.savefig('./asMillWM/asMillWM_BedModel_epsilon_ParaFit.jpg',dpi=300,bbox_inches='tight')

    # BPmodelExp=CPB_Ps['BP_ModelExp']    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    for i_exp in range(len(Pc)):
        label = '$\mathregular{P_c}$ = %0.2f kPa' % (Pc[i_exp])
        ax.plot(BPmodelExp[i_exp]['BPData_U'][0]['height'],BPmodelExp[i_exp]['BPData_U'][0]['density'],\
                color='C'+str(i_exp),marker=markerlist[i_exp],markevery=50,markersize = 7, label=label) 
    ax.set_xlabel('z (m)')
    ax.set_ylabel('Bulk density ($\mathregular{kg/m^3}$)')
    #plt.legend(bbox_to_anchor=(1.7,1), borderaxespad=0,loc = 'upper right',prop={'size': 14, 'weight':'bold'},frameon=False,)
    # ax.set_xlim(0, 0.32)
    # ax.set_ylim(120, 165)
    ax.tick_params(axis = 'y', which = 'major', direction = 'out', length = 6.0, width = 2.0)
    ax.tick_params(axis = 'x', which = 'major', direction = 'out', length = 6.0, width = 2.0)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.8)
    fig.tight_layout()
    plt.show()
    #fig.savefig('./asMillWM/asMillWM_BedModel_density_ParaFit.jpg',dpi=300,bbox_inches='tight')

plot_solidP_Ps_fitting(CPB_Ps, Pc_fitting)  

    #%% Pc = 24kPa, different U
def plot_solidP_Ps_velocity(i_exp, CPB_Ps,result):
    N_U = len(CPB_Ps['BP_ModelExp'][i_exp]['BPData_U'])
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    for i_U in range(0,N_U):
        label = 'U = %0.2f mm/s' % (result[i_exp]['avg_sfvel'][i_U])
        ax.plot(CPB_Ps['BP_ModelExp'][i_exp]['BPData_U'][i_U]['height'],\
                CPB_Ps['BP_ModelExp'][i_exp]['BPData_U'][i_U]['stress'],\
                color='C'+str(i_U),marker=markerlist[i_U],markevery=50,markersize = 7, label=label) 
    ax.set_xlabel('z (m)')
    ax.set_ylabel('$\mathregular{P_s}$ (kPa)')
    #plt.legend(loc = 'best',prop={'size': 14, 'weight':'bold'},frameon=False)
    # ax.set_xlim(0, 0.25)
    # ax.set_ylim(5, 25)
    ax.tick_params(axis = 'y', which = 'major', direction = 'out', length = 6.0, width = 2.0)
    ax.tick_params(axis = 'x', which = 'major', direction = 'out', length = 6.0, width = 2.0)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.8)
    fig.tight_layout()
    plt.show()
    #fig.savefig('./asMillWM/asMillWM_BedModel_Ps-Uf_Pc24kPa_ParaFit.jpg',dpi=300,bbox_inches='tight')
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    for i_U in range(0,N_U):
        label = 'U = %0.2f mm/s' % (result[i_exp]['avg_sfvel'][i_U])
        ax.plot(CPB_Ps['BP_ModelExp'][i_exp]['BPData_U'][i_U]['height'],\
                CPB_Ps['BP_ModelExp'][i_exp]['BPData_U'][i_U]['epsilon'],\
                color='C'+str(i_U),marker=markerlist[i_U],markevery=50,markersize = 7, label=label) 
    ax.set_xlabel('z (m)')
    ax.set_ylabel('Effective porosity (-)')
    #plt.legend(loc = 'best',prop={'size': 14, 'weight':'bold'},frameon=False,)
    # ax.set_xlim(0, 0.25)
    # ax.set_ylim(0.14, 0.28)
    ax.tick_params(axis = 'y', which = 'major', direction = 'out', length = 6.0, width = 2.0)
    ax.tick_params(axis = 'x', which = 'major', direction = 'out', length = 6.0, width = 2.0)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.8)
    fig.tight_layout()
    plt.show()
    #fig.savefig('./asMillWM/asMillWM_BedModel_Porosity-Uf_Pc24kPa_ParaFit.jpg',dpi=300,bbox_inches='tight')
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    for i_U in range(0,N_U):
        label = 'U = %0.2f mm/s' % (result[i_exp]['avg_sfvel'][i_U])
        ax.plot(CPB_Ps['BP_ModelExp'][i_exp]['BPData_U'][i_U]['height'],\
                CPB_Ps['BP_ModelExp'][i_exp]['BPData_U'][i_U]['density'],\
                color='C'+str(i_U),marker=markerlist[i_U],markevery=50,markersize = 7, label=label) 
    ax.set_xlabel('z (m)')
    ax.set_ylabel('Bulk density ($\mathregular{kg/m^3}$)')
    #plt.legend(loc = 'best',prop={'size': 14, 'weight':'bold'},frameon=False,)
    #plt.legend(bbox_to_anchor=(1.7,1), borderaxespad=0,loc = 'upper right',prop={'size': 14, 'weight':'bold'},frameon=False,)
    # ax.set_xlim(0, 0.25)
    # ax.set_ylim(125, 165)
    ax.tick_params(axis = 'y', which = 'major', direction = 'out', length = 6.0, width = 2.0)
    ax.tick_params(axis = 'x', which = 'major', direction = 'out', length = 6.0, width = 2.0)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.8)
    fig.tight_layout()
    plt.show()
    #fig.savefig('./asMillWM/asMillWM_BedModel_BulkDensity-Uf_Pc24kPa_ParaFit.jpg',dpi=300,bbox_inches='tight')


plot_solidP_Ps_velocity(2, CPB_Ps, result_fitting)
#%% Ps corelation result
CPB_data=CPB_Ps
#Model_DP_pb = CPB_data['BP_ModelExp']
avgStressDp2=CPB_data['avgStressP1']
stdStressDp2=CPB_data['stdStressP1']
avgStressDp3=CPB_data['avgStressP2']
stdStressDp3=CPB_data['stdStressP2']
avgStressDp=CPB_data['avgStressDP']
stdStressDp=CPB_data['stdStressDP']
Model2_avgDP=CPB_data['Model_avgDP']
Model2_stdDP=CPB_data['Model_stdDP']
Model_DpP2=CPB_data['Model_dpP1']
Model_DpP3=CPB_data['Model_dpP2']
##%%
#fig = plt.figure(figsize=(8, 6))
#ax = fig.add_subplot(1, 1, 1)
#for i_exp in range(1,N_exp):
#    plt.plot(result[i_exp]['avg_sfvel'],avgStressDp[i_exp-1],label='dp2')

#%%

def plotDP1DP2Compare(i_exp,CPBdata,result,setylimit=False,ylimit=[0,40,0,100],strU='',save=False):
    CPB_data=CPBdata
    #Model_DP_pb = CPB_data['BP_ModelExp']
    avgStressDp2=CPB_data['avgStressP1']
    stdStressDp2=CPB_data['stdStressP1']
    avgStressDp3=CPB_data['avgStressP2']
    stdStressDp3=CPB_data['stdStressP2']
    avgStressDp=CPB_data['avgStressDP']
    stdStressDp=CPB_data['stdStressDP']
    Model2_avgDP=CPB_data['Model_avgDP']
    Model2_stdDP=CPB_data['Model_stdDP']
    Model_DpP2=CPB_data['Model_dpP1']
    Model_DpP3=CPB_data['Model_dpP2']
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    #for i_exp in range(1,N_exp):
    #    plt.plot(result[i_exp]['avg_sfvel'],avgStressDp2[i_exp-1],label='dp2')
    #    plt.plot(result[i_exp]['avg_sfvel'],avgStressDp3[i_exp-1],label='dp3')
    color = 'tab:red'
    ax.errorbar(result[i_exp]['avg_sfvel'],avgStressDp2[i_exp],\
                     yerr=stdStressDp2[i_exp],\
                     fmt='',capsize= 6, alpha=1,capthick=2,linewidth = 2,color=color,ls='none',label='Avg $\mathregular{P_s}$: DP-1')
    ax.errorbar(result[i_exp]['avg_sfvel'],avgStressDp3[i_exp],\
                     yerr=stdStressDp3[i_exp],\
                     fmt='',capsize= 6, alpha=0.7,capthick=2,linewidth = 2,color=color,ls='none',label='Avg $\mathregular{P_s}$: DP-2')
    
    lns1=ax.plot(result[i_exp]['avg_sfvel'],avgStressDp2[i_exp],'--',color=color,label='Avg $\mathregular{P_s}$: DP-1')
    lns2=ax.plot(result[i_exp]['avg_sfvel'],avgStressDp3[i_exp],':',color=color,label='Avg $\mathregular{P_s}$: DP-2')
    ax.set_xlabel('Superficial velocity (mm/s)')
    ax.set_ylabel('$\mathregular{P_s}$ (kPa)',color=color)
    ax.tick_params(axis = 'y', which = 'major', direction = 'out', length = 6.0, width = 2.0,color=color,labelcolor=color)
    ax.tick_params(axis = 'x', which = 'major', direction = 'out', length = 6.0, width = 2.0)
    
    
    
    ax2 = ax.twinx()
    color = 'tab:blue'
    lns3=ax2.plot(result[i_exp]['avg_sfvel'],result[i_exp]['pg_1'],'o',markersize =9,color=color,label='Experiment: DP-1') 
    lns4=ax2.plot(result[i_exp]['avg_sfvel'],result[i_exp]['pg_2'],'v',markersize =9,color=color,label='Experiment: DP-2')
    lns5=ax2.plot(result[i_exp]['avg_sfvel'],Model_DpP2[i_exp],'--',linewidth = 3,color=color,label='Simulation: DP-1')
    lns6=ax2.plot(result[i_exp]['avg_sfvel'],Model_DpP3[i_exp],':',linewidth = 3,color=color,label='Simulation: DP-2')
    #plt.errorbar(result[N_exp-1]['avg_sfvel'],result[N_exp-1]['pg_1'],\
    #            xerr=result[N_exp-1]['stdev_sfvel'],yerr=result[N_exp-1]['std_pg_1'],label='dp1') 
    #plt.errorbar(result[N_exp-1]['avg_sfvel'],result[N_exp-1]['pg_2'],\
    #            xerr=result[N_exp-1]['stdev_sfvel'],yerr=result[N_exp-1]['std_pg_2'],label='dp2') 
    ax2.set_xlabel('Superficial velocity (mm/s)')
    ax2.set_ylabel('dP/dz (kPa/m)',color=color) 
    ax2.tick_params(axis = 'y', which = 'major', direction = 'out', length = 6.0, width = 2.0,color=color,labelcolor=color)
    ax2.tick_params(axis = 'x', which = 'major', direction = 'out', length = 6.0, width = 2.0)
    if setylimit:
        ax.set_xlim(0, 5)
        ax.set_ylim(ylimit[0], ylimit[1])
        ax2.set_ylim(ylimit[2], ylimit[3])
    #for axis in ['top','bottom','left','right']:
    #    ax2.spines[axis].set_linewidth(1.8)
    lns=lns3+lns4+lns5+lns6+lns1+lns2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='upper center',prop={'size': 14, 'weight':'bold'},frameon=False) 
    ax.spines['left'].set_color('tab:red')
    ax.spines['right'].set_color('tab:blue')
    #for t in ax.yaxis.get_ticklines(): t.set_color('green')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.8)
    #ax2.legend(loc='best')
    fig.tight_layout()
    plt.show()
    if save:
        fig.savefig('./asMillWM/asMillWM_BedModel_DP1DP2_U_Pc'+strU+'kPa_ParaFit.jpg',dpi=300,bbox_inches='tight')

#%%    Figure - 10
i_DP1DP2=0
# plotDP1DP2Compare(i_DP1DP2,CPB_Ps,result_fitting,setylimit=True,ylimit=[3,30,0,90],strU='11.96')
plotDP1DP2Compare(i_DP1DP2,CPB_Ps,result_fitting)

i_DP1DP2_2=2
# plotDP1DP2Compare(i_DP1DP2_2,CPB_Ps,result_fitting,setylimit=True,ylimit=[3,30,0,90],strU='24.42')
plotDP1DP2Compare(i_DP1DP2_2,CPB_Ps,result_fitting)

#%% Scale-up
def plotScaleUp(scaleUpData,variable,v_name='',v_unit='',setylimit=False,ylimit=[],save=False):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    color2='tab:blue'
    color1='tab:red'
    ax2 = ax.twinx()
    
    for i in range(len(variable)):
        deltaPPlot_tspan=scaleUpData[i]['height'][::-1]
        deltaP_head = scaleUpData[i]['deltaP']+rouf*g*deltaPPlot_tspan/1000
        ax2.plot(scaleUpData[i]['height'],scaleUpData[i]['stress'],ls='none',color='C7',\
                marker=markerlist[i],markevery=50,markersize = 9,label=v_name+'='+str(variable[i])+v_unit)
        ax2.plot(scaleUpData[i]['height'],scaleUpData[i]['stress'],'--',linewidth=3,color=color2)
        
        ax.plot(scaleUpData[i]['height'],scaleUpData[i]['deltaP'],ls='none',color='C7',\
                marker=markerlist[i],markevery=50,markersize = 9)
        ax.plot(scaleUpData[i]['height'],scaleUpData[i]['deltaP'],'--',linewidth=3,color=color1)
    #    ax.plot(scaleUp_LDratio[i]['height'],deltaP_head,ls='none',color='C7',\
    #            marker=markerlist[i],markevery=50,markersize = 9)
    #    ax.plot(scaleUp_LDratio[i]['height'],deltaP_head,'--',linewidth=3,color=color1)
    #    ax.plot(scaleUp_LDratio[i]['height'],scaleUp_LDratio[i]['dpdz'],label=str(SU_radius[i]))  
    #    ax.plot(scaleUp_LDratio[i]['height'],deltaP_head,label=str(SU_radius[i]))
    ax.set_xlabel('z (m)')
    ax.set_ylabel('Cumulative dp/dz (kPa)',color=color1)
    ax.tick_params(axis = 'y', which = 'major', direction = 'out', length = 6.0, width = 2.0,color=color1,labelcolor=color1)
    ax.tick_params(axis = 'x', which = 'major', direction = 'out', length = 6.0, width = 2.0)
    ax2.set_ylabel('$\mathregular{P_s}$ (kPa)',color=color2)
    ax2.tick_params(axis = 'y', which = 'major', direction = 'out', length = 6.0, width = 2.0,color=color2,labelcolor=color2)
    ax2.tick_params(axis = 'x', which = 'major', direction = 'out', length = 6.0, width = 2.0)
    ax2.legend(bbox_to_anchor=(0.7,1),loc='upper center',prop={'size': 14, 'weight':'bold'},frameon=False) 
    ax.spines['left'].set_color('tab:red')
    ax.spines['right'].set_color('tab:blue')
    ax.set_xlim(0, 3)
    if setylimit:
        ax.set_ylim(ylimit[0], ylimit[1])
        ax2.set_ylim(ylimit[2], ylimit[3])
    for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.8)
    fig.tight_layout()
    plt.show() 
    if save:
        fig.savefig('./asMillWM/asMillWM_CPBM_ScaleUp_'+v_name+'.jpg',dpi=300,bbox_inches='tight')  
#%%
# L/D ratio
SU_frictionCoeff=0.1
SU_U_f = 1 # in mm/s
SU_Height = 3
#SU_radius = 1
SU_yinit=0
viscositySU=0.00089
rouf = 1000

# radiusHeight L/D ratio
SU_radius = [0.5,1,8]
scaleUp_LDratio=[]


for i in range(len(SU_radius)):
    scaleUp_LDratio.append(getBedProperties(SU_frictionCoeff,SU_U_f,SU_Height,SU_radius[i],SU_yinit,\
                         DensityCoeff=densityCoeffPs,epsCoeff=EpsCoeffPs,scaleUp=True,\
                         svCoeff = SvCoeffPs_Modeldp23_fit,rouf = 1000,viscosity=viscositySU))
    
plotScaleUp(scaleUp_LDratio,SU_radius,v_name='Radius',v_unit='m',setylimit=True,ylimit=[0,18,0,18])

#%%   
##SU_frictionCoeff=0.1
SU_U_f = 1 # in mm/s
SU_radius = 1
viscositySU=0.00089
rouf = 1000

# radiusHeight L/D ratio
SU_frictionCoeff = [0.1,0.5,1]
scaleUp_frict=[]

for i in range(len(SU_frictionCoeff)):
    scaleUp_frict.append(getBedProperties(SU_frictionCoeff[i],SU_U_f,SU_Height,SU_radius,SU_yinit,\
                         DensityCoeff=densityCoeffPs,epsCoeff=EpsCoeffPs,scaleUp=True,\
                         svCoeff = SvCoeffPs_Modeldp23_fit,rouf = 1000,viscosity=viscositySU))

plotScaleUp(scaleUp_frict,SU_frictionCoeff,v_name='Friction Coefficient',v_unit='',setylimit=True,ylimit=[0,15,0,15])
#%%   
SU_frictionCoeff=0.1
#SU_U_f = 1 # in mm/s
SU_radius = 1
viscositySU=0.00089
rouf = 1000

# radiusHeight L/D ratio
SU_U_f = [1.0,1.25,1.5] # mm/s
scaleUp_Uf=[]
for i in range(len(SU_U_f)):
    scaleUp_Uf.append(getBedProperties(SU_frictionCoeff,SU_U_f[i],SU_Height,SU_radius,SU_yinit,\
                         DensityCoeff=densityCoeffPs,epsCoeff=EpsCoeffPs,scaleUp=True,\
                         svCoeff = SvCoeffPs_Modeldp23_fit,rouf = 1000,viscosity=viscositySU))
        
plotScaleUp(scaleUp_Uf,SU_U_f,v_name='U',v_unit='mm/s',setylimit=True,ylimit=[0,150,0,150])
#%%   
SU_frictionCoeff=0.1
SU_U_f = 1 # in mm/s
SU_radius = 1
viscositySU=0.00089
rouf = 1000

# radiusHeight L/D ratio
SU_T=[25,50,75]
visc1=WaterViscosity(SU_T[0]).vd
visc2=WaterViscosity(SU_T[1]).vd
visc3=WaterViscosity(SU_T[2]).vd
SU_visc= [visc1,visc2,visc3] # mm/s

scaleUp_visc=[]
for i in range(len(SU_visc)):
    scaleUp_visc.append(getBedProperties(SU_frictionCoeff,SU_U_f,SU_Height,SU_radius,SU_yinit,\
                         DensityCoeff=densityCoeffPs,epsCoeff=EpsCoeffPs,scaleUp=True,\
                         svCoeff = SvCoeffPs_Modeldp23_fit,rouf = 1000,viscosity=SU_visc[i]))
        
plotScaleUp(scaleUp_visc,SU_T,v_name='Viscosity at T',v_unit='$\mathregular{^oC}$',setylimit=True,ylimit=[0,15,0,15])
#%%
#SU_frictionCoeff=0.5
#SU_U_f = 1 # in mm/s
#SU_Height = 2
#SU_radius = 1
#SU_yinit=0
#viscositySU=0.00089
#
#scaleUp=getBedProperties(SU_frictionCoeff,SU_U_f,SU_Height,SU_radius,SU_yinit,\
#                         DensityCoeff=densityCoeffPs,epsCoeff=EpsCoeffPs,scaleUp=True,\
#                         svCoeff = SvCoeffPs_Modeldp23_fit,rouf = 1000,viscosity=viscositySU)
#     
#prop=['stress','dpdz','deltaP','density','epsilon','fs','GBouyant','fw']
#for i in prop:
#    fig = plt.figure(figsize=(8, 6))
#    ax = fig.add_subplot(1, 1, 1)
#    ax.plot(scaleUp['height'],scaleUp[i])
#    ax.set_xlabel('z (m)')
#    ax.set_ylabel(i)
#    #plt.legend(loc = 'best',prop={'size': 14, 'weight':'bold'},frameon=False,)
#    #plt.legend(bbox_to_anchor=(1.7,1), borderaxespad=0,loc = 'upper right',prop={'size': 14, 'weight':'bold'},frameon=False,)
#    #ax.set_xlim(0, 0.25)
#    #ax.set_ylim(125, 160)
#    ax.tick_params(axis = 'y', which = 'major', direction = 'out', length = 6.0, width = 2.0)
#    ax.tick_params(axis = 'x', which = 'major', direction = 'out', length = 6.0, width = 2.0)
#    for axis in ['top','bottom','left','right']:
#        ax.spines[axis].set_linewidth(1.8)
#    fig.tight_layout()
#    plt.show()
#%%
N_write=len(Pc[1:])
BPKC_U0_avgEps=np.zeros(N_write)
BP_U0_avgEps=np.zeros(N_write)
BP_U0_avgDensity=np.zeros(N_write)
for iw in range(N_write):
    BPKC_U0_avgEps[iw]= CPB_Pc['BP_ModelExp'][iw]['BPData_U'][0]['avgEps']
    BP_U0_avgEps[iw]= CPB_Ps['BP_ModelExp'][iw]['BPData_U'][0]['avgEps']
    BP_U0_avgDensity[iw]= CPB_Ps['BP_ModelExp'][iw]['BPData_U'][0]['avgDensity']

#ExpSimEpsRouData=np.transpose(np.array([Pc[1:],BedHeight[1:],epsilon[1:],packingDensity[1:],BPKC_U0_avgEps,BP_U0_avgEps,BP_U0_avgDensity]))
np.savetxt('./asMillWM/PcHeightEpsRouSimkcepsSimepsSimrou.dat',np.transpose(np.array([Pc[1:],BedHeight[1:],epsilon[1:],packingDensity[1:],BPKC_U0_avgEps,BP_U0_avgEps,BP_U0_avgDensity]))) 

 
#%%
"""
#i_exp = N_exp -2 
i_exp = 0   
N_Uf = len(result[i_exp]['avg_sfvel'])
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)
for i_Uf in range(0,N_Uf):
#    label = 'Pc = %0.2f kPa' % (Pc[i_pc])
    label = 'U = %0.2f mm/s' % (result[i_exp]['avg_sfvel'][i_Uf])
    ax.plot(Model_DP[i_exp][i_Uf]['height'], Model_DP[i_exp][i_Uf]['stress'], label=label) 
ax.set_xlabel('Height (m)')
ax.set_ylabel('Stress (kPa)')
plt.legend(loc = 'lower right',prop={'size': 16, 'weight':'bold'},frameon=False,)
ax.set_xlim(0, 0.30)
#ax.set_ylim(20, 30)
ax.tick_params(axis = 'y', which = 'major', direction = 'out', length = 6.0, width = 2.0)
ax.tick_params(axis = 'x', which = 'major', direction = 'out', length = 6.0, width = 2.0)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.8)
fig.tight_layout()
plt.show()
#fig.savefig('./asMillWM/asMillWM_BedModel_solidP_5.5kPa.jpg',dpi=300,bbox_inches='tight') 
#%%
#i_exp = N_exp -2
i_exp = 1
i_Uf = 1    
N_Uf = len(result[i_exp]['avg_sfvel'])
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)
#fs=DfsDh2,GBouyant=-DGBouyantDh2,fw=-DfwDh2
ax.plot(Model_DP[i_exp][i_Uf]['height'], Model_DP[i_exp][i_Uf]['stress'],label='stress')
ax.legend(loc='best')
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)
ax.plot(Model_DP[i_exp][i_Uf]['height'], Model_DP[i_exp][i_Uf]['fs'],label='fdpdz')
ax.legend(loc='best')
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)  
ax.plot(Model_DP[i_exp][i_Uf]['height'], Model_DP[i_exp][i_Uf]['GBouyant'],label='fG')
ax.legend(loc='best')
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)
ax.plot(Model_DP[i_exp][i_Uf]['height'], Model_DP[i_exp][i_Uf]['fw'],label='fW')
ax.legend(loc='best')
#%%    
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)
for i_pc in range(1,N_exp):
    label = 'Pc = %0.2f kPa' % (Pc[i_pc])
    i_pc = i_pc -1
    tspan = Model_DP[i_pc][-1]['height']
    deltaPPlot_tspan = tspan[::-1]
    deltaP_head = Model_DP[i_pc][-1]['deltaP']+rouf*g*deltaPPlot_tspan/1000
    ax.plot(tspan , deltaP_head, '--', label=label) 
    ax.set_xlabel('Height (m)')
    ax.set_ylabel('deltaP (kPa)')
    ax.legend(loc='best')
#    ax.set_xlim(0, 0.4)
#    ax.set_ylim(0, 30)
"""
    
#%%
"""
def fMiniDenEps(x,p,ydata,densityCoeff):
    solids = 0.17
    N=1000
    g=9.8
    rouf = 1000
    lamda = 0.5
#    densityCoeff = x[0:3]
#    epsCoeff = x[3:5]
    epsCoeff=x
    miu = 0.1
    Height = BedHeight # unit m
    tspan=np.linspace(Height,0,N)
    c=[]
    res = np.zeros(len(p))
    for i_expP in range(len(p)):
        Height = BedHeight[np.where(Pc==p[i_expP])[0][0]] # unit m
        tspan=np.linspace(Height,0,N)
        c=[]
        def s2(t,y,c): # y (sigma) in kPa
            yA=y
            density=funcDensity(densityCoeff, yA)
            eps=funcEps2(epsCoeff,yA)
            m=solids*(1-eps)
            DGBouyantDh=m*(density/m-rouf)*g/1000# converted to kPa
            DfwDh = (2*lamda*miu*yA/radius) # yA has unit of kPa
            DsigmaDh = - DGBouyantDh + DfwDh
            return DsigmaDh

        sol2 = solve_ivp(lambda t, y: s2(t, y, c),
                    [tspan[0], tspan[-1]], yinit, t_eval=tspan)

        sigmaAbs2 = sol2.y[0]
        
    #    sigmaAbs2 = sol2
        densityProfile2=funcDensity(densityCoeff, sigmaAbs2)
        densityAvg2 = np.sum(densityProfile2*Height/N)/Height
        epsilonProfile2=funcEps2(epsCoeff, sigmaAbs2)
        epsilonAvg2 = np.sum(epsilonProfile2*Height/N)/Height
        DGBouyantDh2=solids*(1-epsilonProfile2)*(densityProfile2/(solids*(1-epsilonProfile2))-rouf)*g/1000 
        DfwDh2 = (2*lamda*miu*sigmaAbs2/radius)
#        res[i_expP] = (densityAvg2 - ydata[i_expP][0])*(epsilonAvg2-ydata[i_expP][1])
        res[i_expP]=pow((epsilonAvg2-ydata[i_expP][1]),2)
#        ax.plot(sol2.t,epsilonProfile2)
    return np.sqrt(np.sum(res))

#x0 = np.concatenate((densityCoeff,EpsCoeff2))
x0=EpsCoeff2
Pcdata=Pc[1:]
ydata=np.zeros([len(Pcdata),2])
for i in range(len(Pcdata)):
    ydata[i][0] = packingDensity[i+1]
    ydata[i][1] = epsilon[i+1]

#fig = plt.figure(figsize=(8, 6))
#ax = fig.add_subplot(1, 1, 1)
res_lsq_PackBedModel=least_squares(fMiniDenEps,x0, loss='soft_l1', f_scale=1, args=(Pcdata,ydata,densityCoeff))
#densityCoeff3 = res_lsq_PackBedModel.x[0:3]
#EpsCoeff3 = res_lsq_PackBedModel.x[3:5]            
EpsCoeff3 = res_lsq_PackBedModel.x    
"""