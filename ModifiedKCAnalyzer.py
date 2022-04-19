#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 16:49:39 2022

@author: yli5
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from scipy.optimize import leastsq

from ClassicKCAnalyzer import ClassicKCAnalyzer
from ColumnExperiment import ColumnExperiment
from Porosity import Porosity

class ModifiedKCAnalyzer(ClassicKCAnalyzer):
    """
    Fit experimental data and get modified KC parameters and fitting
    Takes an ColumnExperiment object to initialize
    """
    def __init__(self, exp:ColumnExperiment,porosity:Porosity=None):
        self.exp = exp
        self.name = exp.name
        self.temperature = exp.temperature
        self.viscosity = exp.viscosity
        if isinstance(porosity, Porosity):
            self.porosity = porosity
        else:
            self.porosity = Porosity(exp)
        self.SvCoeff_Pc ,self.Sv_exp_Pc, self.Pc, self.pressure_fitting_Pc, self.Sv_fitting_Pc = self.fit_Sv()
        self.exp_data_Pc, self.model_data_Pc = self.model_fitting()

    @staticmethod
    def func_Sv(x,p):
        """
        model the specific surface area (volume basis) as a function or pressure (p)
        """
        return x[0]/(1+x[1]*p)
    
    @staticmethod
    def minimize_func_Sv(x,p,y):
        return ModifiedKCAnalyzer.func_Sv(x,p) - y
    
    @staticmethod
    def func_modifiedKC(SvCoeff,U,EpsCoeff,viscosity,k,p):
        """
        instead of passing constant Sv and porosity,
        Modified KC takes SvCoeff and EpsCoeff as well as pressure (p) to compute Sv and porosity as functions of pressure
        """
        Sv = ModifiedKCAnalyzer.func_Sv(SvCoeff,p)
        eps = Porosity.func_porosity(EpsCoeff,p)
        return k*Sv*Sv*U*viscosity*pow((1-eps),2)/(100*pow(eps,3))

    def fit_Sv(self,x_SvKC_0=np.ones(1),x_SvMKC_0=np.array([40,0.02]),Pdata=np.zeros(1),EpsCoeff=np.zeros(1),loss='soft_l1',f_scale=0.1,
                    bounds=([10,0.001],[1000,0.03]), method='trf'):
        """
        fit data for specific surface area
        Default using experimetn measured velocity and pressure differentials
        """
        if Pdata.all() == 0:
            Pdata = self.exp.Pc
#        if result.all() == 0:
#            result = self.exp.result
        if EpsCoeff.all() == 0:
            EpsCoeff = self.porosity.porosityCoeff_Pc

        N_Svfit = len(Pdata)
        specificSurfaceArea = np.zeros(N_Svfit)
        for i in range(N_Svfit):
            specificSurfaceArea[i] = self.specific_area_fit_CKC(n_fit=i,x0=x_SvKC_0,Pdata=Pdata, EpsCoeff=EpsCoeff)

        #self.Sv_exp = specificSurfaceArea # this is a list corresponding to each Pc
        #ydata = specificSurfaceArea
        #PcPlot = np.linspace(0,max(self.exp.Pc)+2,100)
        
        res_lsqSv=least_squares(ModifiedKCAnalyzer.minimize_func_Sv,x0=x_SvMKC_0, loss=loss, f_scale=f_scale, 
                                bounds=bounds, method=method, args=(Pdata,specificSurfaceArea))
        SvCoeff = res_lsqSv.x

        #pressure_exp = self.exp.Pc
        pressure_fitting  = np.linspace(0,max(Pdata)+2,100)
        Sv_fitting = ModifiedKCAnalyzer.func_Sv(SvCoeff,pressure_fitting)
        return SvCoeff, specificSurfaceArea, Pdata, pressure_fitting,Sv_fitting
    
    def update_Sv_fit(self,x_SvKC_0=np.ones(1),x_SvMKC_0=np.array([40,0.02]),Pdata=np.zeros(1),EpsCoeff=np.zeros(1),loss='soft_l1',f_scale=0.1,
                    bounds=([10,0.001],[1000,0.03]), method='trf'):
        self.SvCoeff_Pc ,self.Sv_exp_Pc, self.Pc, self.pressure_fitting_Pc, self.Sv_fitting_Pc = self.fit_Sv(x_SvKC_0=x_SvKC_0, x_SvMKC_0=x_SvMKC_0,
                                                    Pdata=Pdata,EpsCoeff=EpsCoeff,loss=loss,f_scale=f_scale,bounds=bounds,method=method)


    def model_fitting(self, Pdata=np.zeros(1),SvCoeff=np.zeros(1), EpsCoeff=np.zeros(1)):
        #self.specific_area_CKC = self.specific_area_fit_CKC(n_fit=self.n)
        #self.fit_Sv()
        if Pdata.all() == 0:
            Pdata = self.exp.Pc
        if SvCoeff.all() == 0:
            SvCoeff = self.SvCoeff_Pc
        if EpsCoeff.all() == 0:
            EpsCoeff = self.porosity.porosityCoeff_Pc

        exp_result = []
        model_result = []
        for i in range(len(Pdata)):
            #porosity_i =  Porosity.func_porosity(EpsCoeff,Pdata[i])
            exp_result.append(dict(
                avg_sfvel = self.exp.result[i]['avg_sfvel'],
                avg_pg = self.exp.result[i]['avg_pg'],
                stdev_sfvel = self.exp.result[i]['stdev_sfvel'],
                stdev_pg = self.exp.result[i]['stdev_pg'],
                Pc = Pdata[i]
            ))
            model_result.append(dict(
                avg_sfvel = self.exp.result[i]['avg_sfvel'],
                avg_pg = ModifiedKCAnalyzer.func_modifiedKC(SvCoeff,self.exp.result[i]['avg_sfvel'],
                                                            EpsCoeff,self.viscosity,
                                                            ClassicKCAnalyzer.k,Pdata[i])
            ))
        
        return exp_result, model_result

    def update_Ps(self, p_s:np.ndarray):
        """
        update particle experienced pressure to Ps
        """
        self.pressure_Ps = p_s
        self.porosity_Ps = Porosity(self.exp)
        self.porosity_Ps.update_Ps(p_s)
        self.SvCoeff_Ps ,self.Sv_exp_Ps, self.Ps, self.pressure_fitting_Ps, self.Sv_fitting_Ps = self.fit_Sv(Pdata=p_s,EpsCoeff=self.porosity_Ps.porosityCoeff_Ps)
        self.exp_data_Ps, self.model_data_Ps = self.model_fitting(Pdata=self.pressure_Ps,SvCoeff=self.SvCoeff_Ps, EpsCoeff=self.porosity_Ps.porosityCoeff_Ps)
        
#    def predict_pressuredrop(self,U,p):
#        """
#        predict pressure drop based on velocity and pressure
#        """
#        return ModifiedKCAnalyzer.func_modifiedKC(self.SvCoeff,U,self.porosity.porosityCoeff,self.viscosity,
#                                                    ClassicKCAnalyzer.k,p)
#    def predict_Sv(self,p):
#        """
#        predict specific surface area based on pressure
#        """
#        return ModifiedKCAnalyzer.func_Sv(self.SvCoeff,p)



#%%       
if __name__ == '__main__':
    localdbpath='./datapickles/'
    exp_path='./exp/'

    import glob
    import os
    filenames_abs = glob.glob(exp_path+'*.xlsx')
    nexp = len(filenames_abs)
    ## extract basename of files
    filenames = [] # basename of files (with extension)
    names = [] # names of experiments (no extension)
    for i in range(nexp):
        filename = os.path.basename(filenames_abs[i])
        filenames.append(filename)
        names.append(filename.split('.')[0])
    exp = ColumnExperiment(filenames_abs[3], localdbpath)
    porosity = Porosity(exp)
    modifiedKC = ModifiedKCAnalyzer(exp)
    from ColumnPlots import ColumnPlots
    Sv_fittingcheck = ColumnPlots('Sv_fitting_check')
    xlabel = '$\mathregular{P_c}$ (kPa)'
    ylabel = 'Effective specific surface area ($\mathregular{cm^2 cm^{-3}}$)'
    Sv_fittingcheck.compare_exp_fitting(modifiedKC.Pc,modifiedKC.Sv_exp_Pc,modifiedKC.pressure_fitting_Pc,
                                        modifiedKC.Sv_fitting_Pc,xlabel,ylabel)
    #modifiedKC.model_fitting()
    
    MKCpressure_plot = ColumnPlots('test-exp_model','pressuredrop-3')
    MKCpressure_plot.compare_exp_model_pressuredrop(modifiedKC.exp_data_Pc,modifiedKC.model_data_Pc)
        

