import re
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from scipy.optimize import leastsq

from ColumnExperiment import ColumnExperiment
from Porosity import Porosity

class ClassicKCAnalyzer:
    """
    Perform traditional Kozeny Carman fitting
    Takes an ColumnExperiment object to initialize
    This theory assumes specific surface area is not changing with compaction
    The modified KC takes account into the specific surface area change with compaction
    """
    k = 5.55

    def __init__(self, exp:ColumnExperiment,porosity:Porosity=None,n=0):
        self.exp = exp
        self.name = exp.name
        self.temperature = exp.temperature
        self.viscosity = exp.viscosity
        self.n = n # this is the number of experiment used to fit for specific surface area
        if isinstance(porosity, Porosity):
            self.porosity = porosity
        else:
            self.porosity = Porosity(exp)
        self.exp_data_Pc, self.model_data_Pc = self.model_fitting()

        

    @staticmethod
    def func_KC(x,U,eps,viscosity,k):
        S=x[0]
        return k*S*S*U*viscosity*pow((1-eps),2)/(100*pow(eps,3))

    @staticmethod
    def minimize_func_KC(x,U,y,eps,viscosity,k):
        return ClassicKCAnalyzer.func_KC(x,U,eps,viscosity,k)-y

    def specific_area_fit_CKC(self,n_fit,x0=np.ones(1),Pdata=np.zeros(1),EpsCoeff=np.zeros(1),loss='soft_l1',f_scale=1):
        if Pdata.all() == 0:
            Pdata = self.exp.Pc
        if EpsCoeff.all() == 0:
            EpsCoeff = self.porosity.porosityCoeff_Pc
        porosity_n = Porosity.func_porosity(EpsCoeff,Pdata[n_fit])
        Udata=self.exp.result[n_fit]['avg_sfvel'] #mm/s
        ydata=self.exp.result[n_fit]['avg_pg']#kPa/m

        res_lsq=least_squares(ClassicKCAnalyzer.minimize_func_KC,x0, loss=loss, f_scale=f_scale, 
                                args=(Udata,ydata,porosity_n,self.viscosity,ClassicKCAnalyzer.k))
        # specific area calculated by Classical KC equation.
        return res_lsq.x

    def model_fitting(self, Pdata=np.zeros(1),EpsCoeff=np.zeros(1)):
        if Pdata.all() == 0:
            Pdata = self.exp.Pc
        if EpsCoeff.all() == 0:
            EpsCoeff = self.porosity.porosityCoeff_Pc
        specific_area_CKC = self.specific_area_fit_CKC(n_fit=self.n,Pdata=Pdata,EpsCoeff=EpsCoeff)
        exp_result = []
        model_result = []
        for i in range(len(Pdata)):
            porosity_i =  Porosity.func_porosity(EpsCoeff,Pdata[i])
            exp_result.append(dict(
                avg_sfvel = self.exp.result[i]['avg_sfvel'],
                avg_pg = self.exp.result[i]['avg_pg'],
                stdev_sfvel = self.exp.result[i]['stdev_sfvel'],
                stdev_pg = self.exp.result[i]['stdev_pg'],
                Pc = Pdata[i]
            ))
            model_result.append(dict(
                avg_sfvel = self.exp.result[i]['avg_sfvel'],
                avg_pg = ClassicKCAnalyzer.func_KC(specific_area_CKC,self.exp.result[i]['avg_sfvel'],porosity_i,
                                                    self.viscosity,ClassicKCAnalyzer.k)
            ))
        
        exp_data = exp_result
        model_data = model_result
        return exp_data, model_data

    def update_Ps(self,p_s:np.ndarray):
        """
        update particle experienced pressure to Ps
        """
        self.pressure_Ps = p_s
        self.porosity_Ps = Porosity(exp)
        self.porosity_Ps.update_Ps(p_s)
        self.exp_data_Ps, self.model_data_Ps = self.model_fitting(Pdata=self.pressure_Ps,EpsCoeff=self.porosity_Ps.porosityCoeff_Ps)




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
    modifiedKCfit = ClassicKCAnalyzer(exp)

