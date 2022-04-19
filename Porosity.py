import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from scipy.optimize import leastsq

from ColumnExperiment import ColumnExperiment

class Porosity:
    """
    Fit porosity measurments to model
    """
    outlier = {'clean_wet_small_04_08.xlsx': [0] } # should only be cleaned once. Need to be careful about his.

    def __init__(self, exp:ColumnExperiment):
        self.name = exp.name
        self.exp = exp
        self._extract_exp_data()
        self.porosityCoeff_Pc, self.pressure_fitting_Pc, self.porosity_fitting_Pc = self.model_fitting()

    
    def _extract_exp_data(self, skip=None):
        if skip ==None:
            self.porosity_exp = self.exp.epsilon
            self.pressure_exp = self.exp.Pc
        else:
            self.porosity_exp = np.delete(self.exp.epsilon,skip,0)
            self.pressure_exp = np.delete(self.exp.Pc,skip,0)
    
    @staticmethod
    def func_porosity(x: np.ndarray,p:float):
        """
        porosity as function of pressure
        x is a 2 dimensional array (parameters)
        p is the pressure
        """
        return x[0]/(1+x[1]*p)

    @staticmethod
    def minimize_func_porosity(x:np.ndarray, p:float, y:float):
        """
        minimize and fit for parameters in func_porosity
        y is the target experiment porosity results
        """
        return Porosity.func_porosity(x,p) - y
    
    def model_fitting(self,x0=[1,1],Pdata=np.zeros(1),Porositydata=np.zeros(1),loss='soft_l1',f_scale=1):
        """
        fit parameters in func_pressure based on experiment measured result
        need to run _extract_exp_data() first
        """
        if Pdata.all() == 0: # by default, use Pc
            Pdata = self.pressure_exp
        if Porositydata.all() == 0:
            Porositydata = self.porosity_exp
        
        self.res_lsq=least_squares(Porosity.minimize_func_porosity,x0=x0, loss=loss, f_scale=f_scale, args=(Pdata,Porositydata))
        porosityCoeff = self.res_lsq.x

        pressure_fitting  = np.linspace(0,max(Pdata)+2,100)
        porosity_fitting = Porosity.func_porosity(porosityCoeff,pressure_fitting)
        return porosityCoeff, pressure_fitting, porosity_fitting

    def cleanup_Pc(self):
        self.porosity_exp_all = self.porosity_exp
        self.pressure_exp_all = self.exp.Pc
        self.porosityCoeff_Pc_all = self.porosityCoeff_Pc
        self.pressure_fitting_Pc_all = self.pressure_fitting_Pc
        self.porosity_fitting_Pc_all = self.porosity_fitting_Pc
        if self.name in Porosity.outlier:
            skip = Porosity.outlier[self.name]
            self._extract_exp_data(skip)
            self.porosityCoeff_Pc, self.pressure_fitting_Pc, self.porosity_fitting_Pc = self.model_fitting()
        return True
    
#    def predict(self,p:float):
#        """
#        predict porosity based on pressure
#        """
#        return Porosity.func_porosity(self.porosityCoeff,p)
    
    def update_Ps(self,p_s:float):
        """
        update particle experienced pressure to Ps
        """
        self.pressure_exp_Ps = p_s
        self.porosityCoeff_Ps, self.pressure_fitting_Ps, self.porosity_fitting_Ps = self.model_fitting(Pdata=self.pressure_exp_Ps)


    
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
    from ColumnPlots import ColumnPlots
    porosity= Porosity(exp)
    #porosity.model_fitting()
    porosityplot = ColumnPlots('porosity','porosity-3')
    xlabel = '$\mathregular{P_c}$ (kPa)'
    ylabel = 'Effective porosity (-)'
    porosityplot.compare_exp_fitting(porosity.pressure_exp,porosity.porosity_exp,porosity.pressure_fitting,porosity.porosity_fitting,xlabel,ylabel)