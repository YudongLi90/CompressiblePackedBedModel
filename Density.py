import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from scipy.optimize import leastsq

from ColumnExperiment import ColumnExperiment


class Density:
    outlier = {'clean_wet_small_04_08.xlsx': [0] } # should only be cleaned once. Need to be careful about his.
    cleaned = False   

    def __init__(self,  exp:ColumnExperiment):
        self.name = exp.name
        self.exp = exp
        self._extract_density_exp_data()
        self.densityCoeff_Pc, self.pressure_fitting_Pc, self.density_fitting_Pc = self.model_fitting()

    def _extract_density_exp_data(self,skip=None):
        """
        skip should be a list of indexes that you want to skip in extracting data (remove outlier result)
        """
        if skip ==None:
            self.density_exp = [self.exp.result[i]['packingDensity'] for i in range(len(self.exp.result))]
            self.pressure_exp = self.exp.Pc
        else:
            self.density_exp = [self.exp.result[i]['packingDensity'] for i in range(len(self.exp.result)) if i not in skip]
            self.pressure_exp = np.delete(self.exp.Pc,skip,0)

    @staticmethod
    def func_pressure(x: np.ndarray,p:float):
        """
        density as function of pressure
        x is a 3 dimensional array
        p is the pressure
        """
        return x[0]+x[1]*pow(p,x[2])
    
    @staticmethod
    def minimize_func_pressure(x:np.ndarray,p:float,y:float):
        """
        helper function in minimizing the difference between experiment and modeled density data
        see func_pressure for x and p
        y is the target density data
        """
        return Density.func_pressure(x,p)-y

    def model_fitting(self,x0=np.array([100,1,0.72]),Pdata=np.zeros(1),Densitydata=np.zeros(1),loss='soft_l1',f_scale=1,
                        bounds=([0,-2,0.65],[1000,100,0.9]), method='trf'):
        """
        fit parameters in func_pressure based on experiment measured result
        need to run _extract_density_exp_data() first
        """
        if Pdata.all() == 0: # by default, use Pc
            Pdata = self.pressure_exp
        if Densitydata.all() == 0:
            Densitydata = self.density_exp
        self.res_lsq=least_squares(Density.minimize_func_pressure,x0=x0, loss=loss, f_scale=f_scale,bounds=bounds,method=method,args=(Pdata, Densitydata))
        densityCoeff = self.res_lsq.x

        pressure_fitting  = np.linspace(0,max(Pdata)+2,100)
        density_fitting = Density.func_pressure(densityCoeff,pressure_fitting)
        return densityCoeff, pressure_fitting, density_fitting

    def update_model_fitting_Pc(self,x0=np.array([100,1,0.72]),Pdata=np.zeros(1),Densitydata=np.zeros(1),loss='soft_l1',f_scale=1,
                                bounds=([0,-2,0.65],[1000,100,0.9]), method='trf'):
        self.densityCoeff_Pc, self.pressure_fitting_Pc, self.density_fitting_Pc = self.model_fitting(x0=x0,Pdata=Pdata,Densitydata=Densitydata,
                                                                                                        loss=loss,f_scale=f_scale,bounds=bounds,
                                                                                                        method=method)
    def cleanup_Pc(self):
        self.density_exp_all = self.density_exp
        self.pressure_exp_all = self.exp.Pc
        self.densityCoeff_Pc_all = self.densityCoeff_Pc
        self.pressure_fitting_Pc_all = self.pressure_fitting_Pc
        self.density_fitting_Pc_all = self.density_fitting_Pc
        if self.name in Density.outlier:
            skip = Density.outlier[self.name]
            self._extract_density_exp_data(skip)
            self.densityCoeff_Pc, self.pressure_fitting_Pc, self.density_fitting_Pc = self.model_fitting()
        return True



    def update_Ps(self, p_s: np.ndarray):
        """
        update particle experienced pressure to Ps
        Only run this clean up if you are only outputing density results.
        If you cleaned up data in instantiating ColumnExperiment object, the data is very clean for density. Don't run this function again.
        """
        self.pressure_exp_Ps = p_s
        self.densityCoeff_Ps, self.pressure_fitting_Ps, self.density_fitting_Ps = self.model_fitting(Pdata=self.pressure_exp_Ps)



#    def predict(self, p:float):
#        """
#        predict density based on pressure
#        """
#        return Density.func_pressure(self.densityCoeff,p)

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
    density = Density(exp)
    
    #density.model_fitting()
    densityplot = ColumnPlots('density','density-3')
    xlabel = '$\mathregular{P_c}$ (kPa)'
    ylabel = 'Bulk density ($\mathregular{kg/m^3}$)'
    densityplot.compare_exp_fitting(density.pressure_exp,density.density_exp,density.pressure_fitting_Pc,density.density_fitting_Pc,xlabel,ylabel)

