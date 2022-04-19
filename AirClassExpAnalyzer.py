from cmath import exp
import glob
import os

from isort import file
from matplotlib import pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

from ColumnExperiment import ColumnExperiment
from ColumnPlots import ColumnPlots
from Density import Density
from Porosity import Porosity
from ClassicKCAnalyzer import ClassicKCAnalyzer
from ModifiedKCAnalyzer import ModifiedKCAnalyzer
from PackedBedScaleUp import PackedBedScaleUp

plt.rc('font', family='serif', serif='Times New Roman')
plt.rc('text', usetex=False)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.rc('axes', labelsize=18)

class AirClassExpAnalyzer():
    """
    a high level class wraps more lower level classes
    This is supposed to be used for the air classified corn stover permeability experiment
    """
    LARGE_EXP = ["clean_dry_large_03_19","clean_wet_large_03_11","dirty_dry_large_02_25","dirty_wet_large_03_05"]
    SMALL_EXP = ["clean_dry_small_04_21","clean_wet_small_04_08","dirty_dry_small_04_05","dirty_wet_small_04_22"]
    DIRTY_DRY = ["dirty_dry_large_02_25","dirty_dry_small_04_05","dirty_dry_unfractionated_05_12"]

    LABEL_DICT = {
        "dirty_dry_unfractionated_05_12"    : "FF-LM-Unsized",
        "dirty_dry_large_02_25"             : "FF-LM-Large",
        "dirty_dry_small_04_05"             : "FF-LM-Small",
        "dirty_wet_large_03_05"             : "FF-HM-Large",
        "dirty_wet_small_04_22"             : "FF-HM-Small",
        "clean_dry_large_03_19"             : "HF-LM-Large",
        "clean_dry_small_04_21"             : "HF-LM-Small",
        "clean_wet_large_03_11"             : "HF-HM-Large",
        "clean_wet_small_04_08"             : "HF-HM-Small"
    }

    def __init__(self,exp_path,localdbpath):
        self.exp_path = exp_path
        self.localdbpath = localdbpath
        self._read_data()

    def _read_data(self):
        filenames_abs = glob.glob(self.exp_path+'*.xlsx')
        nexp = len(filenames_abs)
        ## extract basename of files
        filenames = [] # basename of files (with extension)
        names = [] # names of experiments (no extension)
        experiment_lst = []
        resultdict = {}
        for i in range(nexp):
            filename = os.path.basename(filenames_abs[i])
            filenames.append(filename)
            name = filename.split('.')[0]
            names.append(name)
            exp = ColumnExperiment(filenames_abs[i],self.localdbpath) # without clean up
            experiment_lst.append(exp)
            _density = Density(exp)
            _density.cleanup_Pc() # clean up density data for one outlier
            porosity = Porosity(exp)
            porosity.cleanup_Pc()
            exp.clean_data()
            classicKC = ClassicKCAnalyzer(exp,porosity)
            modifiedKC = ModifiedKCAnalyzer(exp,porosity)
            su = PackedBedScaleUp(exp,_density,porosity,modifiedKC)
            scaleup = su.scaleup()
            resultdict[filename] = dict(name=name, exp=exp,density=_density,porosity=porosity,classicKC=classicKC,
                                                modifiedKC=modifiedKC,scaleup=scaleup)
        
        self.filenames_abs = filenames_abs
        self.filenames = filenames
        self.names = names
        self.experiment_alldata_lst = experiment_lst
        self.resultdict = resultdict

    def get_property_by_experiment_name(self,name:str,property:str):
        filename = name +'.xlsx'
        if filename in self.resultdict:
            if property in self.resultdict[filename]:
                return self.resultdict[filename][property]
            else:
                raise ValueError("Please provide a valid property name. Choose one from the list ['density','porosity','modifiedKC','scaleup']")
        else:
            raise ValueError("Please provide a valid experiment name (without .xlsx).")


    def compare_all_density(self):
        columplot = ColumnPlots('All_Density')
        label_lst = []
        density = []
        for i,(filename,exp_data) in enumerate(self.resultdict.items()):
            name = exp_data["name"]
            label = AirClassExpAnalyzer.LABEL_DICT[name]
            label_lst.append(label)
            density.append(exp_data["density"])
        columplot.compare_density_plot(density,label_lst)
        
    def compare_density(self,name_lst,xlimit=[0,None],ylimit=None,save=False,savename=None):
        columplot = ColumnPlots('All_Density')
        label_lst = []
        density_lst = []
        for name in name_lst:
            filename = name +'.xlsx'
            label = AirClassExpAnalyzer.LABEL_DICT[name]
            label_lst.append(label)
            density_lst.append(self.resultdict[filename]['density'])
        columplot.compare_density_plot(density_lst,label_lst,xlimit=xlimit,ylimit=ylimit,save=save,savename=savename)
    
    def compare_porosity(self,name_lst,xlimit=[0,None],ylimit=None,save=False,savename=None):
        columplot = ColumnPlots('compare_porosity')
        label_lst = []
        porosity_lst = []
        modifiedKC_lst = []
        for name in name_lst:
            filename = name +'.xlsx'
            label = AirClassExpAnalyzer.LABEL_DICT[name]
            label_lst.append(label)
            porosity_lst.append(self.resultdict[filename]['porosity'])
            modifiedKC_lst.append(self.resultdict[filename]['modifiedKC'])
        columplot.compare_porosity_plot(porosity_lst,modifiedKC_lst,label_lst,xlimit=xlimit,ylimit=ylimit,save=save,savename=savename)
    
    def compare_Sv(self,name_lst,xlimit=[0,None],ylimit=None,save=False,savename=None):
        columplot = ColumnPlots('compare_porosity')
        label_lst = []
        modifiedKC_lst = []
        for name in name_lst:
            filename = name +'.xlsx'
            label = AirClassExpAnalyzer.LABEL_DICT[name]
            label_lst.append(label)
            modifiedKC_lst.append(self.resultdict[filename]['modifiedKC'])
        columplot.compare_Sv_plot(modifiedKC_lst,label_lst,xlimit=xlimit,ylimit=ylimit,save=save,savename=savename)

    def compare_scaleup(self,name_lst,setxlimit=True,xlimit=[0,1],setylimit=False,ylimit=[],save=False,savename=""):
        columplot = ColumnPlots('compare_porosity')
        label_lst = []
        scaleup_lst = []
        for name in name_lst:
            filename = name +'.xlsx'
            label = AirClassExpAnalyzer.LABEL_DICT[name]
            label_lst.append(label)
            scaleup_lst.append(self.resultdict[filename]['scaleup'])
        columplot.plot_scaleup(scaleup_lst,label_lst,setxlimit=setxlimit,xlimit=xlimit,setylimit=setylimit,ylimit=ylimit,save=save,savename=savename)

    def show_all_result(self):
        columnplot = ColumnPlots('all_results')
        for i,(filename,exp_data) in enumerate(self.resultdict.items()):
            name = exp_data['name']
            label = AirClassExpAnalyzer.LABEL_DICT[name]
            columnplot.plot_all_result(i,label,exp_data['exp'],exp_data['density'],
                                    exp_data["porosity"],exp_data['classicKC'],exp_data["modifiedKC"])

    def plot_experiment_results(self,save=False):
        columplot = ColumnPlots('experiment_results')
        for i,(filename,exp_data) in enumerate(self.resultdict.items()):
            name = exp_data['name']
            short_name = AirClassExpAnalyzer.LABEL_DICT[name]
            #save figure directory; organize figures into each feedstock folders
            save_dir = './Figures/'+short_name
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            #plot density
            figure_name_density = save_dir+'/'+'Density_'+short_name
            colplot = ColumnPlots("Density_"+short_name)
            colplot.plot_density(exp_data["density"],label="Experiment",label_model="Model",xlimit=[0,None],save=save, savename=figure_name_density)
            #plot porosity
            figure_name_porosity = save_dir+'/'+ 'Porosity_'+short_name
            colplot = ColumnPlots("Porosity_"+short_name)
            colplot.plot_porosity(exp_data["porosity"],label="Experiment",label_model="Model",xlimit=[0,None],save=save, savename=figure_name_porosity)
            #plot Sv
            figure_name_Sv = save_dir+'/'+'Sv_'+short_name
            colplot = ColumnPlots("Sv_"+short_name)
            colplot.plot_Sv(exp_data["modifiedKC"],label="Experiment",label_model="Model",xlimit=[0,None],save=save, savename=figure_name_Sv)
            #plot ModifiedKC vs experiment
            figure_name_MKC = save_dir+'/'+'MKC_'+short_name
            colplot = ColumnPlots("MKC_"+short_name)
            exp_result = exp_data["modifiedKC"].exp_data_Pc
            model_result = exp_data["modifiedKC"].model_data_Pc
            colplot.compare_exp_model_pressuredrop(exp_result,model_result,xlimit=[0,None],ylimit=[0,None],save=save, savename=figure_name_MKC)
    
    def plot_scaleup_all(self,save=False):
        study_name_lst = ["FF-LM","Large","Small"]
        exp_names_lst = [AirClassExpAnalyzer.DIRTY_DRY,AirClassExpAnalyzer.LARGE_EXP,AirClassExpAnalyzer.SMALL_EXP]
        base_dir = './Figures/CompareScaleUp/'
        density_ylimits_lst = [[100,180],[90,150],[100,180]]
        porosity_ylim_lst = [[0.1,0.5],[0.25,0.55],[0.1,0.3]]
        SU_ylim_lst = [[0,37,0,37],[0,0.6,0,0.6],[0,37,0,37]]
        # compare Dirty Dry material with different size
        for study_name, exp_names, density_ylimits,porosity_ylim,SU_ylim in zip(study_name_lst,exp_names_lst,density_ylimits_lst,porosity_ylim_lst,SU_ylim_lst):
            save_dir = base_dir + study_name
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            density_save_name = save_dir+'/'+"Density_"+study_name
            porosity_save_name = save_dir+'/'+"Porosity_"+study_name
            Sv_save_name = save_dir+'/'+"Sv_"+study_name
            scaleup_save_name = save_dir+'/'+"ScaleUp_"+study_name
            self.compare_density(exp_names,save=save,ylimit=density_ylimits,savename=density_save_name)
            self.compare_porosity(exp_names,save=save,ylimit=porosity_ylim,savename=porosity_save_name)
            self.compare_Sv(exp_names,save=save,savename=Sv_save_name)
            self.compare_scaleup(exp_names,save=save,setylimit=True,ylimit=SU_ylim,savename=scaleup_save_name)




    def get_exp_name(self,n,printall=True):
        """
        return experiment name of experiment number n (n start from 0)
        This depends on the glob function which decides which file goes first
        """
        print(f"The number {n} file (indexing from 0) is {self.names[n]}.")
        if printall:
            for i,name in enumerate(self.names):
                print(f"{i} : {name}")
        return self.names[n]
    
    def get_idx_from_name(self):
        """
        This is a little tricky. Not implemented for now.
        """
        pass

        


