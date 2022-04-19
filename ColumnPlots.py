import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

from ClassicKCAnalyzer import ClassicKCAnalyzer
from Density import Density
from ModifiedKCAnalyzer import ModifiedKCAnalyzer
from Porosity import Porosity
from typing import List

class ColumnPlots:
    """
    a collection of plot functions usefule for compressible packed bed flow column result visulization
    """
    markerlist = ['o','v','s','h','X','D','P','^','<','>','p','*']

    def __init__(self,name=None,figurename=None, labelsize=18,plot=True, save=False):
        self.name=name
        self.figurename=figurename
        self.labelsize = labelsize
        self.plot = plot
        self.save = save
        if save and figurename==None:
            raise ValueError('Please provide a figurename if you want to save figure to file!')
        plt.rc('font', family='serif', serif='Times New Roman')
        plt.rc('text', usetex=False)
        plt.rc('xtick', labelsize=self.labelsize)
        plt.rc('ytick', labelsize=self.labelsize)
        plt.rc('axes', labelsize=self.labelsize)

    def single_figure_head(self):
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(1, 1, 1)

    def set_x_y_limit(self,xlimit=None,ylimit=None):
        if xlimit != None and ylimit != None:
            self.ax.set_xlim(left=xlimit[0], right=xlimit[1])
            self.ax.set_ylim(bottom=ylimit[0],top=ylimit[1])
        elif ylimit == None and xlimit != None:
            self.ax.set_xlim(left=xlimit[0], right=xlimit[1])
        elif xlimit == None and ylimit != None:
            self.ax.set_ylim(bottom=ylimit[0],top=ylimit[1])
        else:
            pass
        
        
    def formatting_afterplot(self,legend_loc = 'upper left'):
        plt.legend(loc = legend_loc,prop={'size': 18, 'weight':'bold'},frameon=False,)
        self.ax.tick_params(axis = 'y', which = 'major', direction = 'out', length = 6.0, width = 2.0)
        self.ax.tick_params(axis = 'x', which = 'major', direction = 'out', length = 6.0, width = 2.0)
        for axis in ['top','bottom','left','right']:
            self.ax.spines[axis].set_linewidth(1.8)
        self.fig.tight_layout()

    def compare_exp_fitting(self,exp_x, exp_y, fit_x, fit_y,xlabel,ylabel,xlimit=None,ylimit=None, show_exp=True, show_model=True):
        self.single_figure_head()
        if show_exp:
            plt.plot(exp_x,exp_y, 's',linestyle='',markersize = 12, label='Experiment')
        if show_model:
            plt.plot(fit_x, fit_y, 'C0-', label='Model',linewidth = 3)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.set_x_y_limit(xlimit,ylimit)
        self.formatting_afterplot()
        plt.show()
        if self.save and self.figurename:
            self.fig.savefig(self.figurename,dpi=300,bbox_inches='tight')

    def expriment_pressurdrop_plot(self,result: list,xlabel:str='Superficial velocity (mm/s)',ylabel:str='dP/dz (kPa/m)',xlimit=None,ylimit=None):
        """
        Plot experiment data only
        Pressure drop as a function of superficial velocity (sfvel)
        Use error bars
        """
        self.single_figure_head()
        for i, data in zip(range(len(result)), result):
            label = '$\mathregular{P_c}$ = %0.2f kPa' % (data['Pc'])
            plt.errorbar(data['avg_sfvel'],data['avg_pg'],xerr=data['stdev_sfvel'],yerr=data['stdev_pg'],\
                 color='C'+str(i),markersize = 4, fmt='-s',label=label, capsize= 5, capthick=1)

        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.set_x_y_limit(xlimit,ylimit)
        self.formatting_afterplot()
        plt.show()
        if self.save and self.figurename:
            self.fig.savefig(self.figurename,dpi=300,bbox_inches='tight')

    def compare_exp_model_pressuredrop(self,exp_result:list,model_result:list,
                                        xlabel:str='Superficial velocity (mm/s)',
                                        ylabel:str='dP/dz (kPa/m)',xlimit=None,ylimit=None,save=False, savename=None):
        """
        Plot experiment and model prediction comparison for pressure drop plot
        """
        self.single_figure_head()
        for i, exp_data,model_data in zip(range(len(exp_result)), exp_result,model_result):
            label = '$\mathregular{P_c}$ = %0.2f kPa' % (exp_data['Pc'])
            plt.errorbar(exp_data['avg_sfvel'],exp_data['avg_pg'],xerr=exp_data['stdev_sfvel'],yerr=exp_data['stdev_pg'],
                        color='C'+str(i),markersize = 4, fmt='s',label=label, capsize= 5, capthick=1)
            plt.plot(model_data['avg_sfvel'], model_data['avg_pg'],'C'+str(i)+'-',linewidth = 3)
        
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.set_x_y_limit(xlimit,ylimit)
        self.formatting_afterplot()
        #plt.show()
        if save and savename:
            self.fig.savefig(savename+'.jpg',dpi=300,bbox_inches='tight')

    def plotCPBModelExpDPCompare(self,CPB_Data,Pc, result,fn_str='',xlimit=None,ylimit=None, useMarker=True,markersize = 6, fmt='--',
                             capsize= 5,alpha=1, capthick=1,linewidth = 3,save=False
                             ):
        # Plot data from CPBModelExpCompare return;
        # compare CPB model with expeirment
        self.single_figure_head()
        istart=CPB_Data['i_init']
        N_exp=CPB_Data['totalNo']

        for i_exp in range(istart,N_exp):
            label = '$\mathregular{P_c}$ = %0.2f kPa' % (Pc[i_exp])
            plt.errorbar(result[i_exp]['avg_sfvel'],result[i_exp]['avg_pg'],\
                        xerr=result[i_exp]['stdev_sfvel'],yerr=result[i_exp]['stdev_pg'],\
                        color='C'+str(i_exp),marker=ColumnPlots.markerlist[i_exp],markersize = 12, fmt='s',label=label, capsize= 6, capthick=2)
        for i_exp in range(istart,N_exp):
            Model2_avgDP = CPB_Data['Model_avgDP'][i_exp-istart]
            Model2_stdDP = CPB_Data['Model_stdDP'][i_exp-istart]
            if useMarker:
                plt.errorbar(result[i_exp]['avg_sfvel'],Model2_avgDP,\
                            yerr=Model2_stdDP,\
                            color='C'+str(i_exp),marker=ColumnPlots.markerlist[i_exp],\
                            markersize = markersize, fmt=fmt,capsize= capsize, \
                            alpha=alpha,capthick=capthick,linewidth = linewidth)
            else:
                plt.errorbar(result[i_exp]['avg_sfvel'],Model2_avgDP,\
                            yerr=Model2_stdDP,\
                            color='C'+str(i_exp),\
                            fmt=fmt,capsize= capsize, \
                            alpha=alpha,capthick=capthick,linewidth = linewidth)
                        
        plt.legend(loc = 'upper left',prop={'size': 16, 'weight':'bold'},frameon=False,)
        self.ax.set_xlabel('Superficial velocity (mm/s)')
        self.ax.set_ylabel('dP/dz (kPa/m)') 
        self.set_x_y_limit(xlimit,ylimit)
        self.formatting_afterplot()
        plt.show()
        if save:
            self.fig.savefig('./asMillWM/asMillWM_BedModelPredictionValidation_Parameters_'+fn_str+'.jpg',dpi=300,bbox_inches='tight')
    
    def plot_scaleup(self,scaleUpData:list,variable:list,setxlimit=False,xlimit=[],setylimit=False,ylimit=[],rouf = 1000.0,save=False,savename=""):
        """
        Plot scale up study result
        ScaleUpData: a list of dictionary (return from PackedBed.get_bedproperties).
        variable: a list of strings for legend: friction coefficients = 1 etc.
        savename: a string to distinguish from other saved figures using different variable
        """
        self.single_figure_head()
        g = 9.8
        color2='tab:blue'
        color1='tab:red'
        self.ax2 = self.ax.twinx()
        
        for i in range(len(variable)):
            deltaPPlot_tspan=scaleUpData[i]['height'][::-1]
            deltaP_head = scaleUpData[i]['deltaP']+rouf*g*deltaPPlot_tspan/1000
            self.ax2.plot(scaleUpData[i]['height'],scaleUpData[i]['stress'],ls='none',color='C7',\
                    marker=ColumnPlots.markerlist[i],markevery=50,markersize = 9,label=variable[i])
            self.ax2.plot(scaleUpData[i]['height'],scaleUpData[i]['stress'],'--',linewidth=3,color=color2)
            
            self.ax.plot(scaleUpData[i]['height'],scaleUpData[i]['deltaP'],ls='none',color='C7',\
                    marker=ColumnPlots.markerlist[i],markevery=50,markersize = 9)
            self.ax.plot(scaleUpData[i]['height'],scaleUpData[i]['deltaP'],'--',linewidth=3,color=color1)
        #    ax.plot(scaleUp_LDratio[i]['height'],deltaP_head,ls='none',color='C7',\
        #            marker=markerlist[i],markevery=50,markersize = 9)
        #    ax.plot(scaleUp_LDratio[i]['height'],deltaP_head,'--',linewidth=3,color=color1)
        #    ax.plot(scaleUp_LDratio[i]['height'],scaleUp_LDratio[i]['dpdz'],label=str(SU_radius[i]))  
        #    ax.plot(scaleUp_LDratio[i]['height'],deltaP_head,label=str(SU_radius[i]))
        self.ax.set_xlabel('z (m)')
        self.ax.set_ylabel('Cumulative dp/dz (kPa)',color=color1)
        self.ax.tick_params(axis = 'y', which = 'major', direction = 'out', length = 6.0, width = 2.0,color=color1,labelcolor=color1)
        self.ax.tick_params(axis = 'x', which = 'major', direction = 'out', length = 6.0, width = 2.0)
        self.ax2.set_ylabel('$\mathregular{P_s}$ (kPa)',color=color2)
        self.ax2.tick_params(axis = 'y', which = 'major', direction = 'out', length = 6.0, width = 2.0,color=color2,labelcolor=color2)
        self.ax2.tick_params(axis = 'x', which = 'major', direction = 'out', length = 6.0, width = 2.0)
        self.ax2.legend(bbox_to_anchor=(0.7,1),loc='upper center',prop={'size': 14, 'weight':'bold'},frameon=False) 
        self.ax.spines['left'].set_color('tab:red')
        self.ax.spines['right'].set_color('tab:blue')
        if setxlimit:
            self.ax.set_xlim(xlimit[0],xlimit[1])
        if setylimit:
            self.ax.set_ylim(ylimit[0], ylimit[1])
            self.ax2.set_ylim(ylimit[2], ylimit[3])
        for axis in ['top','bottom','left','right']:
                self.ax.spines[axis].set_linewidth(1.8)
        self.fig.tight_layout()
        plt.show() 
        if save:
            self.fig.savefig(savename+'.jpg',dpi=300,bbox_inches='tight') 
    
    def compare_density_plot(self,density_lst:List[Density],label_lst:List[str],xlimit=None,ylimit=None,save=False,savename=None):
        """
        plot all density data from density_lst, with label_lst
        """
        self.single_figure_head()
        i = 0
        for density,label in zip(density_lst,label_lst):
            self.ax.plot(density.pressure_exp,density.density_exp,marker=ColumnPlots.markerlist[i],linestyle='',color='C'+str(i),markersize = 12,label=label)
            self.ax.plot(density.pressure_fitting_Pc,density.density_fitting_Pc,color='C'+str(i), linewidth = 3)
            self.ax.set_xlabel('$\mathregular{P_c}$ (kPa)')
            self.ax.set_ylabel('Bulk density ($\mathregular{kg/m^3}$)')
            i += 1
        self.set_x_y_limit(xlimit,ylimit)
        self.formatting_afterplot()
        if save and savename:
            self.fig.savefig(savename+'.jpg',dpi=300,bbox_inches='tight')

    def plot_density(self,density:Density,label,label_model,xlimit=None,ylimit=None,save=False,savename=None):
        """
        plot all density data from density_lst, with label_lst
        """
        i = 0
        self.single_figure_head()
        self.ax.plot(density.pressure_exp,density.density_exp,'s',linestyle='',color='C'+str(i),markersize = 12,label=label)
        self.ax.plot(density.pressure_fitting_Pc,density.density_fitting_Pc,color='C'+str(i), label=label_model,linewidth = 3)
        self.ax.set_xlabel('$\mathregular{P_c}$ (kPa)')
        self.ax.set_ylabel('Bulk density ($\mathregular{kg/m^3}$)')
        self.set_x_y_limit(xlimit,ylimit)
        self.formatting_afterplot()
        if save and savename:
            self.fig.savefig(savename+'.jpg',dpi=300,bbox_inches='tight')
    
    def compare_porosity_plot(self,porosity_lst:list,modifiedKC_lst,label_lst,xlimit=None,ylimit=None,save=False,savename=None):
        """
        plot all density data from porosity_lst, and modifiedKC_lst, with label_lst
        The fitted curve use modifedKC_lst fitting. Since this is used for scale up
        """
        if label_lst == None or len(porosity_lst) == 1:
            label_lst = ["Experiment"]
            label_model = ['']
        else:
            label_model_lst = label_lst
        self.single_figure_head()
        i = 0
        for porosity,modifiedKC,label,label_model in zip(porosity_lst,modifiedKC_lst,label_lst,label_model_lst):
            self.ax.plot(porosity.pressure_exp,porosity.porosity_exp,marker=ColumnPlots.markerlist[i],linestyle='',color='C'+str(i),markersize = 12,label=label)
            self.ax.plot(porosity.pressure_fitting_Pc,porosity.porosity_fitting_Pc,color='C'+str(i), linewidth = 3)
            #self.ax.plot(modifiedKC.porosity.pressure_fitting_Pc,modifiedKC.porosity.porosity_fitting_Pc,color='C'+str(i), label='Model-'+label,linewidth = 3)
            self.ax.set_xlabel('$\mathregular{P_c}$ (kPa)')
            self.ax.set_ylabel('Effective porosity (-)')
            i += 1
        self.set_x_y_limit(xlimit,ylimit)
        self.formatting_afterplot(legend_loc='upper right')
        if save and savename:
            self.fig.savefig(savename+'.jpg',dpi=300,bbox_inches='tight')

    def plot_porosity(self,porosity:Porosity,label,label_model,xlimit=None,ylimit=None,save=False,savename=None):
        """
        plot all density data from porosity_lst, and modifiedKC_lst, with label_lst
        The fitted curve use modifedKC_lst fitting. Since this is used for scale up
        """
        i = 1
        self.single_figure_head()
        self.ax.plot(porosity.pressure_exp,porosity.porosity_exp,'s',linestyle='',color='C'+str(i),markersize = 12,label=label)
        self.ax.plot(porosity.pressure_fitting_Pc,porosity.porosity_fitting_Pc,color='C'+str(i), label=label_model,linewidth = 3)
        #self.ax.plot(modifiedKC.porosity.pressure_fitting_Pc,modifiedKC.porosity.porosity_fitting_Pc,color='C'+str(i), label='Model-'+label,linewidth = 3)
        self.ax.set_xlabel('$\mathregular{P_c}$ (kPa)')
        self.ax.set_ylabel('Effective porosity (-)')
        self.set_x_y_limit(xlimit,ylimit)
        self.formatting_afterplot(legend_loc='upper right')
        if save and savename:
            self.fig.savefig(savename+'.jpg',dpi=300,bbox_inches='tight')

    def compare_Sv_plot(self,modifiedKC_lst,label_lst,xlimit=None,ylimit=None,save=False,savename=None):
        """
        plot all Sv data
        """
        self.single_figure_head()
        i = 0
        for modifiedKC,label in zip(modifiedKC_lst,label_lst):
            self.ax.plot(modifiedKC.Pc,modifiedKC.Sv_exp_Pc,marker=ColumnPlots.markerlist[i],linestyle='',color='C'+str(i),markersize = 12,label=label)
            self.ax.plot(modifiedKC.pressure_fitting_Pc,modifiedKC.Sv_fitting_Pc,color='C'+str(i), linewidth = 3)
            #self.ax.plot(modifiedKC.porosity.pressure_fitting_Pc,modifiedKC.porosity.porosity_fitting_Pc,color='C'+str(i), label='Model-'+label,linewidth = 3)
            self.ax.set_xlabel('$\mathregular{P_c}$ (kPa)')
            self.ax.set_ylabel('Effective specific surface area ($\mathregular{cm^2 cm^{-3}}$)')
            i += 1
        self.set_x_y_limit(xlimit,ylimit)
        self.formatting_afterplot(legend_loc='upper right')
        if save and savename:
            self.fig.savefig(savename+'.jpg',dpi=300,bbox_inches='tight')
    
    def plot_Sv(self,modifiedKC:ModifiedKCAnalyzer,label,label_model,xlimit=None,ylimit=None,save=False,savename=None):
        """
        plot all Sv data
        """
        i = 3
        self.single_figure_head()
        self.ax.plot(modifiedKC.Pc,modifiedKC.Sv_exp_Pc,'s',linestyle='',color='C'+str(i),markersize = 12,label=label)
        self.ax.plot(modifiedKC.pressure_fitting_Pc,modifiedKC.Sv_fitting_Pc,color='C'+str(i), label=label_model,linewidth = 3)
        #self.ax.plot(modifiedKC.porosity.pressure_fitting_Pc,modifiedKC.porosity.porosity_fitting_Pc,color='C'+str(i), label='Model-'+label,linewidth = 3)
        self.ax.set_xlabel('$\mathregular{P_c}$ (kPa)')
        self.ax.set_ylabel('Effective specific surface area ($\mathregular{cm^2 cm^{-3}}$)')
        self.set_x_y_limit(xlimit,ylimit)
        self.formatting_afterplot(legend_loc='upper right')
        if save and savename:
            self.fig.savefig(savename+'.jpg',dpi=300,bbox_inches='tight')

    def plot_all_result(self,i,name,exp,density,porosity,classicKC,modifiedKC):
        """
        uses all experiment data (removing outliers) before experiment cleanning for modifiedKC fitting
        """
        xlabel = '$\mathregular{P_c}$ (kPa)'
        ylabel = 'Bulk density ($\mathregular{kg/m^3}$)'
        fig, ax = plt.subplots(2,3,figsize=[12,8])
        fig.suptitle(f"{i}:{name}",fontsize=20)
        #density
        ax[0][0].plot(density.pressure_exp,density.density_exp,'s',linestyle='',markersize = 12)
        ax[0][0].plot(density.pressure_fitting_Pc,density.density_fitting_Pc,'C0-', label='Model',linewidth = 3)
        ax[0][0].set_ylabel('Bulk density ($\mathregular{kg/m^3}$)')
        ax[0][0].set_title(list(map('{:.4f}'.format,density.densityCoeff_Pc)),fontsize=16)
        #porosity
        ax[0][1].plot(porosity.pressure_exp,porosity.porosity_exp,'s',linestyle='',markersize = 12)
        ax[0][1].plot(modifiedKC.porosity.pressure_fitting_Pc,modifiedKC.porosity.porosity_fitting_Pc,'C0-', 
                    label='Model',linewidth = 3)
        ax[0][1].set_ylabel('Effective porosity (-)')
        ax[0][1].set_title(list(map('{:.4f}'.format,porosity.porosityCoeff_Pc)),fontsize=16)
        #specific area
        ax[0][2].plot(modifiedKC.Pc,modifiedKC.Sv_exp_Pc,'s',linestyle='',markersize = 12)
        ax[0][2].plot(modifiedKC.pressure_fitting_Pc,modifiedKC.Sv_fitting_Pc,'C0-', label='Model',linewidth = 3)
        ax[0][2].set_ylabel('Effective specific surface area ($\mathregular{cm^2 cm^{-3}}$)')
        ax[0][2].set_title(list(map('{:.4f}'.format,modifiedKC.SvCoeff_Pc)),fontsize=16)
        #pressure-drop
        for i, data in zip(range(len(exp.result)), exp.result):
            label = '$\mathregular{P_c}$ = %0.2f kPa' % (data['Pc'])
            ax[1][0].errorbar(data['avg_sfvel'],data['avg_pg'],xerr=data['stdev_sfvel'],yerr=data['stdev_pg'],\
                color='C'+str(i),markersize = 4, fmt='-s',label=label, capsize= 5, capthick=1)
        ax[1][0].legend(loc = 'upper left',prop={'size': 12, 'weight':'bold'},frameon=False,)
        #pressure-drop classicKC
        for i, exp_data,model_data in zip(range(len(classicKC.exp_data_Pc)), classicKC.exp_data_Pc,
                                        classicKC.model_data_Pc):
            label = '$\mathregular{P_c}$ = %0.2f kPa' % (exp_data['Pc'])
            ax[1][1].errorbar(exp_data['avg_sfvel'],exp_data['avg_pg'],xerr=exp_data['stdev_sfvel'],
                            yerr=exp_data['stdev_pg'],
                        color='C'+str(i),markersize = 4, fmt='-s',label=label, capsize= 5, capthick=1)
            ax[1][1].plot(model_data['avg_sfvel'], model_data['avg_pg'],'C'+str(i)+'-',linewidth = 3)
        ax[1][1].legend(loc = 'upper left',prop={'size': 12, 'weight':'bold'},frameon=False,)
        #pressure-drop modifiedKC
        for i, exp_data,model_data in zip(range(len(modifiedKC.exp_data_Pc)), modifiedKC.exp_data_Pc,
                                        modifiedKC.model_data_Pc):
            label = '$\mathregular{P_c}$ = %0.2f kPa' % (exp_data['Pc'])
            ax[1][2].errorbar(exp_data['avg_sfvel'],exp_data['avg_pg'],xerr=exp_data['stdev_sfvel'],
                            yerr=exp_data['stdev_pg'],
                        color='C'+str(i),markersize = 4, fmt='-s',label=label, capsize= 5, capthick=1)
            ax[1][2].plot(model_data['avg_sfvel'], model_data['avg_pg'],'C'+str(i)+'-',linewidth = 3)
        ax[1][2].legend(loc = 'upper left',prop={'size': 12, 'weight':'bold'},frameon=False,)  
        fig.tight_layout()



        

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
    from ColumnExperiment import ColumnExperiment
    exp = ColumnExperiment(filenames_abs[3], localdbpath)
    pressure_plot = ColumnPlots('test-exp','pressuredrop-3')
    pressure_plot.expriment_pressurdrop_plot(exp.result)
    from ClassicKCAnalyzer import ClassicKCAnalyzer
    from Porosity import Porosity
    porosity = Porosity(exp)
    claKC = ClassicKCAnalyzer(exp)
    #claKC.specific_area_fit_CKC()
    #claKC.predict()
    KCpressure_plot = ColumnPlots('test-exp_model','pressuredrop-3')
    KCpressure_plot.compare_exp_model_pressuredrop(claKC.exp_data_Pc,claKC.model_data_Pc)


