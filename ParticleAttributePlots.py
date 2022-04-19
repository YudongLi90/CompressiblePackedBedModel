from matplotlib import pyplot as plt
import numpy as np
from ColumnPlots import ColumnPlots

class ParticleAttributePlots(ColumnPlots):

    def __init__(self,name=None):
        super().__init__(name=name)
    
    def compare_vol_cumudist_plot(self,data_lst,data_vol_cumu_lst,label_lst,xlabel="Size (mm)",ylabel="",xlimit=None,ylimit=None,
                                    save=False,savename=None):
        self.single_figure_head()
        i = 0
        for data, data_vol_cumu, label in zip(data_lst,data_vol_cumu_lst,label_lst):
            plt.plot(data,data_vol_cumu,label=label,color='C'+str(i),linewidth = 3)
            i += 1
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.set_x_y_limit(xlimit,ylimit)
        self.formatting_afterplot(legend_loc="best")
        if save and savename:
            self.fig.savefig(savename+'.jpg',dpi=300,bbox_inches='tight')
