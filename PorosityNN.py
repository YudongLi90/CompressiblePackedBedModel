import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
"""
Plot preferences
"""
plt.rc('font', family='serif', serif='Times New Roman')
plt.rc('text', usetex=False)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.rc('axes', labelsize=18)

from PropertyNN import PropertyNN


class PorosityNN(PropertyNN):
    """
    Porosity Neural Network model
    """
    def __init__(self,name=None,exp_path='./exp/',localdbpath='./datapickles/',savepath='./NN_output/',checkpoint_path='./checkpoints/porosity/'):
        super().__init__(name=name,exp_path=exp_path,localdbpath=localdbpath,savepath=savepath,checkpoint_path=checkpoint_path)
    
    def _prepare_dataset(self, property_str='porosity_fitting',property_exp_str="porosity_exp", 
                                 pressure_str="pressure_fitting_porosity", pressure_exp_str="pressure_exp"):
        super()._prepare_dataset(property_str=property_str, property_exp_str=property_exp_str,
                                                                    pressure_str=pressure_str,pressure_exp_str=pressure_exp_str)
    
    def plot_training_performance(self,plot_training=True,save=False):
        if plot_training:
            fig = plt.figure(figsize=(12, 5))
            plt.plot(self.history.history['loss'])
            plt.plot(self.history.history['val_loss'])
            plt.title('loss vs. epochs')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Training', 'Validation'], loc='upper right')
        # compare test data with model
        fig = plt.figure(figsize=(8,6))
        plt.scatter(self.property_all,self.predicted_reversed.reshape((len(self.predicted_reversed),)))
        plt.plot(np.linspace(min(self.property_all),max(self.property_all),100), 
                np.linspace(min(self.property_all),max(self.property_all),100))

        # compare experiment measured data with NN predcition
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(1, 1, 1)
        plt.plot(self.property_exp,self.predicted_exp_reversed.reshape((len(self.predicted_exp_reversed),)),'o',color='C1',markersize=9)
        plt.plot(np.linspace(min(self.property_exp),max(self.property_exp),100), 
                np.linspace(min(self.property_exp),max(self.property_exp),100))
        plt.xlabel('Experiment')
        plt.ylabel('Prediction')
        ax.tick_params(axis = 'y', which = 'major', direction = 'out', length = 6.0, width = 2.0)
        ax.tick_params(axis = 'x', which = 'major', direction = 'out', length = 6.0, width = 2.0)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.8)
        fig.tight_layout()