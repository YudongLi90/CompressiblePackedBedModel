import pandas as pd
from PropertyNN import PropertyNN


class specificAreaNN(PropertyNN):
    """
    Porosity Neural Network model
    """
    def __init__(self,name=None,exp_path='./exp/',localdbpath='./datapickles/',savepath='./NN_output/',checkpoint_path='./checkpoints/porosity/'):
        super().__init__(name=name,exp_path=exp_path,localdbpath=localdbpath,savepath=savepath,checkpoint_path=checkpoint_path)
    
    def _prepare_dataset(self, property_str='Sv_fitting',property_exp_str="Sv_exp", 
                                 pressure_str="pressure_fitting_Sv", pressure_exp_str="pressure_exp"):
        super()._prepare_dataset(property_str=property_str, property_exp_str=property_exp_str,
                                                                    pressure_str=pressure_str,pressure_exp_str=pressure_exp_str)