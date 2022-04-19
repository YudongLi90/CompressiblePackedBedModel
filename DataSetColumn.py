
import os
import numpy as np
import pandas as pd
from AirClassExpAnalyzer import AirClassExpAnalyzer
from ParticleSizeAnalyzer import ParticleSizeAnalyzer
from MaterialAttributePCA import MaterialAttributePCA
import datetime


class DataSetColumn:
    """
    Generate dataset for training
    """
    names = ['clean_wet_large_03_11',
    'dirty_wet_small_04_22', 
    'clean_dry_small_04_21', 
    'dirty_dry_large_02_25', 
    'clean_wet_small_04_08',
    'clean_dry_large_03_19',
    'dirty_dry_small_04_05', 
    'dirty_dry_unfractionated_05_12', 
    'dirty_wet_large_03_05'
    ]
    train = names[:-2]
    test = names[-2:]

    def __init__(self,exp_path,localdbpath,read=True,cache=True):
        #self.name = name
        self.exp_path = exp_path
        self.localdbpath = localdbpath
        self.cache = cache
        self.data_pickle_path = os.path.join(localdbpath,'dataset/')
        if not os.path.exists(self.data_pickle_path):
            os.makedirs(self.data_pickle_path)
        self.picklefile_train = os.path.join(self.data_pickle_path, 'column_dataset_train.pkl')
        self.picklefile_test = os.path.join(self.data_pickle_path, 'column_dataset_test.pkl')
        if read and os.path.isfile(self.picklefile_train) and os.path.isfile(self.picklefile_test):
            print("Read pickle file to training dataset")
            self.df_train = pd.read_pickle(self.picklefile_train)
            print("Read pickle fiel to test dataset")
            self.df_test = pd.read_pickle(self.picklefile_test)
        else:
            print("Preparing dataset. This process will take a couple of minutes.")    
            self.df_train = pd.DataFrame()
            self.df_test = pd.DataFrame()
            self._prepare_particle_data()
            self._prepare_density_data()
            self._prepare_porosity_data()
            self._prepare_Sv_data()
            if cache:
                self._dump()

    def _get_data_from_name_list(self,name_lst):
        df_list = pd.DataFrame()
        df_Dv_data = pd.DataFrame()
        for name in name_lst:
            particle_data = self.psd.get_particle_by_name(name)
            df_list = pd.concat(df_list,particle_data)
        return df_list

    def _prepare_PC_data_to_pd(self,pc1_dict:dict,pc2_dict:dict) -> pd.DataFrame:
        df_temp = pd.DataFrame(pc1_dict)
        df_final = df_temp.rename(columns={'Dv_result': 'PC_1_Dv'})
        df2 = pd.DataFrame(pc2_dict)
        df_final['PC_2_Dv'] = df2['Dv_result']
        return df_final

    def _prepare_particle_data(self):
        """
        prepare particle data for training
        """
        self.psd = ParticleSizeAnalyzer()
        attribute_lst=['PC_1','PC_2']
        # get particle data from a preset list (train,test)
        # get DV10,DV50,DV90 for componenets
        self.PC1_Dv_data_train = self.psd.get_Dv_values(DataSetColumn.train,attribute_lst[0],byExpName=True)
        self.PC2_Dv_data_train = self.psd.get_Dv_values(DataSetColumn.train,attribute_lst[1],byExpName=True)
        self.PC1_Dv_data_test = self.psd.get_Dv_values(DataSetColumn.test,attribute_lst[0],byExpName=True)
        self.PC2_Dv_data_test = self.psd.get_Dv_values(DataSetColumn.test,attribute_lst[1],byExpName=True)
        # unpack
        self.df_train = self._prepare_PC_data_to_pd(self.PC1_Dv_data_train,self.PC2_Dv_data_train)
        self.df_test = self._prepare_PC_data_to_pd(self.PC1_Dv_data_test,self.PC2_Dv_data_test)

    def _find_exp_name_index_in_df(self,df,name,col_name='exp_name'):
        return df[col_name][df[col_name]==name].index[0]

    def _prepare_density_data(self):
        self.experiments = AirClassExpAnalyzer(exp_path=self.exp_path,localdbpath=self.localdbpath)
        #self.df_train['pressure_fitting'] = np.zeros(len(DataSetColumn.train))
        pressure_fitting_train = []
        density_fitting_train = []
        pressure_exp_train = []
        density_exp_train = []
        for name in self.df_train['exp_name']:
            #idx = self._find_exp_name_index_in_df(self.df_train,name)
            pressure_fitting_train.append(self.experiments.resultdict[name+'.xlsx']["density"].pressure_fitting_Pc)
            density_fitting_train.append(self.experiments.resultdict[name+'.xlsx']["density"].density_fitting_Pc)
            pressure_exp_train.append(self.experiments.resultdict[name+'.xlsx']["density"].pressure_exp)
            density_exp_train.append(self.experiments.resultdict[name+'.xlsx']["density"].density_exp)

        self.df_train["pressure_fitting"] = pressure_fitting_train
        self.df_train["density_fitting"] = density_fitting_train
        self.df_train["pressure_exp"] = pressure_exp_train
        self.df_train["density_exp"] = density_exp_train

        pressure_fitting_test = []
        density_fitting_test = []
        pressure_exp_test = []
        density_exp_test = []
        for name in self.df_test['exp_name']:
            pressure_fitting_test.append(self.experiments.resultdict[name+'.xlsx']["density"].pressure_fitting_Pc)
            density_fitting_test.append(self.experiments.resultdict[name+'.xlsx']["density"].density_fitting_Pc)
            pressure_exp_test.append(self.experiments.resultdict[name+'.xlsx']["density"].pressure_exp)
            density_exp_test.append(self.experiments.resultdict[name+'.xlsx']["density"].density_exp)

        self.df_test["pressure_fitting"] = pressure_fitting_test
        self.df_test["density_fitting"] = density_fitting_test
        self.df_test["pressure_exp"] = pressure_exp_test
        self.df_test["density_exp"] = density_exp_test
    
    def _prepare_porosity_data(self):
        #self.experiments = AirClassExpAnalyzer(exp_path=self.exp_path,localdbpath=self.localdbpath)
        #self.df_train['pressure_fitting'] = np.zeros(len(DataSetColumn.train))
        pressure_fitting_train = []
        porosity_fitting_train = []
        pressure_exp_train = []
        porosity_exp_train = []
        for name in self.df_train['exp_name']:
            #idx = self._find_exp_name_index_in_df(self.df_train,name)
            pressure_fitting_train.append(self.experiments.resultdict[name+'.xlsx']["porosity"].pressure_fitting_Pc)
            porosity_fitting_train.append(self.experiments.resultdict[name+'.xlsx']["porosity"].porosity_fitting_Pc)
            pressure_exp_train.append(self.experiments.resultdict[name+'.xlsx']["porosity"].pressure_exp)
            porosity_exp_train.append(self.experiments.resultdict[name+'.xlsx']["porosity"].porosity_exp)

        self.df_train["pressure_fitting_porosity"] = pressure_fitting_train
        self.df_train["porosity_fitting"] = porosity_fitting_train
        self.df_train["pressure_exp"] = pressure_exp_train
        self.df_train["porosity_exp"] = porosity_exp_train

        pressure_fitting_test = []
        porosity_fitting_test = []
        pressure_exp_test = []
        porosity_exp_test = []
        for name in self.df_test['exp_name']:
            pressure_fitting_test.append(self.experiments.resultdict[name+'.xlsx']["porosity"].pressure_fitting_Pc)
            porosity_fitting_test.append(self.experiments.resultdict[name+'.xlsx']["porosity"].porosity_fitting_Pc)
            pressure_exp_test.append(self.experiments.resultdict[name+'.xlsx']["porosity"].pressure_exp)
            porosity_exp_test.append(self.experiments.resultdict[name+'.xlsx']["porosity"].porosity_exp)

        self.df_test["pressure_fitting_porosity"] = pressure_fitting_test
        self.df_test["porosity_fitting"] = porosity_fitting_test
        self.df_test["pressure_exp"] = pressure_exp_test
        self.df_test["porosity_exp"] = porosity_exp_test

    def _prepare_Sv_data(self):
        #self.experiments = AirClassExpAnalyzer(exp_path=self.exp_path,localdbpath=self.localdbpath)
        #self.df_train['pressure_fitting'] = np.zeros(len(DataSetColumn.train))
        pressure_fitting_train = []
        Sv_fitting_train = []
        pressure_exp_train = []
        Sv_exp_train = []
        for name in self.df_train['exp_name']:
            #idx = self._find_exp_name_index_in_df(self.df_train,name)
            pressure_fitting_train.append(self.experiments.resultdict[name+'.xlsx']["modifiedKC"].pressure_fitting_Pc)
            Sv_fitting_train.append(self.experiments.resultdict[name+'.xlsx']["modifiedKC"].Sv_fitting_Pc)
            pressure_exp_train.append(self.experiments.resultdict[name+'.xlsx']["modifiedKC"].Pc)
            Sv_exp_train.append(self.experiments.resultdict[name+'.xlsx']["modifiedKC"].Sv_exp_Pc)

        self.df_train["pressure_fitting_Sv"] = pressure_fitting_train
        self.df_train["Sv_fitting"] = Sv_fitting_train
        self.df_train["pressure_exp"] = pressure_exp_train
        self.df_train["Sv_exp"] = Sv_exp_train

        pressure_fitting_test = []
        Sv_fitting_test = []
        pressure_exp_test = []
        Sv_exp_test = []
        for name in self.df_test['exp_name']:
            pressure_fitting_test.append(self.experiments.resultdict[name+'.xlsx']["modifiedKC"].pressure_fitting_Pc)
            Sv_fitting_test.append(self.experiments.resultdict[name+'.xlsx']["modifiedKC"].Sv_fitting_Pc)
            pressure_exp_test.append(self.experiments.resultdict[name+'.xlsx']["modifiedKC"].Pc)
            Sv_exp_test.append(self.experiments.resultdict[name+'.xlsx']["modifiedKC"].Sv_exp_Pc)

        self.df_test["pressure_fitting_Sv"] = pressure_fitting_test
        self.df_test["Sv_fitting"] = Sv_fitting_test
        self.df_test["pressure_exp"] = pressure_exp_test
        self.df_test["Sv_exp"] = Sv_exp_test



    def _dump(self,force=False):
        if self.cache and force:
            time = datetime.datetime.now()
            filename_train_bakup = os.path.join(self.data_pickle_path, f"train_backup_{time.date()}-{time.hour}-{time.minute}-{time.second}.pkl")
            filename_test_bakup = os.path.join(self.data_pickle_path, f"train_backup_{time.date()}-{time.hour}-{time.minute}-{time.second}.pkl")
            if os.path.isfile(self.picklefile_train):
                print("Backing up current train pickle file to {}".format(filename_train_bakup))
                os.rename(self.picklefile_train,filename_train_bakup)
            self.df_train.to_pickle(self.picklefile_train)
            print("Saved a local data cache at {}".format(self.picklefile_train))
            if os.path.isfile(self.picklefile_test):
                print("Backing up current test pickle file to {}".format(filename_test_bakup))
                os.rename(self.picklefile_test,filename_test_bakup)
            self.df_test.to_pickle(self.picklefile_test)
            print("Saved a local data cache at {}".format(self.picklefile_test))
        elif self.cache and not force:
            if os.path.isfile(self.picklefile_train):
                print("Pickle file for training dataset exist. To overwrite it, set force to True.")
            else:
                self.df_train.to_pickle(self.picklefile_train)
                print("Saved a local data cache at {}".format(self.picklefile_train))
            if os.path.isfile(self.picklefile_test):
                print("Pickle file for training dataset exist. To overwrite it, set force to True.")
            else:
                self.df_test.to_pickle(self.picklefile_test)
                print("Saved a local data cache at {}".format(self.picklefile_test))
        else:
            print("dump function not saving files, please check if you have selected the correct parameters for this method.")
        








        

        
        






        

        




            
