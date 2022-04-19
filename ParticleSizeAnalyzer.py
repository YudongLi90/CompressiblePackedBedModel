
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import binarize
from tqdm import tqdm
from ColumnPlots import ColumnPlots

from MaterialAttributePCA import MaterialAttributePCA
from ParticleAttributePlots import ParticleAttributePlots


class ParticleSizeAnalyzer:
    """
    Analyzer particle size information from PartAn 3D
    """
    def __init__(self,path_to_data='./particlesize/AnalysisResultsRaw/',samplenumberfile='./particlesize/SampleNumbers.xlsx'):
        self.datapath = path_to_data
        self.samples = pd.read_excel(samplenumberfile)
        self._load_data()
        print('Setting up PCA model.')
        self.data_complete = self.data
        self._get_attribute_PCA()
        self._get_principle_components()
        self.data = self.data_clean_with_PCs_lst
        
        
    def _load_data(self):
        df = []
        for i in tqdm(self.samples.index,desc='loading particle data: '):
            
            dfS = pd.read_pickle(self.datapath+self.samples['Sample#'][i]+'.pickle').drop(0).drop(['Filter0', 'Filter1', 
                                                                                    'Filter2', 'Filter3', 
                                                                                    'Filter4','Filter5', 
                                                                                    'Filter6'], axis=1)
            df.append(dfS)
        self.data = df
    
    @staticmethod
    def filter_particle_data(data:pd.DataFrame) -> pd.DataFrame:
        """
        remove some of the information from particle data
        """
        data_clean = data.drop(["Id",'Img Id','hash','L/T Ratio', 'T/L Aspect Ratio','Ellipse Ratio','L/W Ratio', 'W/L Ratio', 'W/T Ratio', 'T/W Ratio'],axis=1)
        data_clean = data_clean.drop(['Sieve','Perimeter','Extent','Circularity'],axis=1)
        return data_clean
    
    def _filter_all_particle_data (self):
        """
        filter for all particle data
        """
        df_all=pd.DataFrame()
        data_clean_lst = []
        for data in self.data:
            data_clean = ParticleSizeAnalyzer.filter_particle_data(data)
            data_clean_lst.append(data_clean)
            df_all = pd.concat([df_all,data_clean])
        self.data_clean = df_all
        self.data_clean_lst = data_clean_lst

    def _get_attribute_PCA(self,n_component=2):
        """
        instantiate material attribute PCA model,
        fit for all data after filter
        """
        self._filter_all_particle_data()
        maPCA = MaterialAttributePCA(n_component=n_component)
        maPCA.get_model(self.data_clean)
        self.maPCA = maPCA
    
    def _get_principle_components(self):
        cleandata_with_PCs_lst = []
        for data in tqdm(self.data_clean_lst,desc="PCA: "):
            df_data = data
            df_PCs = self.maPCA.transform_data(data)
            n = df_PCs.shape[1]
            for i in range(n):
                name = f'PC_{i+1}'
                df_data[name] = df_PCs[:,i]
            cleandata_with_PCs_lst.append(df_data)
        self.data_clean_with_PCs_lst = cleandata_with_PCs_lst


    def get_particle_by_name(self,name:list):
        """
        return particle size information by experiment name
        name: should idealy be a list (one element list)
        """
        idx = self.get_samplelist_index(name,byExpName=True)
        if len(idx)>1:
            raise ValueError("Please only use this function to get particle information for one feedstock")
        else:           
            return self.data[idx[0]]

    def before_plot_format(self):
        labelsize = 18
        plt.rc('font', family='serif', serif='Times New Roman')
        plt.rc('text', usetex=False)
        plt.rc('xtick', labelsize=labelsize)
        plt.rc('ytick', labelsize=labelsize)
        plt.rc('axes', labelsize=labelsize)

    def plot_vol_cumuDist(self,df,attributes_list):
        """
        Plot volume based cumulative size distributions
        Parameters
        ----------
        df : data from particle analysis
        attributes_list : list of strings (the characteristics of the particle)      
        """
        self.before_plot_format()
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1)
        
        for i in range(len(attributes_list)):
            attribute = attributes_list[i]
            sorted_data = self.data.astype(float).sort_values(by=attribute)
            data_vol_cumu = sorted_data['Volume'].cumsum()
            plt.plot(sorted_data[attribute],data_vol_cumu/max(data_vol_cumu),label=attribute)
        
        plt.legend(loc='best')
    
    def get_samplelist_index(self,sample_list,byExpName=False):
        idx = []
        df = self.samples
        if not isinstance(sample_list,list):
            sample_list = [sample_list]
        for name in sample_list:
            if byExpName:
                if name in df['ExpName'].to_list():
                    #idx.append(..............
                    idx.append(df.index[self.samples['ExpName']==name].tolist()[0])
                else:
                    raise ValueError("Please provice correct Experiment name. Otherwise, set byExpName=False.")
            else:
                if name in df['Sample#'].to_list():
                    idx.append(df.index[self.samples['Sample#']==name].tolist()[0])
                else:
                    raise ValueError("Please provide correct Sample #. Otherwise, use Experiment name and set byExpName=True.")
        return idx


    def compare_plot_vol_cumudist(self,sample_list,attribute,xlabel,ylabel,xlimit=[0,None],ylimit=[0,100],byExpName=False,save=False,savename=None):
        """
        compare same attribute from different samples

        Parameters
        ----------
        samplelist: sample name list: e.g. ['S1','S5','S6']
        attribute: e.g. 'FLength'
        """
        plot = ParticleAttributePlots(attribute)
        index = self.get_samplelist_index(sample_list,byExpName)

        data_lst = []
        data_vol_cumu_lst = []
        label_lst = []
        for sample_index in index:
            #sample_index = self.samples.index[self.samples['Sample#']==samplelist[i]].tolist()[0]
            sorted_data = self.data[sample_index].astype(float).sort_values(by=attribute)
            data_vol_cumu = sorted_data['Volume'].cumsum()
            label=self.samples['ShortName'][sample_index]
            data_lst.append(sorted_data[attribute].to_numpy()/1000) # unit mm
            data_vol_cumu_lst.append(np.array(data_vol_cumu/max(data_vol_cumu))*100)
            label_lst.append(label)
            #plt.plot(sorted_data[attribute].to_numpy()/1000,np.array(data_vol_cumu/max(data_vol_cumu))*100,
            #        label=self.samples['ShortName'][sample_index],color='C'+str(i),linewidth = 3)
        #plt.grid(b=False)
        plot.compare_vol_cumudist_plot(data_lst,data_vol_cumu_lst,label_lst,xlabel=xlabel,ylabel=ylabel,xlimit=xlimit,ylimit=ylimit,
                                        save=save, savename=savename)
        #self.ax.set_xlabel("Size (mm)")
        #self.ax.set_ylabel("Dp cumulative distribution (volume %)")

    def plot_vol_dist(self,sample_list,attribute,bins=50,byExpName=False):
        """
        Plot volume based cumulative size distributions
        Parameters
        ----------
        df : data from particle analysis
        attributes_list : list of strings (the characteristics of the particle)      
        """

        index = self.get_samplelist_index(sample_list,byExpName)
        self.before_plot_format()
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1)
        
        for sample_index in index:
            #sample_index = self.samples.index[self.samples['Sample#']==samplelist[i]].tolist()[0]
            sorted_data = self.data[sample_index].astype(float).sort_values(by=attribute)
            attribute_data = sorted_data[attribute].astype(float).to_numpy()
            volume_data = sorted_data['Volume'].astype(float).to_numpy()
            data_vol_cumu_all = sum(volume_data)
            index = self.get_index_from_bins(attribute_data,bins)
            data_vol_cumu = []
            label_x = []
            for i in range(len(index)-1):
                start = index[i]
                end = index[i+1]
                data_vol_cumu.append(sum(volume_data[start:end]))
                #print(data_vol_cumu)
                label_x.append(attribute_data[end])
                #print(f"start: {start}, end: {end}; label: {sorted_data[attribute].to_numpy()[start]}")
            data_vol_cumu = np.array(data_vol_cumu)
            label_x = np.array(label_x)

            #print(f"length of data : {data_vol_cumu/data_vol_cumu_all}")

#            plt.plot(label_x,data_vol_cumu/data_vol_cumu_all,
#                    label=self.samples['Sample'][sample_index])
#            plt.scatter(label_x,data_vol_cumu/data_vol_cumu_all)
            plt.bar(label_x,data_vol_cumu/data_vol_cumu_all,width=max(label_x)/bins)
            
        
        #plt.legend(loc='best')    
        plt.show()

    def get_index_from_bins(self,attribute_data:np.ndarray,bins:float):
        """
        find indexes of bins from sorted attribute data
        """
        data = attribute_data
        low = min(data)
        high = max(data)
        pos = np.linspace(low,high,bins)
        idx = []
        for p in pos[:-1]:
            found, i = ParticleSizeAnalyzer.get_index_from_value_in_sorted_list(data,p)
            if found:
                idx.append(i)
            else:
                raise ValueError('Please check the array provided!')
        idx.append(len(data)-1)
        index = np.array(idx)
        return index


    @staticmethod
    def get_index_from_value_in_sorted_list(sorted_array:np.ndarray, value:float):
        """
        sorted_array is sorted in the order of low -> high
        """
        found = False
        idx = None
        for i, v in enumerate(sorted_array):
            if v > value:
                idx = i
                found = True
                return found, idx   
        return found,idx 


    def get_Dv_values(self,sample_list,attribute,byExpName=False):
        """
        calculate Dv values for the attribute of interest for samples in sample_list
        For example, if the attribute is set to particle diameter Dp:
        Dv(10) means that 10% of the particle (volume basis) is less than or equal to this number.
        """
        index = self.get_samplelist_index(sample_list,byExpName)
        intersted_Dv_values = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        result = []
        

        for sample_index in index: #loop through each sample of interest
            sorted_data = self.data[sample_index].astype(float).sort_values(by=attribute) #sort data according to attribute of interest
            attribute_data = sorted_data[attribute].astype(float).to_numpy()
            volume_data = sorted_data['Volume'].astype(float).to_numpy()
            volume_cumsum = volume_data.cumsum()
            data_vol_sum = sum(volume_data)
            volume_pct_cumsum = volume_cumsum/data_vol_sum # percentage cumsum distribution for volume
            #cumsum list is low->high (monotonically increasing)
            Dv_result = []
            for DvValue in intersted_Dv_values:
                idx = ParticleSizeAnalyzer.get_index_from_value_in_sorted_list(volume_pct_cumsum,DvValue)
                Dv_result.append(attribute_data[idx].item())
            result.append(dict(sample_index=sample_index,exp_name=self.samples["ExpName"][sample_index],
                                Dv_values=intersted_Dv_values,Dv_result=Dv_result))
        return result
            





