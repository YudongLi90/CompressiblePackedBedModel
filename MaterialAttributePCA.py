from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

class MaterialAttributePCA:
    """
    perform Principle Component Analysis for material attribute
    Do this if you want to train/regress material attribute to models(linear/non-linear/NeuralNetworks,etc.)
    This should give you reduced dimensionality with most material information
    perform get_model on all data you have
    then for each datapoint, perform transfor_data to get principle component
    """
    def __init__(self,n_component=2):
        self.sc = StandardScaler()
        self.pca = PCA(n_components=n_component)
        self.pca_all = PCA()

    def prepare_airclass_feedstocks(self):
        pass

    def get_model(self,df_alldata:pd.DataFrame):
        self.dataset_all = df_alldata
        self.feature_names = df_alldata.columns
        df = df_alldata
        self.sc_data = self.sc.fit_transform(df)
        self.model = self.pca.fit(self.sc_data)
        self.model_allComponent = self.pca_all.fit(self.sc_data)
    
    def transform_data(self,df_singledata:pd.DataFrame):
        df = df_singledata
        return self.model.transform(df)
    
    def get_most_important_features(self,print_to_screen=False):
        #number of components
        n_pcs = self.model.components_.shape[0]
        #index of most important feature on EACH component
        most_important = [np.abs(self.model.components_[i]).argmax() for i in range (n_pcs)]
        most_important_names = [self.feature_names[most_important[i]] for i in range(n_pcs)]
        dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}
        df = pd.DataFrame(dic.items())
        if print_to_screen:
            print(df)
        return df

    def bar_plot_format(self):
        labelsize = 18
        sns.set(style="whitegrid", color_codes=True)
        plt.rc('font', family='serif', serif='Times New Roman')
        plt.rc('text', usetex=False)
        plt.rc('xtick', labelsize=labelsize)
        plt.rc('ytick', labelsize=labelsize)
        plt.rc('axes', labelsize=labelsize)
    
    def after_plot_format(self):
        for axis in ['top','bottom','left','right']:
            self.ax.spines[axis].set_linewidth(1.8)
        self.fig.tight_layout()
    
    def plot_component(self,save=False,savename=None):
        self.bar_plot_format()
        ydata = self.model_allComponent.explained_variance_ratio_
        rank = ydata.argsort().argsort()
        pal = sns.color_palette("Greens_d", len(ydata))
        self.fig = plt.figure(figsize=(12, 8))
        #plt.bar(range(1,len(self.model_allComponent.explained_variance_ratio_)+1),self.model_allComponent.explained_variance_ratio_)
        #plt.xlabel('number of components')
        xticklabels = [f"PC-{i+1}" for i in range(len(ydata))]
        self.ax = sns.barplot(x=xticklabels,y=ydata*100, palette=np.array(pal)[rank])
        self.ax.set_xticklabels(self.ax.get_xticklabels(), rotation=40, ha="right")
        self.ax.set_ylabel("Contribution (%)")
        self.after_plot_format()
        if save and savename:
            self.fig.savefig(savename+'.jpg',dpi=300,bbox_inches='tight')


    def plot_contribution_to_component(self,n,save=False,savename=None):
        self.bar_plot_format()
        ydata = np.abs(self.model.components_[n])
        rank = ydata.argsort().argsort()
        pal = sns.color_palette("Greens_d", len(ydata))
        self.fig = plt.figure(figsize=(12, 6))
        #plt.bar(range(len(np.abs(self.model.components_[n]))),np.abs(self.model.components_[n]))

        self.ax = sns.barplot(x=self.feature_names,y=ydata*100, palette=np.array(pal)[rank])
        self.ax.set_xticklabels(self.ax.get_xticklabels(), rotation=40, ha="right")
        self.ax.set_ylabel("Contribution (%)")
        self.after_plot_format()
        if save and savename:
            self.fig.savefig(savename+'.jpg',dpi=300,bbox_inches='tight')

    def plot_component_feature_relation(self):
        self.bar_plot_format()
        plt.figure()
        newdata = self.model.fit_transform(self.dataset_all)

        def myplot(score,coeff,labels=None):
            xs = score[:,0]
            ys = score[:,1]
            n = coeff.shape[0]
            scalex = 1.0/(xs.max() - xs.min())
            scaley = 1.0/(ys.max() - ys.min())
            plt.scatter(xs * scalex,ys * scaley)
            for i in range(n):
                plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
                if labels is None:
                    plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
                else:
                    plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.xlabel("PC{}".format(1))
        plt.ylabel("PC{}".format(2))
        plt.grid()

        #Call the function. Use only the 2 PCs.
        myplot(newdata[:,0:2],np.transpose(self.model.components_[0:2, :]))
        plt.show()

