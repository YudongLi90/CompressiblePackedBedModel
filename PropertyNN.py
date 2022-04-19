from matplotlib import pyplot as plt
from numpy.random import seed
from tensorflow import random

import os
import numpy as np
import pandas as pd
from DataSetColumn import DataSetColumn
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.initializers import GlorotUniform


class PropertyNN:
    """
    a neural network to predict density based on material attribute
    """

    def __init__(self,name=None,exp_path='./exp/',localdbpath='./datapickles/',savepath='./property_output/',checkpoint_path='./checkpoints/'):
        seed(1)
        random.set_seed(2)
        self.name = name
        self.exp_path = exp_path
        self.localdbpath = localdbpath
        self.savepath = savepath
        self.checkpoint_path = checkpoint_path
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)

    def _prepare_dataset(self,property_str:str,property_exp_str:str,pressure_str="pressure_fitting",pressure_exp_str="pressure_exp"):
        dataset = DataSetColumn(exp_path=self.exp_path,localdbpath=self.localdbpath)
        self.dataset = dataset
        self.data_train, self.property_train = self._prepare_data_from_dataframe(dataset.df_train,property_str=property_str,
                                                                                                    pressure_str=pressure_str)
        self.data_test, self.property_test = self._prepare_data_from_dataframe(dataset.df_test,property_str=property_str,
                                                                                                    pressure_str=pressure_str)
        self.data_exp_train, self.property_exp_train = self._prepare_data_from_dataframe(dataset.df_train, property_str=property_exp_str,
                                                                                                    pressure_str=pressure_exp_str)
        self.data_exp_test, self.property_exp_test = self._prepare_data_from_dataframe(dataset.df_test, property_str=property_exp_str,
                                                                                                    pressure_str=pressure_exp_str)                                                                                            
        self.data_all = np.concatenate([self.data_train,self.data_test])
        self.property_all = np.concatenate([self.property_train,self.property_test])
        self.data_exp = np.concatenate([self.data_exp_train,self.data_exp_test]) # experiment data to be used for evaluation
        self.property_exp = np.concatenate([self.property_exp_train,self.property_exp_test]) # experiment result to be used for evaluation
        self._get_data_normalization_standardization()
        self._get_property_standardization() # this takes care of property standardization
        self.data_scaled_train = self.data_normalization_standardization(self.data_train)
        self.data_scaled_test = self.data_normalization_standardization(self.data_test)
        self.data_scaled_exp = self.data_normalization_standardization(self.data_exp)
        # include experiment data in trainning
        self.data_train_w_exp = np.concatenate([self.data_train,self.data_exp_train])
        self.property_train_w_exp = np.concatenate([self.property_train,self.property_exp_train])
        self.data_test_w_exp = np.concatenate([self.data_test,self.data_exp_test])
        self.property_test_w_exp = np.concatenate([self.property_test,self.property_exp_test])
        self.data_scaled_train_w_exp = self.data_normalization_standardization(self.data_train_w_exp)
        self.data_scaled_test_w_exp = self.data_normalization_standardization(self.data_test_w_exp)
        self.property_standardized_train_w_exp = self.property_standardization(self.property_train_w_exp)
        self.property_standardized_test_w_exp = self.property_standardization(self.property_test_w_exp)
    
    def _get_data_normalization_standardization(self):
        data_normalizer = MinMaxScaler()
        data_standardizer = StandardScaler()
        data_normalizer.fit(self.data_all)
        data_normalized = data_normalizer.transform(self.data_all)
        data_standardizer.fit(data_normalized)
        self.data_norm_std = data_standardizer.transform(data_normalized)
        self.data_normalizer = data_normalizer
        self.data_standardizer = data_standardizer

    def data_normalization_standardization(self,data:np.ndarray) -> np.ndarray:
        """
        outfacing function to transform data to be first normalized then standardized
        """
        normalized = self.data_normalizer.transform(data)
        norm_std = self.data_standardizer.transform(normalized)
        return norm_std

    def inverse_normalization_standardization(self,data:np.ndarray) -> np.ndarray:
        """
        outfacing function to do inverse standardization then normalization for training data
        """
        de_standardize = self.data_standardizer.inverse_transform(data)
        de_norm_std = self.data_normalizer.inverse_transform(de_standardize)
        return de_norm_std

    def _get_property_standardization(self):
        property_standardizer = StandardScaler()
        property_standardizer.fit(self.property_all.reshape(len(self.property_all),1))
        self.property_standardized = property_standardizer.transform(self.property_all.reshape(len(self.property_all),1))
        self.property_standardized_train = property_standardizer.transform(self.property_train.reshape(len(self.property_train),1))
        self.property_standardized_test = property_standardizer.transform(self.property_test.reshape(len(self.property_test),1))
        self.property_standardizer = property_standardizer
    
    def property_standardization(self,property:np.ndarray) -> np.ndarray:
        """
        outfacing function to standardize the ydata (property in this context)
        """
        return self.property_standardizer.transform(property.reshape(len(property),1))
    
    def inverse_property_standardization(self,data:np.ndarray) -> np.ndarray:
        """
        outfacing fuction to do inverse standardize transformation for the property value
        This should be used to inverse transform the NN prediction property result to comput R^2 or mse
        """
        return self.property_standardizer.inverse_transform(data)

    def _prepare_data_from_dataframe(self,df:pd.DataFrame,property_str:str, pressure_str:str):
        data_lst = []   
        y_lst = []
        for i,data in df.iterrows():
            PC1_Dv40 = data["PC_1_Dv"][3] # Dv40
            PC1_Dv80 = data["PC_1_Dv"][7] # Dv80
            PC2_Dv40 = data["PC_2_Dv"][3]
            PC2_Dv80 = data["PC_2_Dv"][7]
            pressure_fitting = data[pressure_str]
            property_fitting = data[property_str]
            for pressure,property in zip(pressure_fitting,property_fitting):
                data_lst.append([PC1_Dv40,PC1_Dv80,PC2_Dv40,PC2_Dv80,pressure])
                y_lst.append(property)
        data_np = np.array(data_lst)
        y_np = np.array(y_lst)
        return data_np, y_np

    def get_NN_model(self,input_shape=(5,),node_lst=[24,12,1],activation='tanh',use_batchnormalization=False,
                                                use_dropout=False,dropout_rate=0.1,optimizer_selection="SGD",
                                                                learning_rate=0.01,loss='mse',metrics=['mse']):
        initializer = GlorotUniform(seed = 3)
        model = Sequential()
        model.add(tf.keras.Input(shape=input_shape))
        # take care of all the node requested, except the last one (last layer should always have one node)
        for i,node in enumerate(node_lst[:-1]): 
            model.add(Dense(name=f'Dense_{i+1}',units=node,kernel_initializer=initializer,kernel_regularizer=regularizers.l1(),activation=activation))
            if use_batchnormalization:
                BatchNormalization()
            if use_dropout:
                model.add(Dropout(name=f'Dropout_{i+1}',rate=dropout_rate))
        model.add(Dense(name=f'Dense_Output',units=1,activation='linear'))
        if optimizer_selection == "SGD":
            model.compile(optimizer=SGD(learning_rate=learning_rate,momentum=0.9),loss=loss,metrics=metrics)
        elif optimizer_selection == "Adam":
            model.compile(optimizer=Adam(learning_rate=learning_rate),loss=loss,metrics=metrics)
        else:
            raise ValueError("Please choose optimizer from SGD or Adam")
        self.model = model

    def predict_and_evaluate(self,data,property_original):
        predicted = self.model.predict(data)
        predicted_reversed = self.inverse_property_standardization(predicted)
        property_prediction_mse = mse(property_original,predicted_reversed)
        property_prediction_Rsq = r2(property_original,predicted_reversed)
        return predicted, predicted_reversed,property_prediction_mse,property_prediction_Rsq
    
    def evaluate_NN_model(self,include_exp=True,batch_size=100,shuffle=True,epochs=400,verbose=0):
        learning_rate_reduction = ReduceLROnPlateau(factor=0.2,patience=20)
        checkpoint_best = ModelCheckpoint(filepath=self.checkpoint_path,
                                      monitor='val_loss',
                                      save_best_only= True,
                                      save_weights_only=True,
                                      save_freq='epoch',
                                      verbose=verbose)
        if include_exp:
            data_train = self.data_scaled_train_w_exp
            y_train = self.property_standardized_train_w_exp
            data_val = self.data_scaled_test_w_exp
            y_val = self.property_standardized_test_w_exp
        else:
            data_train = self.data_scaled_train
            data_val = self.data_scaled_test
            y_train = self.property_standardized_train
            y_val = self.property_standardized_test

        history = self.model.fit(data_train,y_train, batch_size=batch_size,shuffle=shuffle,
                                validation_data=(data_val,y_val),
                                callbacks=[learning_rate_reduction,checkpoint_best],epochs=epochs,verbose=verbose)
        self.model.load_weights(filepath=self.checkpoint_path)
        test_result = self.model.evaluate(self.data_scaled_test,self.property_standardized_test,verbose=verbose)
        #predicted = self.model.predict(self.data_scaled_test)
        #predicted_reversed = self.inverse_property_standardization(predicted)
        #property_prediction_mse = mse(self.property_test,predicted_reversed)
        #property_prediction_Rsq = r2(self.property_test,predicted_reversed)
        # keeping these information
        predicted, predicted_reversed,property_prediction_mse,property_prediction_Rsq = self.predict_and_evaluate(self.data_norm_std,
                                                                                                                    self.property_all)
        predicted_exp, predicted_exp_reversed,exp_mse,exp_Rsq = self.predict_and_evaluate(self.data_scaled_exp,self.property_exp)
        self.history = history
        self.predicted_reversed = predicted_reversed
        self.predicted_exp_reversed = predicted_exp_reversed
        return history,test_result,predicted,predicted_reversed,property_prediction_mse,property_prediction_Rsq,predicted_exp,predicted_exp_reversed,exp_mse,exp_Rsq

    def load_NN_and_predict(self,checkpoint_path=None):
        if checkpoint_path == None:
            checkpoint_path = self.checkpoint_path
        
        self.model.load_weights(filepath=checkpoint_path)
        predicted, predicted_reversed,property_prediction_mse,property_prediction_Rsq = self.predict_and_evaluate(self.data_norm_std,
                                                                                                                    self.property_all)
        predicted_exp, predicted_exp_reversed,exp_mse,exp_Rsq = self.predict_and_evaluate(self.data_scaled_exp,self.property_exp)
        self.predicted_reversed = predicted_reversed
        self.predicted_exp_reversed = predicted_exp_reversed
        self.plot_training_performance(plot_training=False)
        return predicted, predicted_reversed,property_prediction_mse,property_prediction_Rsq,predicted_exp, predicted_exp_reversed,exp_mse,exp_Rsq

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
        plt.scatter(self.property_exp,self.predicted_exp_reversed.reshape((len(self.predicted_exp_reversed),)))
        plt.plot(np.linspace(min(self.property_exp),max(self.property_exp),100), 
                np.linspace(min(self.property_exp),max(self.property_exp),100))


