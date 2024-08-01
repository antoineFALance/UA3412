import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import models
from keras.layers import Dense,LSTM,Conv1D
from keras.layers import Dropout
import sys
import re
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
is_windows = hasattr(sys, 'getwindowsversion')
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from itertools import combinations,chain,product
from csv import writer

DNN_HYPER_PARAMETERS={"layers":[3,4],"units":[32,64,128],"drop_out_rate":[0.2]}
DNNhyper = [DNN_HYPER_PARAMETERS['layers'], DNN_HYPER_PARAMETERS['units'], DNN_HYPER_PARAMETERS['drop_out_rate']]
hyperCombinationDNN = list(product(*DNNhyper))

CNN_HYPER_PARAMETERS={"layers":[2,3],"filters":[16,32],"kernel_size":[2,3]}
CNNhyper = [CNN_HYPER_PARAMETERS['layers'], CNN_HYPER_PARAMETERS['filters'],CNN_HYPER_PARAMETERS['kernel_size']]
hyperCombinationCNN = list(product(*CNNhyper))

LSTM_HYPER_PARAMETERS=[10,20,50]

WINDOW_RANGE_PARAMETERS=[2]

def save_figure(ax,filename,directory):
    fig = ax.get_figure()
    fig.savefig(directory + filename + ".png")

def windowing(dataset,windowRange):
    x = np.array([list(dataset.to_numpy()[idx:idx+windowRange,:][:-1,].flatten()) for idx in range(dataset.shape[0]-windowRange)])
    y=   np.array([list(dataset.to_numpy()[idx:idx+windowRange,:][-1,-1].flatten()) for idx in range(dataset.shape[0]-windowRange)])
    return x,y

def build_LSTM_model(nbLSTcell,input_shape_1):
    model = models.Sequential()
    model.add(LSTM(nbLSTcell, input_shape=(input_shape_1, 1)))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

def build_DNN_model(layers=3, units=512, dropout_rate=0.2):
    model = models.Sequential()
    for _ in range(layers):
        model.add(Dense(units=units, activation='relu'))
        model.add(Dropout(rate=dropout_rate))
    model.add(Dense(units=1))
    return model

def y_hat_inv(scaler,y_array):
    return pd.DataFrame(data=scaler.inverse_transform(y_array), columns=['pred', 'reel'])

if is_windows:
    PATH_TO_INPUTS_DIR=os.path.dirname(os.path.dirname(os.getcwd()))+"\\data_\\main_dataset\\"
    PATH_TO_OUTPUT_MODEL=os.path.dirname(os.path.dirname(os.getcwd()))+"\\Models\\LSTM\\models\\"
    PATH_TO_OUTPUT_RESULTS = os.path.dirname(os.path.dirname(os.getcwd()))+"\\Models\\RNN\\results\\"
else:
    PATH_TO_INPUTS_DIR = os.path.dirname(os.path.dirname(os.getcwd())) + "/data_/main_dataset/"
    PATH_TO_OUTPUT_MODEL = os.path.dirname(os.path.dirname(os.getcwd())) + "/Models/LSTM/models/"
    PATH_TO_OUTPUT_RESULTS = os.path.dirname(os.path.dirname(os.getcwd())) + "/Models/RNN/results/"

df_results=pd.DataFrame()
for file in os.listdir(PATH_TO_INPUTS_DIR):
    filename = os.fsdecode(file)
    fullFileName=PATH_TO_INPUTS_DIR+'/'+filename
    home_id = re.search('weather_(.*).csv', filename).group(1)
    df_dataset=pd.read_csv(fullFileName,sep=";")
    df_dataset['gas_value'].fillna(0,inplace=True)
    df_dataset['Text'].interpolate(inplace=True)
    df_dataset['Tint'].interpolate(inplace=True)
    features=['Text','gas_value','Tint']

    with open(PATH_TO_OUTPUT_RESULTS + 'results.csv', 'a') as f_object:
        writer_object = writer(f_object)
        for wdwr in WINDOW_RANGE_PARAMETERS:
            x,y=windowing(df_dataset[features],windowRange=wdwr)

            scaler_x = preprocessing.StandardScaler().fit(x)
            x_stand = scaler_x.transform(x)

            scaler_y = preprocessing.StandardScaler().fit(y)
            y_stand = scaler_y.transform(y)

            x_train_val, x_test, y_train_val, y_test = train_test_split(x_stand,y_stand,test_size=0.1,random_state=123)
            x_train,x_val,y_train,y_val=train_test_split(x_train_val,y_train_val,test_size=0.2,random_state=123)

            #LSTM model
            for cells_nb in LSTM_HYPER_PARAMETERS:
                print(home_id+"_"+str(cells_nb)+"_"+str(wdwr)+"LSTM")
                LSTM_model = build_LSTM_model(cells_nb,x_train.shape[1])
                history = LSTM_model.fit(x_train,y_train, epochs=10, batch_size=32, validation_data=(x_val,y_val), verbose=False,shuffle=False)
                y_hat = LSTM_model.predict(x_test)

                # invert scaling for forecast
                df_yhat=y_hat_inv(scaler_y,np.concatenate((y_hat,y_test), axis=1))
                ax=df_yhat.plot.scatter(x='pred',y="reel",marker='x',c='black')
                plt.plot(range(int(df_yhat['pred'].min()),int(df_yhat['pred'].max())),range(int(df_yhat['pred'].min()),int(df_yhat['pred'].max())),c='green')
                mse=mean_squared_error(df_yhat['pred'].to_list(),df_yhat['reel'].to_list())
                writer_object.writerow([home_id,'LSTM',cells_nb,wdwr,mse])
                save_figure(ax=ax,filename=home_id+"_"+str(cells_nb)+"LSTM.png",directory=PATH_TO_OUTPUT_RESULTS)
                plt.close()

            #DNN model
            for config in hyperCombinationDNN:
                layers, units, dropout_rate = config[0], config[1], config[2]
                print(home_id + "_"+str(wdwr) + '_'.join([str(cfg) for cfg in config]) + "DNN")
                DNN_model = build_DNN_model(layers=layers, units=units,dropout_rate=dropout_rate)
                learning_rate =0.01
                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                callbacks_ = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20),
                              tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss')]
                DNN_model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
                history = DNN_model.fit(x_train,y_train,epochs=1000,callbacks=callbacks_,validation_data=(x_val,y_val),verbose=False,batch_size=32)

                # Prediction model
                df_yhat = y_hat_inv(scaler_y, np.concatenate((DNN_model.predict(x_test), y_test), axis=1))
                mse = mean_squared_error(df_yhat['pred'].to_list(), df_yhat['reel'].to_list())
                writer_object.writerow([home_id,'DNN', '|'.join([str(cfg) for cfg in config]), wdwr, mse])
                ax = df_yhat.plot.scatter(x='pred', y="reel", marker='x', c='black')
                plt.plot(range(int(df_yhat['pred'].min()),int(df_yhat['pred'].max())),range(int(df_yhat['pred'].min()),int(df_yhat['pred'].max())),c='green')
                save_figure(ax=ax, filename=home_id + "_"+str(wdwr)+'_' + '_'.join([str(cfg) for cfg in config]) + "_DNN.png",directory=PATH_TO_OUTPUT_RESULTS)
                plt.close()



        f_object.close()

