import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import models
from keras.layers import Dense,LSTM
from keras.layers import Dropout
import sys
import re
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
is_windows = hasattr(sys, 'getwindowsversion')
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from itertools import combinations,chain,product

DNN_HYPER_PARAMETERS={"layers":[3,4,5,6],"units":[64,128,256,512],"drop_out_rate":[0.2,0.25,0.3]}

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


if is_windows:
    PATH_TO_INPUTS_DIR=os.path.dirname(os.path.dirname(os.getcwd()))+"\\data_\\main_dataset\\"
    PATH_TO_OUTPUT_MODEL=os.path.dirname(os.path.dirname(os.getcwd()))+"\\Models\\LSTM\\models\\"
    PATH_TO_OUTPUT_RESULTS = os.path.dirname(os.path.dirname(os.getcwd()))+"\\Models\\LSTM\\results\\"
else:
    PATH_TO_INPUTS_DIR = os.path.dirname(os.path.dirname(os.getcwd())) + "/data_/main_dataset/"
    PATH_TO_OUTPUT_MODEL = os.path.dirname(os.path.dirname(os.getcwd())) + "/Models/LSTM/models/"
    PATH_TO_OUTPUT_RESULTS = os.path.dirname(os.path.dirname(os.getcwd())) + "/Models/LSTM/results/"

for file in os.listdir(PATH_TO_INPUTS_DIR):
    filename = os.fsdecode(file)
    fullFileName=PATH_TO_INPUTS_DIR+'/'+filename
    home_id = re.search('weather_(.*).csv', filename).group(1)
    df_dataset=pd.read_csv(fullFileName,sep=";")
    df_dataset['gas_value'].fillna(0,inplace=True)
    df_dataset['Text'].interpolate(inplace=True)
    df_dataset['Tint'].interpolate(inplace=True)
    features=['Text','gas_value','Tint']
    # print(df_dataset[features].isnull().values.any())

    x,y=windowing(df_dataset[features],windowRange=6)
    # x=df_dataset[features].to_numpy()
    # y=df_dataset['Tint'].to_numpy().reshape(-1,1)

    scaler_x = preprocessing.StandardScaler().fit(x)
    x_stand = scaler_x.transform(x)

    scaler_y = preprocessing.StandardScaler().fit(y)
    y_stand = scaler_y.transform(y)

    x_train_val, x_test, y_train_val, y_test = train_test_split(x_stand,y_stand,test_size=0.1,random_state=123)
    x_train,x_val,y_train,y_val=train_test_split(x_train_val,y_train_val,test_size=0.2,random_state=123)

    # LSTM model
    LSTM_model = build_LSTM_model(10,x_train.shape[1])
    history = LSTM_model.fit(x_train,y_train, epochs=10, batch_size=32, validation_data=(x_val,y_val), verbose=False,shuffle=False)
    y_hat = LSTM_model.predict(x_test)
    # invert scaling for forecast
    inv_yhat = np.concatenate((y_hat,y_test), axis=1)
    df_yhat = pd.DataFrame(data=scaler_y.inverse_transform(inv_yhat),columns=['pred','reel'])
    ax=df_yhat.plot.scatter(x='pred',y="reel",marker='x',c='black')
    mse=mean_squared_error(df_yhat['pred'].to_list(),df_yhat['reel'].to_list())
    plt.show()

    #DNN model
    hyper = [DNN_HYPER_PARAMETERS['layers'], DNN_HYPER_PARAMETERS['units'], DNN_HYPER_PARAMETERS['drop_out_rate']]
    hyperCombination = list(product(*hyper))
    for config in hyperCombination:
        layers, units, dropout_rate = config[0], config[1], config[2]
        DNN_model = build_DNN_model(layers=layers, units=units,dropout_rate=dropout_rate)
        learning_rate =0.001

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        callbacks_ = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20),
                      tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss')]
        DNN_model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
        history = DNN_model.fit(x_train,y_train,epochs=200,callbacks=callbacks_,validation_data=(x_val,y_val),verbose=0,batch_size=32)

