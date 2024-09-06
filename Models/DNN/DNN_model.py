import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import models
from keras.layers import Dense
from keras.layers import Dropout
import sys
import re
is_windows = hasattr(sys, 'getwindowsversion')
from sklearn import preprocessing
from matplotlib import pyplot as plt
from itertools import combinations,chain,product
import hashlib
import warnings
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
warnings.filterwarnings('ignore')
home_id='home_id_106'
# PATH DEFINITION
if is_windows:
    PATH_TO_INPUTS_DIR='C:\\Users\\a.lance\\PycharmProjects\\UA3412_\\IDEAL_home106\\home106_main_data_set_HR.csv'
    PATH_TO_OUTPUT_MODEL=os.path.dirname(os.path.dirname(os.getcwd()))+"\\Models\\DNN\\models\\"
    PATH_TO_OUTPUT_RESULTS = os.path.dirname(os.path.dirname(os.getcwd()))+"\\Models\\DNN\\results\\"
else:
    PATH_TO_INPUTS_DIR = 'C:\\Users\\a.lance\\PycharmProjects\\UA3412_\\IDEAL_home106\\home106_main_data_set_HR.csv'
    PATH_TO_OUTPUT_MODEL = os.path.dirname(os.path.dirname(os.getcwd())) + "/Models/DNN/models/"
    PATH_TO_OUTPUT_RESULTS = os.path.dirname(os.path.dirname(os.getcwd())) + "/Models/DNN/results/"

# FUNCTIONS
def save_figure(ax,filename,directory):
    fig = ax[0].get_figure()
    fig.savefig(directory + filename + ".png")

def save_model(model,filename,directory):
    model.save(directory+filename)

def buildRegressionRatingModel(layers=3,units=512,dropout_rate=0.2):
    model = models.Sequential()
    for _ in range(layers):
        model.add(Dense(units=units, activation='relu'))
        model.add(Dropout(rate=dropout_rate))
    model.add(Dense(units=1))
    return model


def train_regression_model(home_id,data,learning_rate=1e-2,epochs=2000,batch_size=128,layers=3,units=64,dropout_rate=0.25):
    # Get the data.
    train_ds,valid_ds,x_test_scaled=data
    loss='mse'
    # Create model instance.

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    callbacks_ = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50),
                 tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss')
                 ]

    model = buildRegressionRatingModel(layers=layers,units=units,dropout_rate=dropout_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['mse'])

    # Train  model.
    history = model.fit(
            train_ds,
            epochs=epochs,
            callbacks=callbacks_,
            validation_data=valid_ds,
            verbose=0,  # Logs once per epoch.
            batch_size=batch_size)

    # Prediction model
    y_pred=model.predict(x_test_scaled).flatten()
    test_mse=mean_squared_error(y_test.flatten(),y_pred)
    # Print results.
    history = history.history
    # plot results

    ax=plt.plot(y_test.flatten(), y_pred, c='black', marker='x', linestyle="")
    plt.plot(np.array(range(0,5,1)), np.array(range(0,5,1)), c='green', linestyle="-")
    plt.title("predictions / valeurs reel")
    plt.xlabel("y_pred")
    plt.ylabel("y_reel")
    plt.grid()
    plt.gca().set_aspect("equal")
    save_figure(ax=ax,filename=home_id+'_'+str(layers)+'_'+str(units)+'_'+str(dropout_rate)+'.png',directory=PATH_TO_OUTPUT_RESULTS)
    plt.close()
    save_model(model=model,filename=home_id+'_'+str(layers)+'_'+str(units)+'_'+str(dropout_rate)+'.h5',directory=PATH_TO_OUTPUT_MODEL)
    print('Validation mse: {acc}, loss: {loss}'.format(acc=history['val_mse'][-1],loss=history['val_loss'][-1]))
    return [len(history['val_mse']), history['val_loss'][-1],test_mse]

df_results=pd.DataFrame()
HyperParameters={"layers":[2,3,4,5],"units":[64,128,256,512],"drop_out_rate":[0.02,0.05,0.1]}

# FEATURES
features_col=['ti_1','te_1','phi_1','Tint']

historyRegressionList=[]
df_main_results=pd.DataFrame()

df_dataset=pd.read_csv(PATH_TO_INPUTS_DIR,sep=";")
scaler= StandardScaler().fit(df_dataset)
joblib.dump(scaler,'DNN_scaler.gz')
df_dataset_scaled=pd.DataFrame(data=scaler.transform(df_dataset),columns=df_dataset.columns)

df_dataset_scaled['Tint1']=df_dataset_scaled['Tint'].shift(-1)

df_DNN = df_dataset_scaled[['Tint', 'Text', 'gas_value', 'Tint1']].dropna()


x = df_DNN[['Tint', 'Text', 'gas_value']].to_numpy()
y = df_DNN[['Tint1']].to_numpy()

x_train_val, x_test, y_train_val, y_test = train_test_split(x,y,test_size=0.1,random_state=123)
x_train,x_val,y_train,y_val=train_test_split(x_train_val,y_train_val,test_size=0.2,random_state=123)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
valid_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(32).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

#DNN model
hyper = [HyperParameters['layers'], HyperParameters['units'], HyperParameters['drop_out_rate']]
hyperCombination = list(product(*hyper))
for config in hyperCombination:
    layers, units, dropout_rate = config[0], config[1], config[2]
    print(home_id + '_' + str(layers) + '_'+str(units)+'_'+str(dropout_rate))
    # RESULT_NAME=home_id + "_" + str(window_range) + "_" + '_'.join(list(features_combination[:-1]))+"_"+'_'.join([str(cfg) for cfg in config])
    # MODEL_HASH_NAME=hashlib.md5(RESULT_NAME.encode('utf-8')).hexdigest()
    # MODEL_PATH = PATH_TO_OUTPUT_MODEL +MODEL_HASH_NAME+'.h5'
    trainResults=train_regression_model(home_id=home_id,data=[train_ds,valid_ds,x_test], layers=layers, units=units,dropout_rate=dropout_rate)
    historyRegressionList.append([home_id]+trainResults+list(config))


df_main_results=pd.DataFrame(historyRegressionList)
df_main_results.columns=['home_id','epoch','valid_mse','test_mse','layers','units','dropout']
df_main_results.to_csv(PATH_TO_OUTPUT_RESULTS+"DNN_results.csv",sep=";",index=False)

