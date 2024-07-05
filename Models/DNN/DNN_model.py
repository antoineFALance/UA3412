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
warnings.filterwarnings('ignore')

# PATH DEFINITION
if is_windows:
    PATH_TO_INPUTS_DIR=os.path.dirname(os.path.dirname(os.getcwd()))+"\\data_\\main_dataset\\"
    PATH_TO_OUTPUT_MODEL=os.path.dirname(os.path.dirname(os.getcwd()))+"\\Models\\DNN\\models\\"
    PATH_TO_OUTPUT_RESULTS = os.path.dirname(os.path.dirname(os.getcwd()))+"\\Models\\DNN\\results\\"
else:
    PATH_TO_INPUTS_DIR = os.path.dirname(os.path.dirname(os.getcwd())) + "/data_/main_dataset/"
    PATH_TO_OUTPUT_MODEL = os.path.dirname(os.path.dirname(os.getcwd())) + "/Models/DNN/models/"
    PATH_TO_OUTPUT_RESULTS = os.path.dirname(os.path.dirname(os.getcwd())) + "/Models/DNN/results/"

# FUNCTIONS
def save_figure(fig,path):
    fig.savefig(path)
    return True

def buildRegressionRatingModel(layers=3,units=512,dropout_rate=0.2):
    model = models.Sequential()
    for _ in range(layers):
        model.add(Dense(units=units, activation='relu'))
        model.add(Dropout(rate=dropout_rate))
    model.add(Dense(units=1))
    return model

def plotResults(y_pred,y_test_scaled,RESULT_PATH):

    df_results=pd.DataFrame(columns=['pred','label'])
    df_results['pred']=y_pred
    df_results['label']=y_test_scaled.flatten()

    plt.figure(figsize=(12, 5))
    plt.xlabel(RESULT_PATH)

    ax1 = df_results.pred.plot(color='blue', grid=True)
    ax2 = df_results.label.plot(color='red', grid=True)
    # plt.show()

    fig = ax2.get_figure()
    result_png_name =PATH_TO_OUTPUT_RESULTS+RESULT_PATH+".png"
    fig.savefig(result_png_name)
    plt.close()

    return df_results

def train_regression_model(model_name,result_name,data,learning_rate=1e-3,epochs=1000,batch_size=128,layers=3,units=64,dropout_rate=0.25):
    # Get the data.
    train_ds,valid_ds,x_test_scaled=data
    loss='mse'
    # Create model instance.

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    callbacks_ = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
                 tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss')
                 ]

    model = buildRegressionRatingModel(layers=layers,units=units,dropout_rate=dropout_rate)

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10)
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
    df_result=plotResults(y_pred=y_pred,y_test_scaled=y_test_scaled,RESULT_PATH=result_name)
    test_mse=mean_squared_error(y_test_scaled.flatten(),y_pred)
    # Print results.
    history = history.history

    # model.save(model_name)
    print(model_name)
    print('Validation mse: {acc}, loss: {loss}'.format(acc=history['val_mse'][-1],loss=history['val_loss'][-1]))
    return [result_name,model_name,len(history['val_mse']), history['val_loss'][-1],test_mse]

def windowing(dataset,windowRange):
    indexRange=list(dataset.index)
    valid_index_range=[]
    discontinuousIndex =[a+1!=b for a, b in zip(indexRange, indexRange[1:])]
    discontinuousIndex2=[index  for index in range(len(discontinuousIndex)) if discontinuousIndex[index]==True]
    if discontinuousIndex2:
        continuousData=[dataset.to_numpy()[0:discontinuousIndex2[0]+1,:]]+[dataset.to_numpy()[discontinuousIndex2[idx]+1:discontinuousIndex2[idx+1]+1,:] for idx in range(len(discontinuousIndex2)-1)]
    else:
        continuousData=[dataset.to_numpy()]
    x_ds,y_ds=[],[]
    for continuousArr in continuousData:
        if continuousArr.shape[0]>=windowRange:
            for step in range(continuousArr.shape[0]-windowRange):
                ds=continuousArr[0+step:windowRange+step,:]
                x_ds.append(np.hstack([ds[:-1,:-1].flatten(),ds[-2,-1]]))
                y_ds.append(ds[-1,-1])
    x=np.stack(x_ds)
    y=np.stack(y_ds).reshape(-1, 1)
    return x,y

df_results=pd.DataFrame()
HyperParameters={"layers":[3,4],"units":[128,256],"drop_out_rate":[0.2]}
features_col=['hum_value',
              # 'central_heat_flow_value',
              # 'central_heat_return_value',
              'gas_value',
              # 'radiator_input_value',
              # 'radiator_output_value',
              # 'Day sin',
              # 'Day cos',
              # 'Year sin',
              # 'Year cos',
              'Text',
              # 'dewPt',
              # 'rh',
              'pressure',
              'wind_x',
              # 'wind_y',
              'Tint']

feat_comb = [list(combinations(features_col[:-1], i)) for i in range(1, len(features_col[:-1]) + 1)]
flat_list = list(chain.from_iterable(feat_comb))
feat_comb = [list(feat)+['Tint'] for feat in flat_list if "Text" in feat and "gas_value" in feat]

historyRegressionList=[]
df_main_results=pd.DataFrame()
for file in os.listdir(PATH_TO_INPUTS_DIR):
    for features_combination in feat_comb:
        for window_range in range(2,3,1):
            df_temp = pd.DataFrame()
            df_temp['features_combination'] = np.array(["|".join(list(features_combination[:-1]))])
            df_temp['window_range'] = np.array([window_range])

            filename = os.fsdecode(file)
            fullFileName=PATH_TO_INPUTS_DIR+'/'+filename
            home_id = re.search('weather_(.*).csv', filename).group(1)
            df_temp['home_id'] = np.array([home_id])
            df_dataset=pd.read_csv(fullFileName,sep=";")
            df_dataset['gas_value'].fillna(df_dataset['gas_value'].min(),inplace=True)

            dataset=df_dataset[features_col]
            mask = [not state for state in pd.isnull(dataset).any(axis=1).to_list()]
            dataset_masked = dataset[mask]
            x,y=windowing(dataset_masked,windowRange=window_range)

            scaler_x = preprocessing.StandardScaler().fit(x)
            x_stand=scaler_x.transform(x)

            scaler_y = preprocessing.StandardScaler().fit(y)
            y_stand = scaler_y.transform(y)

            x_train_scaled,y_train_scaled=np.expand_dims(x_stand[:int(0.7*x_stand.shape[0])],axis=0),np.expand_dims(y_stand[:int(0.7*y_stand.shape[0])],axis=0)
            x_valid_scaled, y_valid_scaled = np.expand_dims(x_stand[int(0.7 * x_stand.shape[0]):int(0.7 * x_stand.shape[0])+int(0.2 * x.shape[0])],axis=0), np.expand_dims(y_stand[int(0.7 * x.shape[0]):int(0.7 * y_stand.shape[0])+int(0.2 * y_stand.shape[0])],axis=0)
            x_test_scaled, y_test_scaled = np.expand_dims(x_stand[int(0.7 * x_stand.shape[0]) + int(0.2 * x_stand.shape[0]):], axis=0), np.expand_dims(y_stand[int(0.7 * y_stand.shape[0]) + int(0.2 * y_stand.shape[0]):], axis=0)

            x_test, y_test = np.expand_dims(x[int(0.7 * x.shape[0])+int(0.2 * x.shape[0]):],axis=0), np.expand_dims(y[int(0.7 * y.shape[0])+int(0.2 * y.shape[0]):],axis=0)

            train_ds = tf.data.Dataset.from_tensor_slices((x_train_scaled, y_train_scaled)).batch(32).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            valid_ds = tf.data.Dataset.from_tensor_slices((x_valid_scaled, y_valid_scaled)).batch(32).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

            #CNN model
            hyper = [HyperParameters['layers'], HyperParameters['units'], HyperParameters['drop_out_rate']]
            hyperCombination = list(product(*hyper))
            for config in hyperCombination:
                layers, units, dropout_rate = config[0], config[1], config[2]
                RESULT_NAME=home_id + "_" + str(window_range) + "_" + '_'.join(list(features_combination[:-1]))+"_"+'_'.join([str(cfg) for cfg in config])
                MODEL_HASH_NAME=hashlib.md5(RESULT_NAME.encode('utf-8')).hexdigest()
                MODEL_PATH = PATH_TO_OUTPUT_MODEL +MODEL_HASH_NAME+'.h5'
                trainResults=train_regression_model(model_name=MODEL_PATH,result_name=RESULT_NAME,data=[train_ds,valid_ds,x_test_scaled], layers=layers, units=units,dropout_rate=dropout_rate)
                historyRegressionList.append(trainResults)
                print(RESULT_NAME)

            df_main_results=pd.DataFrame(historyRegressionList)
            df_main_results.columns=['config','model_path','epoch','valid_mse','test_mse']
            df_main_results.to_csv(PATH_TO_OUTPUT_RESULTS+"DNN_results.csv",sep=";",index=False)

