import os
import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import re
is_windows = hasattr(sys, 'getwindowsversion')
from sklearn import preprocessing
from matplotlib import pyplot as plt


def windowing(dataset,windowRange=2):
    indexRange=list(dataset.index)
    valid_index_range=[]
    discontinuousIndex =[a+1!=b for a, b in zip(indexRange, indexRange[1:])]
    discontinuousIndex2=[index  for index in range(len(discontinuousIndex)) if discontinuousIndex[index]==True]
    continuousData=[dataset.to_numpy()[0:discontinuousIndex2[0]+1,:]]+[dataset.to_numpy()[discontinuousIndex2[idx]+1:discontinuousIndex2[idx+1]+1,:] for idx in range(len(discontinuousIndex2)-1)]
    x_ds,y_ds=[],[]
    for continuousArr in continuousData:
        if continuousArr.shape[0]>=windowRange:
            for step in range(continuousArr.shape[0]-windowRange):
                ds=continuousArr[0+step:windowRange+step,:]
                x_ds.append(ds[:-1,:-1].flatten())
                y_ds.append(ds[-1,-1])
    x=np.stack(x_ds)
    y=np.stack(y_ds).reshape(-1, 1)
    return x,y

if is_windows:
    PATH_TO_INPUTS_DIR=os.path.dirname(os.path.dirname(os.getcwd()))+"\\data_\\main_dataset\\"
    PATH_TO_OUTPUT_MODEL=os.path.dirname(os.path.dirname(os.getcwd()))+"\\Models\\CNN\\models\\"
    PATH_TO_OUTPUT_RESULTS = os.path.dirname(os.path.dirname(os.getcwd()))+"\\Models\\CNN\\results\\"
else:
    PATH_TO_INPUTS_DIR = os.path.dirname(os.path.dirname(os.getcwd())) + "/data_/main_dataset/"
    PATH_TO_OUTPUT_MODEL = os.path.dirname(os.path.dirname(os.getcwd())) + "/Models/CNN/models/"
    PATH_TO_OUTPUT_RESULTS = os.path.dirname(os.path.dirname(os.getcwd())) + "/Models/CNN/results/"

for file in os.listdir(PATH_TO_INPUTS_DIR):
    filename = os.fsdecode(file)
    fullFileName=PATH_TO_INPUTS_DIR+'/'+filename
    home_id = re.search('weather_(.*).csv', filename).group(1)
    df_dataset=pd.read_csv(fullFileName,sep=";")
    df_dataset['gas_value'].fillna(df_dataset['gas_value'].min(),inplace=True)
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
    dataset=df_dataset[features_col]
    mask = [not state for state in pd.isnull(dataset).any(axis=1).to_list()]
    dataset_masked = dataset[mask]
    x,y=windowing(dataset_masked)


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
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=256, kernel_size=4, padding="causal", activation="relu"),
        tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding="causal", activation="relu"),
        tf.keras.layers.Conv1D(filters=64, kernel_size=2, padding="causal", activation="relu"),
        tf.keras.layers.Conv1D(filters=32, kernel_size=2, padding="causal", activation="relu"),
        tf.keras.layers.Dense(1)])

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse')
    history=model.fit(train_ds, validation_data=valid_ds, epochs=2000, callbacks=[early_stopping_cb])
    MODEL_NAME = home_id+"_"+str(int(history.history['val_loss'][-1]))+"_CNN"
    model.save(PATH_TO_OUTPUT_MODEL + MODEL_NAME)

    y_pred = model.predict(x_test_scaled).flatten()

    df_results=pd.DataFrame(columns=['pred','label'])
    df_results['pred']=y_pred
    df_results['label']=y_test_scaled.flatten()

    plt.figure(figsize=(12, 5))
    plt.xlabel('Number of requests every 10 minutes')

    ax1 = df_results.pred.plot(color='blue', grid=True)
    ax2 = df_results.label.plot(color='red', grid=True)
    # plt.show()

    fig = ax2.get_figure()
    result_png_name = PATH_TO_OUTPUT_RESULTS + home_id + "_result_plot.png"
    fig.savefig(result_png_name)
    plt.close()


    print('ok')