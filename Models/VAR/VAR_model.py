import math

import pandas as pd
import os
import numpy as np
import re
from matplotlib import pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
import sys
is_windows = hasattr(sys, 'getwindowsversion')
import scipy.stats as stats
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.stattools import adfuller,ccf
from sklearn.preprocessing import StandardScaler
from scipy import signal
import warnings
warnings.filterwarnings("ignore")
from csv import writer
from sklearn.metrics import mean_squared_error,r2_score
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def save_figure(ax,filename,directory):
    fig = ax.get_figure()
    fig.savefig(directory + filename + ".png")

def window_range(data,wdw):
    return [data[idx:idx+wdw,:] for idx in range(data.shape[0]-wdw)]


# PATH DEFINITION
if is_windows:
    PATH_TO_INPUT_DIR_DATA= "C:\\Users\\a.lance\\PycharmProjects\\UA3412_\\IDEAL_home106\\home106_main_data_set_HR.csv"
    PATH_TO_INPUT_DIR_MODEL=os.path.dirname(os.path.dirname(os.getcwd()))+"\\Models\\VAR\\results\\"
    PATH_TO_OUTPUT_RESULT = os.path.dirname(os.path.dirname(os.getcwd()))+"\\Models\\VAR\\results\\"

else:
    PATH_TO_INPUT_DIR_DATA= os.path.dirname(os.path.dirname(os.getcwd()))+"/data_/dataset_gas_value_corrected/"
    PATH_TO_INPUT_DIR_MODEL=os.path.dirname(os.path.dirname(os.getcwd()))+"/Models/VAR/results/"
    PATH_TO_OUTPUT_RESULT = os.path.dirname(os.path.dirname(os.getcwd())) + "/Models/VAR/results/"


with open(PATH_TO_OUTPUT_RESULT + 'results.csv', 'a') as f_object:
    writer_object = writer(f_object)
    diffFeg = 0
    home_id='home_106'

    df_temp=pd.DataFrame()
    df_dataset = pd.read_csv(PATH_TO_INPUT_DIR_DATA, sep=";")
    df_dataset['datetime']=pd.to_datetime(df_dataset[['year', 'month', 'day', 'hour']])
    df_dataset['yearmonth']=df_dataset['year']*100+df_dataset['month']

    # FILTRE SUR MOIS HIVER
    # df_dataset=df_dataset[(df_dataset['yearmonth'] ==201711) | (df_dataset['yearmonth'] ==201712) | (df_dataset['yearmonth'] ==201801)| (df_dataset['yearmonth'] ==201802)]


    df_dataset.index = pd.DatetimeIndex(df_dataset['datetime'])
    VAR_data=df_dataset[['Tint','Text','gas_value_corrected']]
    VAR_data.fillna(method='ffill',inplace=True)
    VAR_data.dropna(inplace=True)


    # Johansen cross-integration test
    # try:
    scaler = StandardScaler()
    # VAR_data[VAR_data.columns] = scaler.fit_transform(VAR_data)
    ad_Fuller_p_values=[adfuller(VAR_data[col])[1] for col in VAR_data.columns ]

    johansen_test = coint_johansen(VAR_data,-1,1).eig

    if all([abs(p_values)<0.05 for p_values in ad_Fuller_p_values]):

        # creating the train and validation set
        train = VAR_data[:int(0.8 * (len(VAR_data)))]
        valid = VAR_data[int(0.8 * (len(VAR_data))):].to_numpy()

        # valid = VAR_data[int(0.8 * (len(VAR_data))):].to_csv('valid.csv')

        # Construction du modèle
        model = VAR(train)
        optimal_lags = model.select_order()
        lag_order = optimal_lags.selected_orders['bic']
        lag_order=1
        # print('home_id: '+str(home_id)+'-BIC: '+str(lag_order))
        data_test = window_range(valid,lag_order)
        results = model.fit(lag_order)
        print(results.summary())
        var_model = results.model

        # Vérification du modèle
        Tint_hat =[scaler.inverse_transform(results.forecast(x_test,1)).flatten()[0] for x_test in data_test]
        Tint_reel = list(scaler.inverse_transform(valid)[1:,0])
        mse = mean_squared_error(Tint_reel,Tint_hat)
        rmse =math.sqrt(mse)
        r2=r2_score(Tint_reel,Tint_hat)
        writer_object.writerow([home_id, mse,rmse,r2])
        fig, ax1 = plt.subplots()
        ax1.plot(Tint_reel,Tint_hat,marker='x',linestyle="",c='black')
        ax1.plot(range(int(min(Tint_reel)),int(max(Tint_reel))), range(int(min(Tint_reel)),int(max(Tint_reel))), c='green')
        plt.xlabel("Tint prediction")
        plt.ylabel("Tint true")
        plt.show()
        # save_figure(ax=ax1, filename=home_id + "_regression_result.png", directory=PATH_TO_OUTPUT_RESULT)
        # plt.close()

        txt_filename = PATH_TO_OUTPUT_RESULT + home_id + '_' + str(diffFeg) + "_lag_Var_results.txt"

        with open(txt_filename, "w") as text_file:
            test = str(results.summary())
            text_file.write(str(results.summary()))

    else :
        mdata_diff = VAR_data.diff().dropna()
        diffFeg+=1
    # except:
    #     pass

f_object.close()





