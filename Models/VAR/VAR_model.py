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
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
from scipy import signal
import warnings
warnings.filterwarnings("ignore")

def save_figure(ax,filename,directory):
    fig = ax.get_figure()
    fig.savefig(directory + filename + ".png")

def window_range(data,wdw):
    return [data[idx:idx+wdw,:] for idx in range(data.shape[0]-wdw)]





# PATH DEFINITION
if is_windows:
    PATH_TO_INPUT_DIR_DATA= os.path.dirname(os.path.dirname(os.getcwd()))+"\\data_\\dataset_gas_value_corrected\\"
    PATH_TO_INPUT_DIR_MODEL=os.path.dirname(os.path.dirname(os.getcwd()))+"\\Models\\VAR\\results\\"
    PATH_TO_OUTPUT_RESULT = os.path.dirname(os.path.dirname(os.getcwd()))+"\\Models\\VAR\\results\\"

else:
    PATH_TO_INPUT_DIR_DATA= os.path.dirname(os.path.dirname(os.getcwd()))+"/data_/dataset_gas_value_corrected/"
    PATH_TO_INPUT_DIR_MODEL=os.path.dirname(os.path.dirname(os.getcwd()))+"/Models/VAR/results/"
    PATH_TO_OUTPUT_RESULT = os.path.dirname(os.path.dirname(os.getcwd())) + "/Models/VAR/results/"

directory = os.fsencode(PATH_TO_INPUT_DIR_DATA)

for file in os.listdir(directory):
    print(file)
    filename = os.fsdecode(file)
    home_id = re.search('(.*)_dataset.csv', filename).group(1)
    fullFileName=PATH_TO_INPUT_DIR_DATA+filename
    df_temp=pd.DataFrame()
    df_dataset = pd.read_csv(fullFileName, sep=";")
    df_dataset['datetime']=pd.to_datetime(df_dataset[['year', 'month', 'day', 'hour']])
    df_dataset['yearmonth']=df_dataset['year']*100+df_dataset['month']
    df_dataset.index = pd.DatetimeIndex(df_dataset['datetime'])
    df_dataset['is_winter']=np.where((df_dataset['yearmonth']==201709) |\
                                     (df_dataset['yearmonth']==201710) |\
                                     (df_dataset['yearmonth']==201711) |\
                                     (df_dataset['yearmonth']==201712) |\
                                     (df_dataset['yearmonth']==201801) |\
                                     (df_dataset['yearmonth']==201802) |\
                                     (df_dataset['yearmonth']==201803),1,0)

    # mdata = df_dataset[df_dataset['is_winter']==1][['Tint','Text','gas_value']].dropna()
    mdata =df_dataset[['Tint','Text','gas_value']].dropna()

    # Nettoyage signal
    windowPerc=0.05
    for col in mdata.columns:
        mdata[col+'_savgolfilter'] = signal.savgol_filter(mdata[col].to_numpy(),int(mdata.shape[0]*windowPerc),3)
        mdata[col + '_MAfilter'] =mdata[col].rolling(20).mean()
        mdata[col + '_EMAfilter'] = mdata[col].ewm(span=20, adjust=False).mean()

    # ax1 = mdata.Tint.plot(color='black', grid=True,)
    # ax1 = mdata.Tint_savgolfilter.plot(color='red', grid=True)
    # ax1 = mdata.Tint_MAfilter.plot(color='green', grid=True)
    # ax1 = mdata.Tint_EMAfilter.plot(color='blue', grid=True)
    # plt.show()


    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    # ax1.plot(range(mdata.shape[0]),mdata['Text_filter'].to_numpy(),c='red',label="Text")
    # ax1.plot(range(mdata.shape[0]), mdata['Tint_filter'].to_numpy(),c='blue',label='Tint')
    # ax2.plot(range(mdata.shape[0]), mdata['gas_value_filter'].to_numpy(),c='green',label="Gas")
    # ax1.legend()
    # ax2.legend()
    # save_figure(ax=ax1,filename=home_id+"_data_filtered.png",directory=PATH_TO_OUTPUT_RESULT)
    # # plt.show()
    # plt.close()
    #
    mdata_filter=mdata[['Text','gas_value','Tint']]

    mdata_diff = mdata_filter
    diffFeg=0
    while True:
        p_value_list = []
        for variable in mdata_diff.columns:
            result = adfuller(mdata_diff[variable])
            p_value_list.append(result[1])
        if all(pvalue<0.05 for pvalue in p_value_list):
            print('pvalues ok')
            split_index=0.7
            mdata_train,mdata_test=mdata_diff.values[:int(split_index*mdata_diff.shape[0]),:],mdata_diff.values[int(split_index*mdata.shape[0]):,:]

            # Construction du modèle
            model = VAR(mdata_train)
            optimal_lags = model.select_order()
            lag_order = optimal_lags.selected_orders['bic']
            data_test = window_range(mdata_test,lag_order)
            results = model.fit(lag_order)
            var_model = results.model

            # Vérification du modèle
            Tint_hat = [results.forecast(y_test,1).flatten()[2] for y_test in data_test]
            Tint_reel = [mdata_test[idx][2] for idx in range(lag_order,mdata_test.shape[0])]
            fig, ax1 = plt.subplots()
            ax1.plot(Tint_reel,Tint_hat,marker='x',linestyle="",c='black')
            ax1.plot(range(int(min(Tint_reel)),int(max(Tint_reel))), range(int(min(Tint_reel)),int(max(Tint_reel))), c='green')
            save_figure(ax=ax1, filename=home_id + "_regression_result.png", directory=PATH_TO_OUTPUT_RESULT)
            plt.close()

            txt_filename = PATH_TO_OUTPUT_RESULT + home_id + '_' + str(diffFeg) + "_lag_Var_results.txt"

            with open(txt_filename, "w") as text_file:
                test = str(results.summary())
                text_file.write(str(results.summary()))
            break
        else :
            mdata_diff = mdata_diff.diff().dropna()
            diffFeg+=1




