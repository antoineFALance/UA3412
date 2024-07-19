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


def adf_test(timeseries):
    print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    print(dfoutput)
    return dftest[1]



# PATH DEFINITION
if is_windows:
    PATH_TO_INPUT_DIR_DATA= os.path.dirname(os.path.dirname(os.getcwd()))+"\\data_\\dataset_gas_value_corrected\\"
    PATH_TO_INPUT_DIR_MODEL=os.path.dirname(os.path.dirname(os.getcwd()))+"\\Models\\RC\\results\\RC_model.csv"
    PATH_TO_OUTPUT_RESULT = os.path.dirname(os.path.dirname(os.getcwd()))+"\\Models\\RC\\results\\"
else:
    PATH_TO_INPUT_DIR_DATA= os.path.dirname(os.path.dirname(os.getcwd()))+"/data_/dataset_gas_value_corrected/"
    PATH_TO_INPUT_DIR_MODEL=os.path.dirname(os.path.dirname(os.getcwd()))+"/Models/RC/results/RC_model.csv"
    PATH_TO_OUTPUT_RESULT = os.path.dirname(os.path.dirname(os.getcwd())) + "/Models/RC/results/"

directory = os.fsencode(PATH_TO_INPUT_DIR_DATA)

for file in os.listdir(directory):
    print(file)
    filename = os.fsdecode(file)
    # home_id = re.search('(.*)_dataset.csv', filename).group(1)
    fullFileName=PATH_TO_INPUT_DIR_DATA+filename
    df_temp=pd.DataFrame()
    df_dataset = pd.read_csv(fullFileName, sep=";")
    df_dataset['datetime']=pd.to_datetime(df_dataset[['year', 'month', 'day', 'hour']])
    df_dataset.index = pd.DatetimeIndex(df_dataset['datetime'])
    mdata = df_dataset[['Tint','Text','gas_value','wind_x','pressure','hum_value','rh','dewPt']].dropna()
    # TEST DE STAIONARITE
    p_value=[]
    crossCorrDict={}
    colDict={}
    max_lag=4
    for col in ['Tint'] :
        p_value.append(adf_test(mdata[col]))
        print('ok')
        # AUTO-CORRELATION
        auto_corr=[mdata[col].corr(mdata[col].shift(lag)) for lag in range(1,max_lag)]
        plot_pacf(mdata[col], lags=max_lag)

        # CROSS-CORR
        varList=[var for var in list(mdata.columns) if var != col]
        resultList=[]
        for var in varList:
            crossData=mdata[[col]+[var]]
            for lag in range(max_lag):
                crossData[var+'_lag_'+str(lag)] = crossData[var].shift(lag)
            crossData.dropna(inplace=True)
            for lag in range(1,max_lag):
                resultList.append([var+'_'+str(lag),
                                   stats.pearsonr(crossData[col].to_numpy(),crossData[var+'_lag_'+str(lag)].to_numpy())[0],
                                   stats.pearsonr(crossData[col].to_numpy(),crossData[var + '_lag_' + str(lag)].to_numpy())[1]
                                   ])
        df_cross_result = pd.DataFrame(data=resultList,columns=['variable','pearson','pvalue'])
        print('ok')

    # mdata_train=np.array_split(mdata,2)[0]
    # mdata_test = np.array_split(mdata, 2)[1]
    print('ok')
    model = VAR(mdata)
    model.select_order(max_lag)
    results = model.fit(maxlags=max_lag, ic='aic')
    print(results.summary())
    Ti_coeff_p_values= results.pvalues['Tint'].to_list()
    test = results.coefs.shape[0]
    test = results.coefs
    test2 = results.coefs_exog[0][0]
    Ti_coeff = np.array([results.coefs_exog[0][0]]+list(np.array([list(results.coefs[i,0,:].flatten()) for i in range(results.coefs.shape[0])]).flatten()))
    df_Ti_coeff=pd.DataFrame(data=Ti_coeff,index=results.pvalues['Tint'].index)
    mask = np.array(Ti_coeff_p_values) < 0.03
    test = df_Ti_coeff[mask]
    print('ok')
