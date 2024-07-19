import math
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import plot_importance,plot_tree
import sys
import re
is_windows = hasattr(sys, 'getwindowsversion')
from sklearn import preprocessing
from sklearn.model_selection import RepeatedKFold,cross_val_score,train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from itertools import combinations,chain
import hashlib


if is_windows:
    PATH_TO_INPUTS_DIR=os.path.dirname(os.path.dirname(os.getcwd()))+"\\data_\\dataset_gas_value_corrected\\"
    PATH_TO_OUTPUT_MODEL=os.path.dirname(os.path.dirname(os.getcwd()))+"\\Models\\XGBoost\\models\\"
    PATH_TO_OUTPUT_RESULTS = os.path.dirname(os.path.dirname(os.getcwd()))+"\\Models\\XGBoost\\results\\"
else:
    PATH_TO_INPUTS_DIR = os.path.dirname(os.path.dirname(os.getcwd())) + "/data_/dataset_gas_value_corrected/"
    PATH_TO_OUTPUT_MODEL = os.path.dirname(os.path.dirname(os.getcwd())) + "/Models/XGBoost/models/"
    PATH_TO_OUTPUT_RESULTS = os.path.dirname(os.path.dirname(os.getcwd())) + "/Models/XGBoost/results/"

df_results=pd.DataFrame()

def save_figure(ax,filename,directory):
    fig = ax[0].get_figure()
    fig.savefig(directory + filename + ".png")

# FEATURES
features_col=['ti_1','te_1','phi_1','Tint']


fileList=[file for file in os.listdir(PATH_TO_INPUTS_DIR)]
for file in fileList:
        print(file)
        df_temp=pd.DataFrame()
        filename = os.fsdecode(file)
        fullFileName=PATH_TO_INPUTS_DIR+filename
        home_id = re.search('(.*)_dataset.csv', filename).group(1)
        df_temp['home_id']=np.array([home_id])
        df_dataset=pd.read_csv(fullFileName,sep=";")
        df_xgboost = df_dataset[['Ti_1', 'Te_1', 'phi_1', 'Tint']].dropna()

        x_train, x_test, y_train, y_test = train_test_split(df_xgboost[['Ti_1', 'Te_1', 'phi_1']].to_numpy(),
                                                            df_xgboost[['Tint']].to_numpy().flatten(), test_size=0.2,
                                                            random_state=123)
        parameters = {
            'max_depth': [3, 4, 5,10],
            'n_estimators': [10, 50, 100,200],
            'learning_rate': [0.1, 0.01, 0.001],
            'subsample': [0.5, 0.7, 1]
        }

        # parameters = {
        #     'max_depth': [3],
        #     'n_estimators': [10],
        #     'learning_rate': [0.1],
        #     'subsample': [0.5]
        # }

        model = xgb.XGBRegressor()
        cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
        clf = GridSearchCV(model, parameters, cv=cv.split(x_train, y_train))
        clf.fit(x_train, y_train)
        best_xgb = clf.best_estimator_
        y_pred = best_xgb.predict(x_test)
        mse = mean_squared_error(y_test,y_pred)

        ax1 = plt.plot(y_test, y_pred, c='black', marker='x', linestyle="")
        plt.plot(np.array(range(12, 30, 1)), np.array(range(12, 30, 1)), c='green', linestyle="-",ax=ax1)
        plt.title("predictions / valeurs reel")
        plt.xlabel("y_pred")
        plt.ylabel("y_reel")
        save_figure(ax=ax1, filename=home_id + "_" + "xgboost_pred_reel", directory=PATH_TO_OUTPUT_RESULTS)
        plt.close()
        df_temp['max_depth']=best_xgb.max_depth
        df_temp['n_estimators'] = best_xgb.n_estimators
        df_temp['learning_rate'] = best_xgb.learning_rate
        df_temp['subsample'] = best_xgb.subsample
        df_temp['mse']=mse
        df_temp['rmse']=math.sqrt(mse)
        df_results=pd.concat([df_results,df_temp])
        print('ok')

df_results.to_csv(PATH_TO_OUTPUT_RESULTS+"xgboost_models_results.csv",sep=";",index=False)