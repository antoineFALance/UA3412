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

def save_figure(fig,path):
    fig.savefig(path)
    return True

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
                test = np.hstack([ds[:-1,:-1].flatten(),ds[-2,-1]])
                x_ds.append(np.hstack([ds[:-1,:-1].flatten(),ds[-2,-1]]))
                # x_ds.append(ds[:-1, :-1].flatten())
                y_ds.append(ds[-1,-1])
    x=np.stack(x_ds)
    y=np.stack(y_ds).reshape(-1, 1)
    return x,y

if is_windows:
    PATH_TO_INPUTS_DIR=os.path.dirname(os.path.dirname(os.getcwd()))+"\\data_\\main_dataset\\"
    PATH_TO_OUTPUT_MODEL=os.path.dirname(os.path.dirname(os.getcwd()))+"\\Models\\XGBoost\\models\\"
    PATH_TO_OUTPUT_RESULTS = os.path.dirname(os.path.dirname(os.getcwd()))+"\\Models\\XGBoost\\results\\"
else:
    PATH_TO_INPUTS_DIR = os.path.dirname(os.path.dirname(os.getcwd())) + "/data_/main_dataset/"
    PATH_TO_OUTPUT_MODEL = os.path.dirname(os.path.dirname(os.getcwd())) + "/Models/XGBoost/models/"
    PATH_TO_OUTPUT_RESULTS = os.path.dirname(os.path.dirname(os.getcwd())) + "/Models/XGBoost/results/"

df_results=pd.DataFrame()

# FEATURES
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

fileList=[file for file in os.listdir(PATH_TO_INPUTS_DIR)]
for file in fileList:
    for features_combination in feat_comb:
        for window_range in range(2,3,1):
            df_temp=pd.DataFrame()
            df_temp['features_combination'] = np.array(["|".join(list(features_combination[:-1]))])
            df_temp['window_range']=np.array([window_range])

            filename = os.fsdecode(file)
            fullFileName=PATH_TO_INPUTS_DIR+'/'+filename
            home_id = re.search('weather_(.*).csv', filename).group(1)
            df_temp['home_id']=np.array([home_id])
            df_dataset=pd.read_csv(fullFileName,sep=";")
            df_dataset['gas_value'].fillna(df_dataset['gas_value'].min(),inplace=True)

            dataset=df_dataset[list(features_combination)]
            mask = [not state for state in pd.isnull(dataset).any(axis=1).to_list()]
            dataset_masked = dataset[mask]
            x,y=windowing(dataset_masked,windowRange=window_range)

            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)
            model = xgb.XGBRegressor()

            parameters={
                'max_depth': [3, 4, 5],
                'n_estimators':[10,50,100],
                'learning_rate': [0.1, 0.01, 0.001],
                'subsample': [0.5, 0.7, 1]
            }

            cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
            clf = GridSearchCV(model, parameters, cv=cv.split(X_train, y_train))
            clf.fit(X_train,y_train)

            best_xgb = clf.best_estimator_
            val =home_id+"_"+str(window_range)+"_"+ '_'.join(list(features_combination[:-1]))
            MODEL_NAME=hashlib.md5(val.encode('utf-8')).hexdigest()

            # FILENAME = f"{home_id}_{window_range}_{features_combination}".format(home_id=home_id,
            #                                                                      window_range=window_range,
            #                                                                    features_combination=test)
            FILENAME = val
            df_temp['model_name']=MODEL_NAME
            best_xgb.save_model(PATH_TO_OUTPUT_MODEL+MODEL_NAME+"_xgboost.json")
            test = [feat + "_" + str(j) for j in range(window_range - 1, 0, -1) for feat in features_combination[:-1]]+['Tint']
            best_xgb.get_booster().feature_names =[feat + "_" + str(j) for j in range(window_range - 1, 0, -1) for feat in features_combination[:-1]]+['Tint']
            y_pred = best_xgb.predict(X_test)

            # MSE
            mse=mean_squared_error(y_test.flatten(),y_pred)
            df_temp['mse']=np.array([mse])
            df_temp['rmse']=df_temp['mse'].apply(lambda x:x**0.5)
            params= best_xgb.get_xgb_params()
            # df_temp = df_temp.join(pd.DataFrame([params], index=['0']))
            df_results=pd.concat([df_results,df_temp])

            # FEATURES IMPORTANCE
            for importanceType in ['weight','gain']:
                ax=plot_importance(best_xgb,importance_type=importanceType )
                fig = ax.get_figure()
                save_figure(fig,PATH_TO_OUTPUT_RESULTS + FILENAME +"_"+importanceType+"_features_importance.png")
                plt.close()

            # COMPARAISON PREDICTION vs LABEL
            y_pred = best_xgb.predict(X_test[0:100])
            df_res_plot=pd.DataFrame(columns=['pred','label'])
            df_res_plot['pred']=y_pred
            df_res_plot['label']=y_test[0:100].flatten()
            plt.figure(figsize=(12, 5))
            plt.xlabel('Number of requests every 10 minutes')
            ax1 = df_res_plot.pred.plot(color='blue', grid=True)
            ax2 = df_res_plot.label.plot(color='red', grid=True)
            fig = ax2.get_figure()

            save_figure(fig, PATH_TO_OUTPUT_RESULTS +FILENAME+ "_result_plot.png")
            plt.close()
            print(f"home:{home_id}-window range:{window_range}-features:{features_combination}".format(home_id=home_id,window_range=window_range,features_combination=features_combination[:-1]))


df_results.to_csv(PATH_TO_OUTPUT_RESULTS+"xgboost_models_results.csv",sep=";",index=False)