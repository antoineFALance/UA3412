import math
import random

import pandas as pd
import os
import re
from matplotlib import pyplot as plt
import numpy as np
import scipy.optimize as so
from statistics import mean,stdev
from sklearn.metrics import mean_squared_error
import sys
is_windows = hasattr(sys, 'getwindowsversion')
import scipy.stats as stats

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

def simpleRC(ti_1,te_1,phi_1,R,C):
    gamma=R*C
    ti = ti_1*math.exp(-1/(gamma))+(R*phi_1+te_1)*(1-math.exp(-1/gamma))
    return ti

df_model=pd.read_csv(PATH_TO_INPUT_DIR_MODEL,sep=";")

window_tange=2000
mseList=[]
for homeId in df_model['home_id'].unique():
# for file in os.listdir(directory):
#     print(file)
    # filename = os.fsdecode(file)
    filename=homeId+'_dataset.csv'
    # home_id = re.search('(.*)_dataset.csv', filename).group(1)
    fullFileName=PATH_TO_INPUT_DIR_DATA+filename
    df_dataset = pd.read_csv(fullFileName, sep=";")
    index =random.randint(0,df_dataset.shape[0]-window_tange)
    R=df_model[df_model['home_id']==homeId]['Rmean'].to_list()[0]
    C = df_model[df_model['home_id'] == homeId]['Cmean'].to_list()[0]
    df_dataset['ti_pred']=df_dataset.apply(lambda x : simpleRC(ti_1=x.Ti_1,te_1=x.Te_1,phi_1=x.phi_1,R=R,C=C),axis=1)
    df_mse=df_dataset[['Tint','ti_pred']].dropna()
    mseList.append(mean_squared_error(df_mse['Tint'].to_numpy(),df_mse['ti_pred'].to_numpy()))
    df_plot=df_dataset.loc[index:index+window_tange]
    ax=df_plot.plot.scatter(x='ti_pred',y='Tint',c='black',marker='x')
    # df_plot.plot(x='t', y='ti_pred',ax=ax,c='green')
    # plt.show()
    fig = ax.get_figure()
    FILENAME = homeId + "_" + "RC"
    fig.savefig(PATH_TO_OUTPUT_RESULT + FILENAME + "_result_plot.png")
    plt.close()
    print('ok')

df_model['mse']=np.array(mseList)
df_model.to_csv(PATH_TO_OUTPUT_RESULT+'RC_model.csv',sep=";",index=False)

