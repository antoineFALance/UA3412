import random

import pandas as pd
import os
import re
from matplotlib import pyplot as plt
import numpy as np
import scipy.optimize as so
from statistics import mean
import sys
import math
is_windows = hasattr(sys, 'getwindowsversion')

# PATH DEFINITION
if is_windows:
    PATH_TO_INPUT_DIR_DATA=os.path.dirname(os.getcwd())+"\\data_\\main_dataset\\"
    PATH_TO_OUTPUT_DIR_DATA=os.path.dirname(os.getcwd())+"\\data_\\dataset_gas_value_corrected\\"
else:
    PATH_TO_INPUT_DIR_DATA= os.path.dirname(os.getcwd())+"/data_/main_dataset/"
    PATH_TO_OUTPUT_DIR_DATA=os.path.dirname(os.getcwd())+"/data_/dataset_gas_value_corrected/"

directory = os.fsencode(PATH_TO_INPUT_DIR_DATA)

def RC_model(t,Te_1,Ti,T0):
    R=random.uniform(1.5, 2)
    C=random.uniform(880, 900)
    phi_1=1/(R*(1-math.exp(-t/(R*C))))*(Ti-Te_1*(1-math.exp(-t/(R*C)))-T0*math.exp(-t/(R*C)))
    bruit=random.uniform(-1,1)
    return max(0,phi_1+bruit)

for file in os.listdir(directory):
    print(file)
    filename = os.fsdecode(file)
    home_id = re.search('weather_(.*).csv', filename).group(1)
    fullFileName=PATH_TO_INPUT_DIR_DATA+filename
    df_dataset = pd.read_csv(fullFileName, sep=";")
    df_dataset['t'] = range(1,df_dataset.shape[0]+1)

    df_dataset['Ti_1']=df_dataset['Tint'].shift(1)
    df_dataset['Te_1'] = df_dataset['Text'].shift(1)
    T0=df_dataset['Tint'].to_list()[0]
    df_dataset['phi_1']=df_dataset.apply(lambda x: RC_model(t=x.t,Ti=x.Tint,Te_1=x.Te_1,T0=T0),axis=1)
    df_dataset['phi_corrected']=df_dataset['phi_1'].shift(-1)

    ax=df_dataset.plot(x='t',y='phi_corrected')
    df_dataset.plot(x='t', y='Tint',ax=ax)
    df_dataset.plot(x='t', y='Te_1', ax=ax)

    output_filename=home_id+'_dataset.csv'
    df_dataset.to_csv(PATH_TO_OUTPUT_DIR_DATA+output_filename,sep=';',index=False)
    print('ok')

