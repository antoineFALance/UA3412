import pandas as pd
from scipy.optimize import milp
from scipy.optimize import LinearConstraint
from scipy.optimize import linprog
import numpy as np
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import math
from sklearn.metrics import r2_score,mean_squared_error
import tqdm

df_result=pd.DataFrame()
df= pd.read_csv('/IDEAL_home106/home106_main_data_set_HR.csv', sep=";")
# df=df[(((df['month']==11) & (df['day']>=25)) | (df['month']==12) | (df['month']==1)) & (df['year']==2017)]
hour_range=500
# Tc=df['consigne'].to_list()[:hour_range]
Tc=df['Tint'].to_list()[:hour_range]
Text=df['Text'].to_list()[:hour_range]
phi=df['gas_value'].to_list()[:hour_range]
T0=df['Tint'].to_list()[0]
gamma =
df['Tint_pred_func']=df.apply(lambda x: )

