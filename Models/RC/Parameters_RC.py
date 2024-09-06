# import math
# import scipy.optimize as so
# import pandas as pd
# from matplotlib import pyplot as plt
# import numpy as np
#
# PATH_TO_HOUR_FILE="C:\\Users\\a.lance\\PycharmProjects\\UA3412_\\IDEAL_home106\\home106_main_data_set_HR.csv"
# df=pd.read_csv(PATH_TO_HOUR_FILE,sep=";")
#
# def extractContinousSeries(df):
#     for index,rows in df.iterrows:
#         if rows['index_gap']>11:
#             decreaseIndexList.append(rows['index'])
#             decreaseTintList.append(rows['Tint'])
#             decreaseTextList.append(rows['Text'])
#         else:
#
#
# # Recupération données "décharge uniquement
# df['Tint1']=df['Tint'].shift(-1)
# df['delta_Tint']=df['Tint1']-df['Tint']
# df['is_Tint_decrease']=np.where(df['delta_Tint']>0,1,0)
# df['next_bool_Tint_decrease']=df['is_Tint_decrease'].shift(-1)
# df['previous_bool_Tint_decrease']=df['is_Tint_decrease'].shift(1)
# df['is_T0_before_cooldown']=np.where((df['next_bool_Tint_decrease']==1) & (df['previous_bool_Tint_decrease']==0)
#                                                                        & (df['is_Tint_decrease']==0),1,0)
#
# df_cooldown=df[((df['is_Tint_decrease']==1) | (df['is_T0_before_cooldown']==1)) & (df['gas_value']==0)]
#
# df_cooldown['index']=df_cooldown.index
# df_cooldown['index_gap']=df_cooldown['index']-df_cooldown['index'].shift(1)
#
# df_cooldown=df[(df['hour']>=0) & (df['hour']<=7) & (df['gas_value']==0)]
# print('ok')
