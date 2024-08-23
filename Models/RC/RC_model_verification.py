import math
import scipy.optimize as so
import pandas as pd
from matplotlib import pyplot as plt

PATH_TO_MAIN_FILE="C:\\Users\\a.lance\\PycharmProjects\\UA3412_\\IDEAL_home106\\home106_main_data_set.csv"
PATH_TO_HOUR_FILE="C:\\Users\\a.lance\\PycharmProjects\\UA3412_\\IDEAL_home106\\home106_main_data_set_HR.csv"

def Tint(input,R,C):
    Tint_1=input[:,0]
    Text_1 = input[:,1]
    phi_1 = input[:,2]
    gamma =  math.exp(-1/(R*C))
    return Tint_1*gamma+(R*phi_1+Text_1)*(1-gamma)

df=pd.read_csv(PATH_TO_MAIN_FILE,sep=";")

df_RC = df[['Tint','Text','gas_value_corrected']]
df_RC['Tint1']=df['Tint'].shift(-1)
df_RC.dropna(inplace=True)

inputs=df_RC[['Tint','Text','gas_value_corrected']].to_numpy()
output=df_RC['Tint1'].to_numpy()

p_opt, p_cov = so.curve_fit(f=Tint,
                            xdata=inputs,
                            ydata=output,
                            p0=(0.01,100),
                            bounds=([0.01,0.05], [100,200])
                            )

gamma=math.exp(-1/(p_opt[0]*p_opt[1]))
df_RC_hr=pd.read_csv(PATH_TO_HOUR_FILE,sep=";")
df_RC_hr['Tint1']=df_RC_hr['Tint'].shift(-1)
df_RC_hr.dropna(inplace=True)

df_RC_hr['Tint_hat_1']=df_RC_hr.apply(lambda x :x.Tint*gamma+(p_opt[0]*x.gas_value_corrected+x.Text)*(1-gamma),axis=1)
plt.plot(df_RC_hr['Tint_hat_1'].to_list(),df_RC_hr['Tint1'].to_list(),marker='x',c='black', linestyle='None')
plt.plot(range(int(df_RC_hr['Tint'].min()),int(df_RC_hr['Tint'].max())),range(int(df_RC_hr['Tint'].min()),int(df_RC_hr['Tint'].max())),c='green')
plt.xlabel("Tint prediction")
plt.ylabel("Tint true")
plt.show()

print('ok')