import pandas as pd
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import heater_env3

df= pd.read_csv('C:\\Users\\a.lance\\PycharmProjects\\UA3412_\\IDEAL_home106\\home106_main_data_set_HR.csv',sep=";")
# df=df[150:]
# df['consigne'] = np.where((df['hour'] >= 8) & (df['hour'] <= 20), 18, 15)
df['consigne'] = np.where((df['hour'] >= 8) & (df['hour'] <= 17), 18, np.nan)
df['consigne'] = np.where((df['hour'] >= 20), 0, df['consigne'])
df['consigne'] = np.where((df['hour'] <= 5) & (df['hour'] <= 23), 0, df['consigne'])
df['consigne'].interpolate(inplace=True)

PATH_TO_OUTPUT_MODELS = "C:\\Users\\a.lance\\PycharmProjects\\UA3412_\\RL\\DQN_3\\models\\target_net"


env = heater_env3.heaterEnvRC()
state=env.reset()
agent = tf.keras.models.load_model(PATH_TO_OUTPUT_MODELS)
actionList,TiList=[],[]

idx=50
for index in range(idx):
    action =np.argmax(agent(np.atleast_2d(np.atleast_2d(state).astype('float32'))))
    actionList.append(action)
    next_state, reward, done, _,__ = env.step(action)
    state=next_state
    TiList.append(next_state[0])


figure,axs =plt.subplots(3)
axs[0].plot(actionList[:idx])
axs[1].plot(TiList[:idx])
axs[1].plot(df['consigne'].to_list()[:idx],c='orange')
axs[2].plot(df['Text'].to_list()[:idx])
plt.show()
print("Conso energie: "+str(sum(actionList[:idx])))

