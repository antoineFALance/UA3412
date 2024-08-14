import pandas as pd
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")

df= pd.read_csv('C:\\Users\\a.lance\\PycharmProjects\\UA3412_\\IDEAL_home106\\home106_main_data_set_HR.csv',sep=";")
# df=df[150:]
# df['consigne'] = np.where((df['hour'] >= 8) & (df['hour'] <= 20), 18, 15)
df['consigne'] = np.where((df['hour'] >= 8) & (df['hour'] <= 17), 18, np.nan)
df['consigne'] = np.where((df['hour'] >= 20), 15, df['consigne'])
df['consigne'] = np.where((df['hour'] <= 5) & (df['hour'] <= 23), 15, df['consigne'])
df['consigne'].interpolate(inplace=True)
import heater_env

PATH_TO_OUTPUT_MODELS = "C:\\Users\\a.lance\\PycharmProjects\\UA3412_\\RL\\DQN\\models\\target_net"


env = heater_env.heaterEnvRC()
state=env.reset()
agent = tf.keras.models.load_model(PATH_TO_OUTPUT_MODELS)
actionList,TiList=[],[]


for index in range(df.shape[0]):
    action =np.argmax(agent(np.atleast_2d(np.atleast_2d(state).astype('float32'))))
    actionList.append(action)
    next_state, reward, done, _,__ = env.step(action)
    state=next_state
    TiList.append(next_state[0])
index=1000
figure,axs =plt.subplots(3)
axs[0].plot(actionList[:index])
axs[1].plot(TiList[:index])
axs[1].plot(df['consigne'].to_list()[:index],c='orange')
axs[2].plot(df['Text'].to_list()[:index])
plt.show()

