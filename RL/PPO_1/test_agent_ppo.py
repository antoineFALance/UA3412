import pandas as pd
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import heater_env_1

df= pd.read_csv('C:\\Users\\a.lance\\PycharmProjects\\UA3412_\\IDEAL_home106\\home106_main_data_set_Min.csv',sep=";", decimal=",")
df = df[(df['year'] == 2018) & (df['month'] == 2) & ((df['day'] >= 4) & (df['day'] <= 4))]
# df=df[150:]
# df['consigne'] = np.where((df['hour'] >= 8) & (df['hour'] <= 20), 18, 15)
df['consigne'] = np.where((df['hour'] >= 10) & (df['hour'] <= 12), 18, np.nan)
df['consigne'] = np.where((df['hour'] >= 14) & (df['hour'] <= 17), 12, df['consigne'])
df['consigne'] = np.where((df['hour'] >= 19) & (df['hour'] <= 22), 10, df['consigne'])
df['consigne'] = np.where((df['hour'] >= 0) & (df['hour'] <= 1), 5, df['consigne'])
df['consigne'] = np.where((df['hour'] >= 3) & (df['hour'] <= 3), 7, df['consigne'])
df['consigne'] = np.where((df['hour'] >= 5) & (df['hour'] <= 7), 3, df['consigne'])
df['consigne'].interpolate(inplace=True)
#
PATH_TO_OUTPUT_MODELS = "C:\\Users\\a.lance\\PycharmProjects\\UA3412_\\RL\\PPO_1\\models\\actor"
seed_generator =tf.random.set_seed(1337)

env = heater_env_1.heaterEnvRC()
state=env.reset()
agent = tf.keras.models.load_model(PATH_TO_OUTPUT_MODELS)
actionList,TiList=[],[]


idx=df.shape[0]
for index in range(idx):
    logits = agent(np.atleast_2d(state))
    action = tf.squeeze(tf.random.categorical(logits, 1, seed=seed_generator), axis=1)

    # action =np.argmax(agent(np.atleast_2d(np.atleast_2d(state).astype('float32'))))
    # action=agent(np.atleast_2d(np.atleast_2d(state).astype('float32')))
    actionList.append(action[0].numpy())
    next_state, reward, done, _,__ = env.step(action[0].numpy())
    state=next_state
    TiList.append(next_state[0])


figure,axs =plt.subplots(3)
axs[0].plot(actionList[:idx])
axs[1].plot(TiList[:idx])
axs[1].plot(df['consigne'].to_list()[:idx],c='orange')
axs[2].plot(df['Text'].to_list()[:idx])
plt.show()
print("Conso energie: "+str(sum(actionList[:idx])))

