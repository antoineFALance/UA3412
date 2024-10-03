import math
import random
from contextlib import closing
from io import StringIO
from os import path
from typing import List, Optional
import pandas as pd
import numpy as np
from typing import Optional, Union
from gym import Env, logger, spaces, utils
import gym
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from matplotlib import pyplot as plt
import random
# random.seed(123)
from IPython import display

class heaterEnvRC(gym.Env[np.ndarray, Union[int, np.ndarray]]):

    def __init__(self):

        self.t=0
        self.horizon=5

        self.data = pd.read_csv('C:\\Users\\a.lance\\PycharmProjects\\UA3412_\\IDEAL_home106\\home106_main_data_set_Min.csv',sep=";", decimal=",")

        self.data = self.data[(self.data['year'] == 2018)
                              & (self.data['month'] == 2)
                              & ((self.data['day'] >= 4) & (self.data['day'] <= 4))
                            ]

        self.window_range = self.data.shape[0]-self.horizon

        self.data['rad_soleil'] = self.data.apply(lambda x: max(0, -2.0 * math.cos(math.pi / 12 * x.hour)), axis=1)
        self.data['rad_soleil'] = np.where(self.data['min'] == 0, self.data['rad_soleil'], np.nan)
        self.data['rad_soleil'].interpolate(inplace=True)

        self.data['consigne'] = np.where((self.data['hour'] >= 10) & (self.data['hour'] <= 12), 18, np.nan)
        self.data['consigne'] = np.where((self.data['hour'] >= 14) & (self.data['hour'] <= 17), 12,  self.data['consigne'])
        self.data['consigne'] = np.where((self.data['hour'] >= 19) & (self.data['hour'] <= 22), 10,self.data['consigne'])
        self.data['consigne'] = np.where((self.data['hour'] >= 0) & (self.data['hour'] <= 1), 5,self.data['consigne'])
        self.data['consigne'] = np.where((self.data['hour'] >= 3) & (self.data['hour'] <= 3), 7, self.data['consigne'])
        self.data['consigne'] = np.where((self.data['hour'] >= 5) & (self.data['hour'] <= 7), 3, self.data['consigne'])


        self.data['consigne'].interpolate(inplace=True)
        self.data.reset_index(inplace=True)

        self.data['gas_value'].fillna(0, inplace=True)

        self.consigneList=self.data['consigne'].to_list()

        self.ci = 90
        self.ce = 3000
        self.ri=6.0e-02
        self.re = 2.5
        self.hrad=3.9e-02
        self.As=5
        self.cwater=4.18

        low = np.array([0]*(1+2*self.horizon), dtype=np.float32, )
        high = np.array([100]*(1+2*self.horizon),dtype=np.float32,)

        self.action_space = spaces.Discrete(101)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.continuousRewardserie=0
        # self.consigne=18
        self.ti=[]
        self.tret=[]
        self.te=[]
        self.Qheat=[]
        self.Qs=self.data['rad_soleil'].to_list()
        self.to= self.data['Text'].to_list()

        # Initial temperatures
        self.Qheat.append(0)
        self.ti.append(5)
        self.tret.append(self.ti[-1])
        self.te.append((self.ri * self.to[0] +self.re * self.ti[0]) / (self.ri + self.re))

        #Render
        self.refresh_rate=5

    def on_launch(self):
        # Set up plot
        self.figure, self.ax = plt.subplots()
        self.tc_lines, = self.ax.plot(range(self.data.shape[0]),self.data['consigne'].to_list())
        self.ti_lines, = self.ax.plot([],[])

        # Autoscale on unknown axis and known lims on the other
        self.ax.set_autoscaley_on(True)
        self.ax.set_xlim(0, self.data.shape[0])

    def on_running(self):
        plt.clf()
        plt.plot(self.data['consigne'].to_list(),c='red')
        plt.plot(self.ti,c='green')
        plt.draw()
        plt.pause(0.1)

    def RCmodel(self,te_1,ti_1,to_1,tret_1,Qs_1,Qheat_1,dt=1):
        tret = tret_1 + dt / self.cwater * (Qheat_1 - self.hrad * (tret_1 - ti_1))
        ti = ti_1 + dt / self.ci * ((te_1 - ti_1) / self.ri + self.hrad * (tret_1 - ti_1) + self.As * Qs_1)
        te=te_1+dt/self.ce*((ti_1-te_1)/self.ri+(to_1-te_1)/self.re)
        return ti,te,tret

    def step(self,action):
        # reward =0
        gas_value=1000*action/100
        ti1,te1,tret1=self.RCmodel(te_1=self.te[-1],ti_1=self.ti[-1],to_1=self.to[-1],tret_1=self.tret[-1],Qheat_1=gas_value,Qs_1=self.Qs[-1])
        self.ti.append(ti1)
        self.tret.append(tret1)
        self.te.append(te1)
        self.Qheat.append(gas_value)

        self.state=tuple([ti1]+\
                        [self.data['consigne'].to_list()[self.t+idx] for idx in range(self.horizon)]+\
                        [self.data['Text'].to_list()[self.t+idx] for idx in range(self.horizon)])

        critere_1=self.t==self.window_range-self.horizon
        critere_2=ti1<0.95*self.data['consigne'].to_list()[self.t]

        terminated=any([self.t==self.window_range-self.horizon,ti1<0.95*self.data['consigne'].to_list()[self.t]])

        if not terminated:
            deltati=abs((ti1 - self.data['consigne'].to_list()[self.t]))
            reward = -2.0 *(action/100)-min(0.09*deltati,1)
            self.t += 1
            self.previousTint.append(ti1)
            self.continuousRewardserie+=1


        else:
            if critere_2:
                reward = -100
            else:
                reward=0
            plt.close()

        return  np.array(self.state, dtype=np.float32), reward, terminated, False,{}


    def reset(self):
        self.t = 0
        self.consigneList = self.data['consigne'].to_list()
        self.previousTint = [self.data['Tint_living'].to_list()[0]]
        self.continuousRewardserie = 0

        self.ti = []
        self.tret = []
        self.te = []
        self.Qheat = []
        self.to = self.data['Text'].to_list()

        # Initial temperatures
        self.Qheat.append(0)
        self.ti.append(5)
        self.tret.append( self.ti[-1])
        self.te.append((self.ri * self.to[0] + self.re * self.ti[0]) / (self.ri + self.re))

        reward=0

        return np.array( tuple([self.data['Tint_living'].to_list()[0]] +
                                [self.data['consigne'].to_list()[idx] for idx in range(self.horizon)] + \
                                [self.data['Text'].to_list()[idx] for idx in range(self.horizon)]), dtype=np.float32)


