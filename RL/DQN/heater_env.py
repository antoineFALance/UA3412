import math
from contextlib import closing
from io import StringIO
from os import path
from typing import List, Optional
import pandas as pd
import numpy as np
from typing import Optional, Union
from gym import Env, logger, spaces, utils
import gym

class heaterEnvRC(gym.Env[np.ndarray, Union[int, np.ndarray]]):

    def __init__(self):
        self.data = pd.read_csv('C:\\Users\\a.lance\\PycharmProjects\\UA3412_\\IDEAL_home106\\home106_main_data_set_HR.csv',sep=";")
        self.data['consigne']=np.where((self.data['hour']>=8) & (self.data['hour']<=20),18,15)
        self.consigneList=self.data['consigne'].to_list()
        self.R = 0.1
        self.C= 100
        self.gamma = math.exp(-1/(self.R*self.C))
        self.previousTint=[self.data['Tint'].to_list()[0]]
        self.Text=self.data['Text'].to_list()
        self.t=0
        self.state=(self.data['Tint'].to_list()[0],self.data['consigne'].to_list()[1],self.data['Text'].to_list()[1])


        low = np.array([0,0,0 ], dtype=np.float32, )
        high = np.array([100,100,100],dtype=np.float32,)

        self.action_space = spaces.Discrete(11)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.continuousRewardserie=0
        # self.consigne=18

    def step(self,action):
        gas_value=200*action/10
        Tint1=self.previousTint[-1]*self.gamma+(self.R*gas_value+self.Text[self.t])*(1-self.gamma)
        self.state=(Tint1,
                    self.data['consigne'].to_list()[self.t+1],
                    self.data['Text'].to_list()[self.t+1]
                   )
        #
        if self.t==0:
            terminated=False
        else:
            # terminated = bool(Tint1 < 0.95*self.consigneList[self.t])
            terminated = bool(Tint1<0.95*self.data['consigne'].to_list()[self.t])

        if not terminated:
            # reward=1.0-0.8*gas_value/200-0.8*abs((Tint1-self.data['consigne'].to_list()[self.t]))
            reward=1-0.8*abs((Tint1-self.data['consigne'].to_list()[self.t]))
            self.t += 1
            self.previousTint.append(Tint1)
            self.continuousRewardserie+=1

        else:
            reward =0
            self.continuousRewardserie =0
            self.t=0
            self.previousTint = [self.data['Tint'].to_list()[0]]


        if self.t>200:
            terminated=True
            reward =0
            self.t = 0
            self.continuousRewardserie =0
            self.previousTint = [self.data['Tint'].to_list()[0]]

        return  np.array(self.state, dtype=np.float32), reward, terminated, False, {}


    def reset(self):
        self.t = 0
        self.previousTint = [self.data['Tint'].to_list()[0]]
        reward=0,
        terminated = False
        return np.array((self.data['Tint'].to_list()[0],
                         self.data['consigne'].to_list()[1],
                         self.data['Text'].to_list()[1]
                         ), dtype=np.float32)
