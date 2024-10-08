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

class heaterEnvRC(gym.Env[np.ndarray, Union[int, np.ndarray]]):

    def __init__(self):
        self.window_range=200

        self.data = pd.read_csv('C:\\Users\\a.lance\\PycharmProjects\\UA3412_\\IDEAL_home106\\home106_main_data_set_HR.csv',sep=";")
        # self.data=self.data[150:]
        self.data['consigne']=np.where((self.data['hour']>=8) & (self.data['hour']<=17),18,np.nan)
        self.data['consigne'] = np.where((self.data['hour'] >= 20), 15, self.data['consigne'])
        self.data['consigne'] = np.where((self.data['hour'] <= 5) & (self.data['hour'] <= 23), 15, self.data['consigne'])
        self.data['consigne'].interpolate(inplace=True)
        self.consigneList=self.data['consigne'].to_list()
        #
        # Energie cost
        self.data['energy_cost']=np.where((self.data['hour']>=0) & (self.data['hour']<=7),0.2,np.nan)
        self.data['energy_cost'] = np.where((self.data['hour'] >= 22) & (self.data['hour'] <= 23), 0.2, self.data['energy_cost'])
        self.data['energy_cost'] = np.where((self.data['hour'] >= 8) & (self.data['hour'] <= 21), 0.25,self.data['energy_cost'])

        self.R = 0.1
        self.C= 100
        self.gamma = math.exp(-1/(self.R*self.C))
        self.previousTint=[self.data['Tint'].to_list()[0]]
        self.Text=self.data['Text'].to_list()
        self.t=0
        self.state=(self.data['Tint'].to_list()[0],self.data['consigne'].to_list()[1],self.data['Text'].to_list()[1])


        low = np.array([0,0,0,], dtype=np.float32, )
        high = np.array([100,100,100],dtype=np.float32,)

        self.action_space = spaces.Discrete(51)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.continuousRewardserie=0
        # self.consigne=18

    def step(self,action):
        terminated=False
        gas_value=1000*action/50
        Tint1=self.previousTint[-1]*self.gamma+(self.R*gas_value+self.Text[self.t])*(1-self.gamma)
        self.state=(Tint1,
                    self.data['consigne'].to_list()[self.t+1],
                    self.data['Text'].to_list()[self.t+1],
                    # self.data['consigne'].to_list()[self.t + 2],
                    # self.data['Text'].to_list()[self.t + 2],
                    # self.data['consigne'].to_list()[self.t + 3],
                    # self.data['Text'].to_list()[self.t + 3],
                    # self.data['consigne'].to_list()[self.t + 4],
                    # self.data['Text'].to_list()[self.t + 4],
                    # self.data['consigne'].to_list()[self.t + 5],
                    # self.data['Text'].to_list()[self.t + 5]
                   )
        #


        # terminated = bool(Tint1 < 0.95 * self.consigneList[self.t] or Tint1 > 1.05 * self.consigneList[self.t] or self.t==200)
        terminated=bool(self.t==1000)
        if self.consigneList[self.t]==18:
            terminated=bool(Tint1 < 0.97 * self.consigneList[self.t])
        if not terminated:
            # reward=1.0-0.8*gas_value/200-0.8*abs((Tint1-self.data['consigne'].to_list()[self.t]))
            if self.data['consigne'].to_list()[self.t]==18:
                # reward=1-0.5*abs((Tint1-self.data['consigne'].to_list()[self.t]))
                reward=1*math.exp(-0.3*abs((Tint1-self.data['consigne'].to_list()[self.t])))
            else:
                reward=1*math.exp(-0.06*action)

            self.t += 1
            self.previousTint.append(Tint1)
            self.continuousRewardserie+=1

        else:
            reward =0
            self.reset()

        return  np.array(self.state, dtype=np.float32), reward, terminated, False,{}


    def reset(self):
        self.t = 0
        self.consigneList = self.data['consigne'].to_list()
        self.previousTint = [self.data['Tint'].to_list()[0]]
        self.continuousRewardserie = 0
        reward=0,
        terminated = False
        return np.array((self.data['Tint'].to_list()[0],
                         self.data['consigne'].to_list()[1],
                         self.data['Text'].to_list()[1],
                         # self.data['consigne'].to_list()[2],
                         # self.data['Text'].to_list()[2],
                         # self.data['consigne'].to_list()[3],
                         # self.data['Text'].to_list()[3],
                         # self.data['consigne'].to_list()[4],
                         # self.data['Text'].to_list()[4],
                         # self.data['consigne'].to_list()[5],
                         # self.data['Text'].to_list()[5]
                         ), dtype=np.float32)
