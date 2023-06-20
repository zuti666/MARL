# -*- coding: utf-8 -*-

"""
@author     : zuti
@software   : PyCharm
@file       : test_2d.py
@time       : 2023/4/9 11:27
@desc       ：

"""
import gym
import numpy as np

from envs.Env_2D.Env_2D_2V2 import  EnvCore





#测试3对2场景
env = EnvCore(n_agents=2, n_enemies=2)
env.reset()
action_0 = np.array([0,0])
action_1 = np.array([0,0])
action_2 = np.array([0,0])
action = [action_0,action_1,action_2]
for i in range(100):
    [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info] =env.step(action)

print('--------------')

