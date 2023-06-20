# -*- coding: utf-8 -*-

"""
@author     : zuti
@software   : PyCharm
@file       : test_3d.py
@time       : 28/04/2023 21:24
@desc       ：

"""

import numpy as np

from envs.Env_3D.Env_3D import  EnvCore

env = EnvCore(n_agents=2, n_enemies=2)
env.reset()
action_0 = np.array([0,0,0])
action_1 = np.array([0,0,0])
action = [action_0,action_1]

for i in range(100):
    [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info] =env.step(action)

print('--------------')