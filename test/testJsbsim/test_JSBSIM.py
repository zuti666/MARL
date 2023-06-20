# -*- coding: utf-8 -*-

"""
@author     : zuti
@software   : PyCharm
@file       : test_JSBSIM.py
@time       : 17/06/2023 23:51
@desc       ：

"""
import numpy as np

from envs.JsbSimEnv.JsbSimEnv import EnvCore

env = EnvCore()
for t in  env.state_before:
    print(f'{t}')



action_0 = np.array([0,0,0,0])
action_1 = [0,0,0,0]
action = [action_0,action_1]

for i in range(100):
    [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info] =env.step(action)
    print(f'obs{sub_agent_obs}')
    print(sub_agent_reward)

