# -*- coding: utf-8 -*-

"""
@author     : zuti
@software   : PyCharm
@file       : test_fdm3D.py
@time       : 28/04/2023 21:31
@desc       ：

测试物理仿真器

"""
import numpy as np
from matplotlib import pyplot as plt

from envs.Env_3D.Fdm_3D import  Fdm3D

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 12})


init_pos_aly_0 = [0,0,86, 2, 0,0]
init_pos_aly_1 = [0,15,86, 2, 0,0]
init_pos_enemy_0 = [110,0,82, 1,  np.pi,0]
init_pos_enemy_1 = [110,50,80, 1, np.pi,0]
init_pos_enemy_2 = [110, 100, 80, 1, np.pi, 0]

init_state = np.ones([4, 6])
init_state[0] = init_pos_aly_0
init_state[1] = init_pos_aly_1
init_state[2] = init_pos_enemy_0
init_state[3] = init_pos_enemy_1
#init_state[4] = init_pos_enemy_2
#init_state = np.array([init_pos_aly_0, init_pos_aly_1, init_pos_enemy_0, init_pos_enemy_1]).flatten()


        # 每一个飞机都是一个飞机动力模拟结构
flight = []
for i in range(0, 4):
    fdm_id = i
    fdm_pos_x = init_state[i][0]
    fdm_pos_y = init_state[i][1]
    fdm_pos_z = init_state[i][2]
    fdm_v = init_state[i][3]
    fdm_gamma = init_state[i][4]
    fdm_varphi = init_state[i][5]
    flight1 = Fdm3D(fdm_id=fdm_id, fdm_pos_x=fdm_pos_x, fdm_pos_y=fdm_pos_y, fdm_pos_z=fdm_pos_z, fdm_v=fdm_v,
                    fdm_gamma=fdm_gamma, fdm_varphi=fdm_varphi)
    flight.append(flight1)

control = np.array([0,0,0])
f1 = flight[0]
state = f1.send_action(control)

print(state)

state_list = []
state_list.append(state)
for i  in range(100):
    control = np.array([0, 0, 0])

    state = f1.send_action(control)
    state_list.append(state)

state_list = np.array(state_list)
#绘制图像
fig = plt.figure()
ax1 = fig.add_subplot(221,projection='3d')
ax1.plot(state_list[:, 0], state_list[:, 1], state_list[:, 2])
ax1.set_title('trajectory 轨迹')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

ax2 = fig.add_subplot(222)
ax2.plot(state_list[:,3])
ax2.set_title('velocity 速度')

ax3 = fig.add_subplot(223)
ax3.plot(state_list[:,4])
ax3.set_title('gamma 航迹倾角')

ax3 = fig.add_subplot(224)
ax3.plot(state_list[:,5])
ax3.set_title('varphi  航向角')

#plt.savefig('test.jpg')
plt.show()
