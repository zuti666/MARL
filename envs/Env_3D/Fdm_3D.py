# -*- coding: utf-8 -*-

"""
@author     : zuti
@software   : PyCharm
@file       : Fdm_3D.py
@time       : 2023/4/11 18:26
@desc       ：

"""
# -*- coding: utf-8 -*-


import numpy as np
from scipy.integrate import odeint

class Fdm3D():
    """
    将3维的每一个飞机都包装为一个对象，
    初始化飞机的编号，与状态
    set 函数设置
    输入飞机的控制量，返回新的状态 sendAction


    """

    def __init__(
            self,
            fdm_id=1,
            flight_n_move=6,  # 飞机的运动量状态个数
            fdm_pos_x=0,  # x
            fdm_pos_y=0,  # y
            fdm_pos_z=0,  # z
            fdm_v=20,  #  vel v
            fdm_gamma = 0,  #  gamma γ
            fdm_varphi = 0, #  φ varphi


            fdm_hp=1, #飞机的初始血量

            flight_name ='f16', #根据飞机类型设置机动性能

            flight_n_action= 3, #飞机的控制量个数

            time=1, n=10, # 仿真时间 与步长

            flight_mode=1,  # 0 for flight test

    ) -> None:

        #飞机编号
        self.fdm_id = fdm_id


        # 飞机初始运动状态
        self.size_move_feats = flight_n_move
        self.fdm_pos_x = fdm_pos_x
        self.fdm_pos_y = fdm_pos_y
        self.fdm_pos_z = fdm_pos_z
        self.fdm_v = fdm_v
        self.fdm_gamma = fdm_gamma   # gamma γ
        self.fdm_varphi = fdm_varphi   # varphi φ


        #飞机血量
        self.fdm_hp = fdm_hp
        #飞机上一轮的动作
        self.n_action = flight_n_action
        self.last_action = np.zeros(self.n_action)

        #飞机性能
        if flight_name == 'f16':
            # 飞机的性能
            self.attack_range = 5000
            # 飞机的视野范围
            self.sight_range = 20000
            # 飞机的速度范围
            self.vel_range = 4




            # 飞机的航行范围 -x,x -y，y
            self.x_range = 20000
            self.y_range = 20000
            self.z_range = 18000

        #仿真设置
            # 一次仿真的时间间隔 与仿真次数
        self.time = time
        self.n = n




    def get_state_move_feats(self):
        """
        获得智能体 运动特征： 位置x,y+速度+角度
        :return:
        """
        move_feats = np.array([self.fdm_pos_x,self.fdm_pos_y,self.fdm_pos_z,self.fdm_v,self.fdm_γ,self.fdm_φ])
        return move_feats

    @staticmethod
    def dmove2(x_input, t, control):
        g = 9.81  # 重力加速度
        velocity, gamma, fai = x_input
        nx, nz, gunzhuan = control

        velocity_ = g * (nx - np.sin(gamma))  # # 米每秒
        gamma_ = (g / velocity) * (nz * np.cos(gunzhuan) - np.cos(gamma))  # 米每秒
        fai_ = g * nz * np.sin(gunzhuan) / (velocity * np.cos(gamma))

        return np.array([velocity_, gamma_, fai_])



    def send_action(self,control):
        """
        对飞机对象发送指令，获得新的状态

        :param control:
        :return:
        """

        time = self.time
        n = self.n

        t = np.linspace(0, time, n)  # 仿真步长
        dt = t[1] - t[0]
        state_list = np.zeros((n, self.size_move_feats))  # 轨迹长度
        state_list[0] = np.array([self.fdm_pos_x,self.fdm_pos_y,self.fdm_pos_z,self.fdm_v,self.fdm_gamma,self.fdm_varphi])  # 轨迹列表第一个元素为初始状态
        x, y, z,velocity, gamma, varphi = state_list[0]

        # 获得控制量 加速度和角加速度 ，并限制范围


        for k in range(1, n):

            tspan = [t[k - 1], t[k]]

            st = odeint(self.dmove2, (velocity, gamma, varphi), tspan, args=([ control[0], control[1], control[2] ],)) #求梯度

            velocity, gamma, varphi = st[1, :]

            dx = velocity * np.cos(gamma) * np.sin(varphi) * dt
            dy = velocity * np.cos(gamma) * np.cos(varphi) * dt
            dz = velocity * np.sin(gamma) * dt

            x = x + dx
            state_list[k, 0] = x
            y = y + dy
            state_list[k, 1] = y
            z = z + dz
            state_list[k, 2] = z

            state_list[k, 3] = velocity
            state_list[k, 4] = gamma
            state_list[k, 5] = varphi

        next_state = state_list[-1, :]  # 轨迹最后一个值结果
        #更新飞机对象的结果
        self.fdm_pos_x=next_state[0]
        self.fdm_pos_y = next_state[1]
        self.fdm_pos_z = next_state[2]
        self.fdm_v = next_state[3]
        self.fdm_γ = next_state[4]
        self.fdm_φ =  next_state[5]

        #
        self.last_action=control


        return next_state



