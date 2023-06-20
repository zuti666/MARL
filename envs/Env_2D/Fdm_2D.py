# -*- coding: utf-8 -*-

"""
@author     : zuti
@software   : PyCharm
@file       : Fdm_2D.py
@time       : 2023/4/9 9:27
@desc       ：

"""
import numpy as np


class Fdm2D():
    """
    将2维的每一个飞机都包装为一个对象，
    初始化飞机的编号，与状态
    set 函数设置
    输入飞机的控制量，返回新的状态 sendAction
    并且通过get函数获得飞机的各种状态参数

    """

    def __init__(
            self,
            fdm_id=1,

            flight_n_move=4,  # 飞机的运动量状态个数
            fdm_pos_x=0,  # x
            fdm_pos_y=0,  # y
            fdm_v=20,  #  vel
            fdm_theta=0,  # theta

            fdm_hp=1, #飞机的初始血量

            flight_name ='f16', #根据飞机类型设置机动性能

            flight_n_action= 2, #飞机的控制量个数

            time=1, n=10, # 仿真时间 与步长

            flight_mode=1,  # 0 for flight test

    ) -> None:

        #飞机编号
        self.fdm_id = fdm_id


        # 飞机初始运动状态
        self.size_move_feats = flight_n_move
        self.fdm_pos_x = fdm_pos_x
        self.fdm_pos_y = fdm_pos_y
        self.fdm_v = fdm_v
        self.fdm_theta = fdm_theta

        #飞机血量
        self.fdm_hp = fdm_hp
        #飞机上一轮的动作
        self.n_action = flight_n_action
        self.last_action = np.zeros(self.n_action)

        #飞机性能
        if flight_name == 'f16':
            # 飞机的性能
            self.attack_range = 100
            # 飞机的视野范围
            self.sight_range = 200
            # 飞机的速度范围
            self.vel_range = 6
            # 飞机的加速度范围
            self.a_range = 1
            # 飞机的角加速度范围
            self.omega_range = 1
            # 飞机的航行范围 -x,x -y，y
            self.x_range = 600
            self.y_range = 600

        #仿真设置
            # 一次仿真的时间间隔 与仿真次数
        self.time = time
        self.n = n




    def get_state_move_feats(self):
        """
        获得智能体 运动特征： 位置x,y+速度+角度
        :return:
        """
        move_feats = np.array([self.fdm_pos_x,self.fdm_pos_y,self.fdm_v,self.fdm_theta])
        return move_feats



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
        state_list[0] = np.array([self.fdm_pos_x,self.fdm_pos_y,self.fdm_v,self.fdm_theta])  # 轨迹列表第一个元素为初始状态
        x, y, velocity, theta = state_list[0]

        # 获得控制量 加速度和角加速度 ，并限制范围
        a = np.clip(control[0], -self.a_range, self.a_range)
        omega = np.clip(control[1], -self.omega_range, self.omega_range)

        for k in range(1, n):


            # 更新计算速度，角度 并限制范围
            velocity = np.clip(velocity + a * dt, 0, self.vel_range)
            theta = np.clip(theta + omega * dt, -np.pi, np.pi - 0.001)

            dx = velocity * np.cos(theta) * dt
            dy = velocity * np.sin(theta) * dt

            x = x + dx
            state_list[k, 0] = x
            y = y + dy
            state_list[k, 1] = y

            state_list[k, 2] = velocity
            state_list[k, 3] = theta

        next_state = state_list[-1, :]  # 轨迹最后一个值结果
        #更新飞机对象的结果
        self.fdm_pos_x=next_state[0]
        self.fdm_pos_y = next_state[1]
        self.fdm_v = next_state[2]
        self.fdm_theta = next_state[3]
        #
        self.last_action=control


        return next_state


