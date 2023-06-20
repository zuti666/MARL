# -*- coding: utf-8 -*-

"""
@author     : zuti
@software   : PyCharm
@file       : dogFilghtEnv.py
@time       : 2023/4/8 22:16
@desc       ：

"""

import os
import random
import re
import sys
import time
import warnings

import gym
import numpy as np
from gym import Env
from gym.spaces import Box, Discrete

#调用dogfight中的通信端口，使用其中的函数进行获取数据与发送数据
from gym.envs.dogfightEnv.dogfight_sandbox_hg2.network_client_example import dogfight_client as df

sys.path.append('gym.envs.dogfightEnv.dogfight_sandbox_hg2.network_client_example/')


# try:

print(os.getcwd())
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class DogfightEnv2v2(Env):

    def __init__(
            self,
            n_agents=2, n_enemies=2,  # 飞机数量
            n_actions=3,  # 每个飞机的动作
            rendering=True,
    ) -> None:
        """
        构造函数，进行初始化

        :param host: 通信地址
        :param port: 端口
        :param plane_slot: 我方飞机编号
        :param enemy_slot: 敌方飞机编号
        :param missile_slot: 导弹编号
        :param rendering: 是否进行渲染，即可视化
        """
        print("gym dogfightEnv_2v2")




        #设置通信地址与端口
        self.host = '10.168.62.25'
        self.port = '50888'
        self.nof = 0
        # 设置是否显示
        self.rendering = rendering
        # 可视化设置，是否进行渲染，当参数rendering为True，即进行渲染，否则默认不进行渲染
        if self.rendering:
            df.set_renderless_mode(True)
        df.set_renderless_mode(False)



        try:
            df.get_planes_list()
        except:
            print('Run for the first time')
            df.connect(self.host, int(self.port))
            time.sleep(2)
        #
        self.plane_slot = [0, 1]
        self.enemy_slot = [2, 3]
        #self.missile_slot = [1, 2]



        # 飞机进行初始化，按着敌方和我方飞机队列获取飞机列表
        print('init')


        planes = df.get_planes_list()  # 从仿真环境获取飞机列表
        print(f'飞机列表')


        #df.disable_log(False)

        #print(f'我方飞机编号{plane_slot},敌方飞机编号{enemy_slot}')
        self.Plane_ID_ally_list = []
        self.Plane_ID_oppo_list = []

        self.Plane_ID_ally_0 = planes[0]
        self.Plane_ID_ally_1 = planes[1]
        self.Plane_ID_oppo_0 = planes[2]
        self.Plane_ID_oppo_1 = planes[3]

        self.Plane_ID_ally_list.append(self.Plane_ID_ally_0)
        self.Plane_ID_ally_list.append(self.Plane_ID_ally_1)
        self.Plane_ID_oppo_list.append(self.Plane_ID_oppo_0)
        self.Plane_ID_oppo_list.append(self.Plane_ID_oppo_1)


        df.set_client_update_mode(True)



        # 初始化飞机
        for t in planes:
            df.reset_machine(t)  #启动机器 # 恢复初始状态

            df.set_health(t,1)   #设置健康水平
            df.activate_post_combustion(t)  #激活燃料



        #设置飞机 位置与姿态

        # 形成对头态势
        #三个参数,分别是 y z x    俯仰 偏航 滚转

        # pos_rand = np.random.normal(loc=0, scale=200)

        df.reset_machine_matrix(self.Plane_ID_ally_0,0,10000,0,0,0,0)
        df.reset_machine_matrix(self.Plane_ID_ally_1,5000,  10000, 0, 0, 0, 0)

        df.reset_machine_matrix(self.Plane_ID_oppo_0,  0,10200,  5000,0, np.pi,0 )
        df.reset_machine_matrix(self.Plane_ID_oppo_1, 5000, 10200, 5000, np.pi, 0, np.pi)



        #设置目标 machine_id,  target_id
        df.set_target_id(self.Plane_ID_ally_0,self.Plane_ID_oppo_0)
        df.set_target_id(self.Plane_ID_ally_1,self.Plane_ID_oppo_1)


        # 设置推力、线速度
        df.set_plane_thrust(self.Plane_ID_ally_0, 0.81)
        df.set_plane_linear_speed(self.Plane_ID_ally_0, 400)


        df.set_plane_thrust(self.Plane_ID_ally_1, 0.82)
        df.set_plane_linear_speed(self.Plane_ID_ally_1, 400)


        df.set_plane_thrust(self.Plane_ID_oppo_0, 0.83)
        df.set_plane_linear_speed(self.Plane_ID_oppo_0, 400)


        df.set_plane_thrust(self.Plane_ID_oppo_1, 0.84)
        df.set_plane_linear_speed(self.Plane_ID_oppo_1, 400)



        # 收回起落架
        df.retract_gear(self.Plane_ID_ally_0)
        df.retract_gear(self.Plane_ID_ally_1)
        df.retract_gear(self.Plane_ID_oppo_0)
        df.retract_gear(self.Plane_ID_oppo_1)

        #激活AI
        # df.activate_IA(self.Plane_ID_oppo_0)

        # 设定攻击目标
        # 敌方无人机使用IA算法
        #df.activate_IA(self.Plane_ID_oppo)

        #我方的导弹编号
        # missiles = df.get_machine_missiles_list(self.Plane_ID_ally)
        # self.missileID = missiles[self.missile_slot]


        # 更新场景
        df.update_scene()
        # 飞机的动作空间




        # self.action_space = Box(
        #     low=np.array([
        #         0,  # 推力
        #         -1,  # Pitch 俯仰角
        #         -1,  # Roll 翻滚角
        #         -1,  # Yaw 偏航角
        #
        #         0,  # 推力
        #         -1,  # Pitch 俯仰角
        #         -1,  # Roll 翻滚角
        #         -1,  # Yaw 偏航角
        #     ]),
        #     high=np.array([
        #         1,
        #         1,
        #         1,
        #         1,
        #
        #         1,
        #         1,
        #         1,
        #         1,
        #
        #     ]),
        # )
        #
        # # 飞机的观测空间
        #
        # self.observation_space = Box(
        #     low=np.array([  # simple normalized
        #
        #         # 我方两架飞机
        #
        #         -300,  # x / 100  横坐标
        #         -300,  # y / 100  纵坐标
        #         -1,  # z / 100    z轴坐标
        #         -360,  # heading  （h-180） *2   heading角度
        #         -360,  # pitch_attitude * 4
        #         -360,  # roll_attitude * 4
        #         0,  # linear_speed
        #         -300,  # x / 100  横坐标
        #         -300,  # y / 100  纵坐标
        #         -1,  # z / 50    z轴坐标
        #         -360,  # heading    heading角度
        #         -360,  # pitch_attitude * 4
        #         -360,  # roll_attitude * 4
        #         0,  # linear_speed  总速度 单位 km/h
        #
        #         # 敌方两架飞机
        #
        #         -300,  # x / 100  横坐标
        #         -300,  # y / 100  纵坐标
        #         -1,  # z / 100    z轴坐标
        #         -360,  # heading  （h-180） *2   heading角度
        #         -360,  # pitch_attitude * 4
        #         -360,  # roll_attitude * 4
        #         0,  # linear_speed
        #         -300,  # x / 100  横坐标
        #         -300,  # y / 100  纵坐标
        #         -1,  # z / 50    z轴坐标
        #         -360,  # heading    heading角度
        #         -360,  # pitch_attitude * 4
        #         -360,  # roll_attitude * 4
        #         0,  # linear_speed
        #
        #     ]),
        #     high=np.array([
        #
        #         # 我方两架飞机
        #         300,  # x / 100
        #         300,  # y / 100
        #         200,  # z / 100
        #         360,  # heading
        #         360,  # pitch_attitude * 4
        #         360,  # roll_attitude * 4
        #         400,  # 速度
        #         300,  # x / 100
        #         300,  # y / 100
        #         200,  # z / 100
        #         360,  # heading
        #         360,  # pitch_attitude * 4
        #         360,  # roll_attitude * 4
        #         2000,  # 最大速度x
        #
        #         # 敌方两架飞机
        #         300,  # x / 100
        #         300,  # y / 100
        #         200,  # z / 100
        #         360,  # heading
        #         360,  # pitch_attitude * 4
        #         360,  # roll_attitude * 4
        #         400,  # 速度
        #         300,  # x / 100
        #         300,  # y / 100
        #         200,  # z / 100
        #         360,  # heading
        #         360,  # pitch_attitude * 4
        #         360,  # roll_attitude * 4
        #         2000,  # 最大速度x
        #
        #     ]),
        #     dtype=np.float64
        # )


    def state_normalized(self,ob):
        """
        状态归一化
        :return:
        """

        # 状态归一化
        ob[0] = ob[0] / 100.0
        ob[1] = ob[1] / 100.0
        ob[2] = ob[2] / 100.0
        ob[3] = (ob[3] -180.0)*2
        ob[4] = ob[4] * 4.0
        ob[5] = ob[5] * 4.0
        ob[6] = ob[6]





        i = 6
        ob[0 + i] = ob[0 + i] / 100.0
        ob[1 + i] = ob[1 + i] / 100.0
        ob[2 + i] = ob[2 + i] / 50.0
        ob[3 + i] = (ob[3 + i] -180.0)*2
        ob[4 + i] = ob[4 + i] * 4.0
        ob[5 + i] = ob[5 + i] * 4.0
        ob[6+i]  = ob[6+i]


        i = 12
        ob[0 + i] = ob[0 + i] / 100.0
        ob[1 + i] = ob[1 + i] / 100.0
        ob[2 + i] = ob[2 + i] / 50.0
        ob[3 + i] = (ob[3 + i] - 180.0) * 2
        ob[4 + i] = ob[4 + i] * 4.0
        ob[5 + i] = ob[5 + i] * 4.0
        ob[6 + i] = ob[6 + i]


        i = 18
        ob[0 + i] = ob[0 + i] / 100.0
        ob[1 + i] = ob[1 + i] / 100.0
        ob[2 + i] = ob[2 + i] / 50.0
        ob[3 + i] = (ob[3 + i] - 180.0) * 2
        ob[4 + i] = ob[4 + i] * 4.0
        ob[5 + i] = ob[5 + i] * 4.0
        ob[6 + i] = ob[6 + i]



        return ob

    def get_init_obv_list(self):
        """
        获得初始场景信息

        :return:  返回初始场景信息
        """
        print(f'飞机总列表{df.get_planes_list()}')
        print(f'当前我方飞机编号{self.Plane_ID_ally_list}')
        print(f'当前敌方飞机编号{self.Plane_ID_oppo_list}')
        print(f'我方飞机状态0{df.get_plane_state(self.Plane_ID_ally_0)}')
        print(f'我方飞机状态1{df.get_plane_state(self.Plane_ID_ally_1)}')
        print(f'敌方飞机状态0{df.get_plane_state(self.Plane_ID_oppo_0)}')
        print(f'敌方飞机状态1{df.get_plane_state(self.Plane_ID_oppo_1)}')



    def get_obv_list(self):
        """
        获得初始场景信息

        :return:  返回初始场景信息
        """

        print(f'当前我方飞机编号{self.Plane_ID_ally_list}')
        print(f'当前敌方飞机编号{self.Plane_ID_oppo_list}')
        print(f'我方飞机状态0{df.get_plane_state(self.Plane_ID_ally_0)}')
        print(f'我方飞机状态1{df.get_plane_state(self.Plane_ID_ally_1)}')
        print(f'敌方飞机状态0{df.get_plane_state(self.Plane_ID_oppo_0)}')
        print(f'敌方飞机状态1{df.get_plane_state(self.Plane_ID_oppo_1)}')





#获取红蓝双方全局信息
    def get_state_list(self):
        """
            返回当前态势

            :return:
        """
        plane_state_0 = df.get_plane_state(self.Plane_ID_ally_0)
        plane_state_1 = df.get_plane_state(self.Plane_ID_ally_1)
        enemy_state_0 = df.get_plane_state(self.Plane_ID_oppo_0)
        enemy_state_1 = df.get_plane_state(self.Plane_ID_oppo_1)

        state = [plane_state_0,plane_state_1, enemy_state_0 ,enemy_state_1]
        return state

    def get_position_list(self):
        """
        获取当前所有飞机的位置信息

            :param self:
            :return:
        """
        plane_position_0 = np.array(df.get_plane_state(self.Plane_ID_ally_0)['position'])
        enemy_position_0 = np.array(df.get_plane_state(self.Plane_ID_oppo_0)['position'])
        plane_position_1 = np.array(df.get_plane_state(self.Plane_ID_ally_1)['position'])
        enemy_position_1 = np.array(df.get_plane_state(self.Plane_ID_oppo_1)['position'])

        position = np.array([plane_position_0,plane_position_1,enemy_position_0,enemy_position_1])
        return position

    def get_moveVector_list(self):
        """
        获取当前所有飞机的速度信息
        :param self:
        :return:
        """
        plane_moveVector_0 = np.array(df.get_plane_state(self.Plane_ID_ally_0)['move_vector'])
        enemy_moveVector_0 = np.array(df.get_plane_state(self.Plane_ID_oppo_0)['move_vector'])

        plane_moveVector_1 = np.array(df.get_plane_state(self.Plane_ID_ally_1)['move_vector'])
        enemy_moveVector_1 = np.array(df.get_plane_state(self.Plane_ID_oppo_1)['move_vector'])


        moveVector = np.array([plane_moveVector_0,plane_moveVector_1,enemy_moveVector_0,enemy_moveVector_1])


        return moveVector


    def get_distance_list(self):
        """
        计算当前两机距离

        :return:
        """
        plane_position_0 = np.array(df.get_plane_state(self.Plane_ID_ally_0)['position'])
        enemy_position_0 = np.array(df.get_plane_state(self.Plane_ID_oppo_0)['position'])

        plane_position_1 = np.array(df.get_plane_state(self.Plane_ID_ally_1)['position'])
        enemy_position_1 = np.array(df.get_plane_state(self.Plane_ID_oppo_1)['position'])




        distance_0 = np.linalg.norm(plane_position_0 - plane_position_1,ord=2)
        distance_1 = np.linalg.norm(enemy_position_0 - enemy_position_1, ord=2)

        distance_2 = np.linalg.norm(plane_position_0 - enemy_position_0, ord=2)
        distance_3 = np.linalg.norm(plane_position_0 - enemy_position_1, ord=2)
        distance_4 = np.linalg.norm(plane_position_1 - enemy_position_0, ord=2)
        distance_5 = np.linalg.norm(plane_position_1 - enemy_position_1, ord=2)

        distance_list = np.array([distance_0,distance_1,distance_2,distance_3,distance_4,distance_5])



        return distance_list

    def get_angle_radian(self,position_vector,velocity_vector):
        """
        根据位置矢量和速度矢量
        计算我方与敌方形成的夹角
        :return:
        """
        angle_radian =  np.arccos(
                    np.dot(position_vector,velocity_vector) /(np.linalg.norm(position_vector) * np.linalg.norm(velocity_vector))
                  ) #夹角的余弦值 #夹角弧度制

        return  angle_radian


    def get_ATA_list(self):
        """
        计算我方两架无人机与敌方分别形成的ATA ，返回结果为4个  位置连线与我方速度形成的夹角
        :return:
        """
        plane_position_0 = np.array(df.get_plane_state(self.Plane_ID_ally_0)['position'])
        enemy_position_0 = np.array(df.get_plane_state(self.Plane_ID_oppo_0)['position'])
        plane_position_1 = np.array(df.get_plane_state(self.Plane_ID_ally_1)['position'])
        enemy_position_1 = np.array(df.get_plane_state(self.Plane_ID_oppo_1)['position'])

        plane_vecltory_0 = np.array(df.get_plane_state(self.Plane_ID_ally_0)['move_vector'])
        plane_vecltory_1 = np.array(df.get_plane_state(self.Plane_ID_ally_0)['move_vector'])



        position_vector_0 =   enemy_position_0 - plane_position_0
        position_vector_1 = enemy_position_1 - plane_position_0
        position_vector_2 = enemy_position_0 - plane_position_1
        position_vector_3 = enemy_position_0 - plane_position_1

        ATA_0 = self.get_angle_radian(position_vector_0,plane_vecltory_0)
        ATA_1 = self.get_angle_radian(position_vector_1, plane_vecltory_0)
        ATA_2 = self.get_angle_radian(position_vector_2, plane_vecltory_1)
        ATA_3 = self.get_angle_radian(position_vector_3, plane_vecltory_1)

        ATA_list = np.array([ATA_0,ATA_1,ATA_2,ATA_3])

        return ATA_list




    # def get_AA(self):
    #     """
    #     计算我方与敌方形成的ATA夹角
    #     :return:
    #     """
    #
    #     red_position = np.array(df.get_plane_state(self.Plane_ID_ally)['position'])
    #     blue_position = np.array(df.get_plane_state(self.Plane_ID_oppo)['position'])
    #     blue_vecltory = np.array(df.get_plane_state(self.Plane_ID_oppo)['move_vector'])
    #
    #
    #
    #
    #
    #     L = blue_position - red_position
    #     cosATA = np.dot(L, blue_vecltory) / (np.linalg.norm(L) * np.linalg.norm(blue_vecltory))  # 夹角的余弦值
    #
    #     AA_radian = np.arccos(cosATA)  # 夹角弧度制
    #
    #     return AA_radian



    def get_height_list(self):
        """
        返回我方与敌方的高度差,结果为矢量，负号表示我方在下
        :return:
        """

        plane_height_0 = np.array(df.get_plane_state(self.Plane_ID_ally_0)['position'])[2]
        enemy_height_0 = np.array(df.get_plane_state(self.Plane_ID_oppo_0)['position'])[2]

        plane_height_1 = np.array(df.get_plane_state(self.Plane_ID_ally_1)['position'])[2]
        enemy_height_1 = np.array(df.get_plane_state(self.Plane_ID_oppo_1)['position'])[2]

        height_0 = plane_height_0 - plane_height_1
        height_1 = enemy_height_0 - enemy_height_1

        height_2 = plane_height_0 - enemy_height_0
        height_3 = plane_height_0 - enemy_height_1
        height_4 = plane_height_1 - enemy_height_0
        height_5 = plane_height_1 - enemy_height_1

        height_list = np.array([height_0, height_1, height_2, height_3, height_4, height_5])


        return height_list



    def get_hp_list(self):

        plane_hp_0 =  df.get_health(self.Plane_ID_ally_0)['health_level']
        plane_hp_1 = df.get_health(self.Plane_ID_ally_1)['health_level']
        enemy_hp_0 = df.get_health(self.Plane_ID_oppo_0)['health_level']
        enemy_hp_1 = df.get_health(self.Plane_ID_oppo_1)['health_level']

        hp_list = np.array([plane_hp_0, plane_hp_1, enemy_hp_0, enemy_hp_1])


        return hp_list


    def get_obs_agent(self,plane_id):
        """

        Returns observation for agent_id. The observation is composed of:

           - agent movement features (where it can move to, height information and pathing grid)
           - enemy features (available_to_attack, health, relative_x, relative_y, shield, unit_type)
           - ally features (visible, distance, relative_x, relative_y, shield, unit_type)
           - agent unit features (health, shield, unit_type)

           All of this information is flattened and concatenated into a list,
           in the aforementioned order. To know the sizes of each of the
           features inside the final list of features, take a look at the
           functions ``get_obs_move_feats_size()``,
           ``get_obs_enemy_feats_size()``, ``get_obs_ally_feats_size()`` and
           ``get_obs_own_feats_size()``.

           The size of the observation vector may vary, depending on the
           environment configuration and type of units present in the map.
           For instance, non-Protoss units will not have shields, movement
           features may or may not include terrain height and pathing grid,
           unit_type is not included if there is only one type of unit in the
           map etc.).

           NOTE: Agents should have access only to their local observations
           during decentralised execution.

        获得局部观测空间 ，以向量的形式
        包含以下部分
         agent 的动作特征
         enemy 的 特征
         ally 联盟的特征   (visible, distance, relative_x, relative_y, shield, unit_type)\
         agent 的战斗状态特征

        :param plane_id:
        :return:
        """

    def success(self):
        """
        到达作战目标,判断结束 ,双机以一机到达重点为结束条件
        :return:
        """

        Flag_success = False

        position = df.get_plane_state(self.Plane_ID_ally)['position']
        distance = self.get_distance()
        HP_oppo = df.get_plane_state(self.Plane_ID_oppo)['health_level']
        #destroyed_oppo = df.get_plane_state(self.Plane_ID_oppo)['destroyed']




        if distance < 1000:
            Flag_success = True

        if HP_oppo < 0.5:
            Flag_success

        # if destroyed_oppo:
        #     Flag_success




        return  Flag_success


    def terminate(self):
        """
        出现意外,或者飞出区间
        设置终止条件
        :return:
        """



        Flag_terminate = False

        #飞机碰撞后，终止
        destoryed  = df.get_plane_state(self.Plane_ID_ally)['destroyed']
        if destoryed:
            Flag_terminate = True

        crashed = df.get_plane_state(self.Plane_ID_ally)['crashed']
        if crashed:
            Flag_terminate = True

        health_level = df.get_plane_state(self.Plane_ID_ally)['health_level']
        if health_level < 0.2 :
            Flag_terminate = True


         #当飞机过低，终止

        if df.get_plane_state(self.Plane_ID_ally)['position'][1]  <100  :
            Flag_terminate = True





        return Flag_terminate




    def setProperty(
            self,
            prop,
            value,
    ):
        if prop == 'plane':
            self.plane_slot = value
        elif prop == 'enemy':
            self.enemy_slot = value
        elif prop == 'missile':
            self.missile_slot = value
        else:
            raise Exception("Property {} doesn't exist!".format(prop))


    def sendAction(
            self,
            action
            #actionType=None,
    ):
        """
        设置动作，只调节飞机的推力，和偏航控制

        :param action:
        :param actionType:
        :return:
        """

        #设定动作 第一架飞机
        df.set_plane_thrust(self.Plane_ID_ally_0, float(action[0]))
        df.set_plane_pitch(self.Plane_ID_ally_0, float(action[1]))
        df.set_plane_roll(self.Plane_ID_ally_0, float(action[2]))
        df.set_plane_yaw(self.Plane_ID_ally_0, float(action[3]))
        #




        #设定动作 第二架飞机

        df.set_plane_thrust(self.Plane_ID_ally_1, float(action[4]))
        df.set_plane_pitch(self.Plane_ID_ally_1, float(action[5]))
        df.set_plane_roll(self.Plane_ID_ally_1, float(action[6]))
        df.set_plane_yaw(self.Plane_ID_ally_1, float(action[7]))











    def count_reward(self,before_state):
        """
        奖励函数
        :return:
        """

        #获取上一状态
        print("计算奖励 上一状态")
        print(before_state)

        before_state_red = before_state[0]
        before_state_blue = before_state[1]
        before_red_position = np.array(before_state_red['position'])
        before_blue_position = np.array(before_state_blue['position'])
        before_red_vecltory = np.array(before_state_red['move_vector'])
        before_blue_vecltory = np.array(before_state_blue['move_vector'])
        #由上一状态来计算距离，高度差等
        before_distance = np.linalg.norm(before_red_position-before_blue_position)  #前一状态的距离
        before_height = before_red_position[2] - before_blue_position[2]  #前一状态的高度差，正负表示顺序先后
        before_ATA_radian = np.arccos(np.dot(before_blue_position - before_red_position, before_red_vecltory) / (np.linalg.norm(before_blue_position - before_red_position) * np.linalg.norm(before_red_vecltory)))  # 前一状态的夹角弧度制




        #获取当前态势
        current_state = self.get_state()

        print("当前态势")
        print(current_state)


        red_state = current_state[0]
        blue_state = current_state[1]
        red_position = np.array([red_state['position'][0], red_state['position'][2], red_state['position'][1]])
        distance = self.get_distance()
        height = self.getHeight()
        ATA = self.get_ATA()
        AA = self.get_AA()







        #奖励函数设置
        reward = 0.01

        # 判断是否终止
        if self.terminate():
            reward -= 50

        #终止条件奖励
        #当距离接近，成功 获取奖赏
        if distance < 1200:
            reward += 50


        #飞行范围奖励
        #高度限制奖励
        if  red_position[1] >500:
            reward += 0.1
        else:
            reward -= 5




        #距离奖励
        if  distance > before_distance:
            reward += 10
        if distance < 10000:
            reward +=  2


        #高度差奖励
        if   -1000< height < 1000 :
            reward +=  0.5



        #角度奖励

        if  0<=ATA <= np.pi /2:
            reward += 20
        if  0<= AA  <= np.pi/2:
            reward += 20



        #整体态势奖励




        return reward

    def send_action_oppo(self):
        """
        蓝方采取随机策略进行飞机

        :return:
        """
        trust = np.random.normal(0.8, 0.1)
        pitch = np.random.normal(0, 0.1)
        roll = np.random.normal(0, 0.1)
        yaw = np.random.normal(0, 0.1)





        df.set_plane_thrust(self.Plane_ID_oppo_0, trust)
        df.set_plane_pitch(self.Plane_ID_oppo_0, pitch)
        df.set_plane_roll(self.Plane_ID_oppo_0, roll)
        df.set_plane_yaw(self.Plane_ID_oppo_0, yaw)
        df.set_plane_thrust(self.Plane_ID_oppo_1, trust)
        df.set_plane_pitch(self.Plane_ID_oppo_1, pitch)
        df.set_plane_roll(self.Plane_ID_oppo_1, roll)
        df.set_plane_yaw(self.Plane_ID_oppo_1, yaw)








    def step(self, action):

        #获取更新之前的状态
        before_state = self.get_state()



        #更新环境
        #
        self.sendAction(action)  # 发送红方的动作

        #设置蓝方的动作
        self.send_action_oppo()






        df.update_scene()  # 场景更新
        self.nof += 1



        reward = self.count_reward(before_state)  #根据更新的场景，奖励计算

        done = self.terminate() or self.success()   # 判断是否终止

        #输出向量
        ob = np.array([  # normalized

            #我方飞机位置，姿态，运动速度
            df.get_plane_state(self.Plane_ID_ally)['position'][0],  #横坐标
            df.get_plane_state(self.Plane_ID_ally)['position'][2] , #纵坐标
            df.get_plane_state(self.Plane_ID_ally)['position'][1] , #z轴坐标
            df.get_plane_state(self.Plane_ID_ally)['heading'],   #偏航角 heading yaw
            df.get_plane_state(self.Plane_ID_ally)['pitch_attitude'] , #俯仰角
            df.get_plane_state(self.Plane_ID_ally)['roll_attitude'] , #滚转角
            df.get_plane_state(self.Plane_ID_ally)['linear_speed'],  #总的速度


            #敌方飞机位置，姿态，运动速度
            df.get_plane_state(self.Plane_ID_oppo)['position'][0],  # 横坐标
            df.get_plane_state(self.Plane_ID_oppo)['position'][2],  # 纵坐标
            df.get_plane_state(self.Plane_ID_oppo)['position'][1],  # z轴坐标
            df.get_plane_state(self.Plane_ID_oppo)['heading'],  # 偏航角 heading yaw
            df.get_plane_state(self.Plane_ID_oppo)['pitch_attitude'],  # 俯仰角
            df.get_plane_state(self.Plane_ID_oppo)['roll_attitude'],  # 滚转角
            df.get_plane_state(self.Plane_ID_ally)['linear_speed'],  #总的速度

        ])

        #观测量归一化
        ob = self.state_normalized(ob)


        #其他信息输出
        distannce = self.get_distance()
        height = self.getHeight()
        ATA = self.get_ATA()
        AA = self.get_AA()



        info = {"distance":distannce,
                "height":height,
                "ATA":ATA,
                "AA":AA
                }


        return ob, reward, done, info

    def render(self, id=0):
        # 关闭无渲染模式，即进行渲染
        df.set_renderless_mode(False)

    # 环境重新进行初始化
    def reset(
            self,
    ):

        self.__init__(
            host=self.host,
            port=self.port,
            plane_slot=self.plane_slot,
            enemy_slot=self.enemy_slot,
            missile_slot=self.missile_slot,
            rendering=self.rendering,
        )

        ob = np.array([  # normalized  对状态向量进行标准化

            # 我方飞机位置，姿态，运动速度
            df.get_plane_state(self.Plane_ID_ally_0)['position'][0],  # 横坐标
            df.get_plane_state(self.Plane_ID_ally_0)['position'][2],  # 纵坐标
            df.get_plane_state(self.Plane_ID_ally_0)['position'][1],  # z轴坐标
            df.get_plane_state(self.Plane_ID_ally_0)['heading'],  # 偏航角 heading yaw
            df.get_plane_state(self.Plane_ID_ally_0)['pitch_attitude'],  # 俯仰角
            df.get_plane_state(self.Plane_ID_ally_0)['roll_attitude'],  # 滚转角
            df.get_plane_state(self.Plane_ID_ally_0)['linear_speed'],  #飞机飞行速度

            df.get_plane_state(self.Plane_ID_ally_1)['position'][0],  # 横坐标
            df.get_plane_state(self.Plane_ID_ally_1)['position'][2],  # 纵坐标
            df.get_plane_state(self.Plane_ID_ally_1)['position'][1],  # z轴坐标
            df.get_plane_state(self.Plane_ID_ally_1)['heading'],  # 偏航角 heading yaw
            df.get_plane_state(self.Plane_ID_ally_1)['pitch_attitude'],  # 俯仰角
            df.get_plane_state(self.Plane_ID_ally_1)['roll_attitude'],  # 滚转角
            df.get_plane_state(self.Plane_ID_ally_1)['linear_speed'],  # 飞机飞行速度


            # 敌方飞机位置，姿态，运动速度
            df.get_plane_state(self.Plane_ID_oppo_0)['position'][0],  # 横坐标
            df.get_plane_state(self.Plane_ID_oppo_0)['position'][2],  # 纵坐标
            df.get_plane_state(self.Plane_ID_oppo_0)['position'][1],  # z轴坐标
            df.get_plane_state(self.Plane_ID_oppo_0)['heading'],  # 偏航角 heading yaw
            df.get_plane_state(self.Plane_ID_oppo_0)['pitch_attitude'],  # 俯仰角
            df.get_plane_state(self.Plane_ID_oppo_0)['roll_attitude'],  # 滚转角
            df.get_plane_state(self.Plane_ID_oppo_0)['linear_speed'],  #飞机飞行速度

            df.get_plane_state(self.Plane_ID_oppo_1)['position'][0],  # 横坐标
            df.get_plane_state(self.Plane_ID_oppo_1)['position'][2],  # 纵坐标
            df.get_plane_state(self.Plane_ID_oppo_1)['position'][1],  # z轴坐标
            df.get_plane_state(self.Plane_ID_oppo_1)['heading'],  # 偏航角 heading yaw
            df.get_plane_state(self.Plane_ID_oppo_1)['pitch_attitude'],  # 俯仰角
            df.get_plane_state(self.Plane_ID_oppo_1)['roll_attitude'],  # 滚转角
            df.get_plane_state(self.Plane_ID_oppo_1)['linear_speed'],  # 飞机飞行速度

        ])
        #ob  = self.state_normalized(ob)

        return ob