# -*- coding: utf-8 -*-

"""
@author     : zuti
@software   : PyCharm
@file       : Env_3D.py
@time       : 2023/4/8 22:16
@desc       ：

三维仿真模拟
"""

# -*- coding: utf-8 -*-


import copy
import math

import numpy as np
from envs.Env_3D.Fdm_3D import  Fdm3D

# class EnvCore(object):
#     """
#     # 环境中的智能体
#     """
#
#     def __init__(self,init_state):
#         self.agent_num = 2  # 设置智能体(小飞机)的个数，这里设置为两个 # set the number of agents(aircrafts), here set to two
#         self.obs_dim = 33  # 设置智能体的观测维度 # set the observation dimension of agents
#         self.action_dim = 2  # 设置智能体的动作维度，这里假定为一个五个维度的 # set the action dimension of agents, here set to a five-dimensional
#
#     def reset(self):
#         """
#         # self.agent_num设定为2个智能体时，返回值为一个list，每个list里面为一个shape = (self.obs_dim, )的观测数据
#         # When self.agent_num is set to 2 agents, the return value is a list, each list contains a shape = (self.obs_dim, ) observation data
#         """
#         sub_agent_obs = []
#         for i in range(self.agent_num):
#             sub_obs = np.random.random(size=(33,))
#             sub_agent_obs.append(sub_obs)
#         print(f'reset维度： sub_agent_obs {len(sub_agent_obs)}, {len(sub_agent_obs[0])}')
#         return sub_agent_obs
#
#     def step(self, actions):
#         """
#         # self.agent_num设定为2个智能体时，actions的输入为一个2纬的list，每个list里面为一个shape = (self.action_dim, )的动作数据
#         # 默认参数情况下，输入为一个list，里面含有两个元素，因为动作维度为5，所里每个元素shape = (5, )
#         # When self.agent_num is set to 2 agents, the input of actions is a 2-dimensional list, each list contains a shape = (self.action_dim, ) action data
#         # The default parameter situation is to input a list with two elements, because the action dimension is 5, so each element shape = (5, )
#         """
#         sub_agent_obs = []
#         sub_agent_reward = []
#         sub_agent_done = []
#         sub_agent_info = []
#         for i in range(self.agent_num):
#             sub_agent_obs.append(np.random.random(size=(33,)))
#             sub_agent_reward.append([np.random.rand()])
#             sub_agent_done.append(False)
#             sub_agent_info.append({})

#         return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]
from gym import Env
from gym.error import DependencyNotInstalled
from gym.spaces import Box


class EnvCore(Env):

    def __init__(self,
                 look=True,
                 n_agents=2, n_enemies=2,  # 飞机数量
                 n_actions=3,  # 每个飞机的动作

                 ):
        """

        # 可以观测所有人[联盟+对手]的血量 ，  联盟队友的上一步动作
        :param n_agents:
        :param n_enemies:
        :param init_state: [x,y,v,theta] *4
        :param mtime:
        """

        print('Env_3D  __init__')

        # 智能体数量设置
        self.n_agents = n_agents  # 智能体数量
        self.n_enemies = n_enemies  # 敌方数量
        self.n_total = self.n_agents + self.n_enemies  # 所有数量
        # 信息数组
        print(f'n_agents:{self.n_agents} ,n_n_enemies:{self.n_enemies} , n_total:{self.n_total} ')

        self.n_actions = n_actions  # 每个智能体的动作数量
        self.n_move_feats = 6  # x,y,z,  v ,gamma, varphi # 位置信息


        # 初始化 四个飞机的姿态


        init_pos_aly_0 = [0,0,8600, 2, 0,0]
        init_pos_aly_1 = [0,1500,8600, 2, 0,0]
        init_pos_enemy_0 = [11000,0,8200, 1,  np.pi,0]
        init_pos_enemy_1 = [11000,5000,8000, 1, np.pi,0]
        init_pos_enemy_2 = [11000, 10000, 8000, 1, np.pi, 0]

        init_state = np.ones([self.n_total, self.n_move_feats])
        init_state[0] = init_pos_aly_0
        init_state[1] = init_pos_aly_1
        init_state[2] = init_pos_enemy_0
        init_state[3] = init_pos_enemy_1
#        init_state[4] = init_pos_enemy_2
        #init_state = np.array([init_pos_aly_0, init_pos_aly_1, init_pos_enemy_0, init_pos_enemy_1]).flatten()


        # 每一个飞机都是一个飞机动力模拟结构
        self.flight = []
        for i in range(0, self.n_total):
            fdm_id = i
            fdm_pos_x = init_state[i][0]
            fdm_pos_y = init_state[i][1]
            fdm_pos_z = init_state[i][2]
            fdm_v = init_state[i][3]
            fdm_gamma = init_state[i][4]
            fdm_varphi = init_state[i][5]
            flight1 = Fdm3D(fdm_id=fdm_id, fdm_pos_x=fdm_pos_x, fdm_pos_y=fdm_pos_y, fdm_pos_z = fdm_pos_z ,fdm_v=fdm_v, fdm_gamma= fdm_gamma,fdm_varphi=fdm_varphi)
            self.flight.append(flight1)


        self.state_before = copy.deepcopy(self.flight)

        print(f'flight init success')
        for i in range(self.n_total):
            print(
                f' flight {i} \n pos_x: {self.flight[i].fdm_pos_x}, pos_y: {self.flight[i].fdm_pos_y}, pos_y: {self.flight[i].fdm_pos_z} ,  v:{self.flight[i].fdm_v},   gamma:{self.flight[i].fdm_gamma} , fdm_varphi:{self.flight[i].fdm_varphi} ')

        # 战场作战范围 /100
        self.x_range = 20000
        self.y_range = 20000
        self.z_range =  18000
        self.vel_range = 400

        # 飞机的观测空间  self.observation_space
        # [
        #   [
        # 我方飞机姿态  6
        # x   横坐标  y   纵坐标   z 纵坐标  v   速度   gamma 航迹角   varphi 航向角
        # 战斗信息  1 + action_dim
        # hp  血量  + if last_action [a,omega=
        # 与联盟飞机的关系 （ 10 + action_dim ） * （n_agnets-1）
        #  visible是否可见
        #  相对距离 rD  rX 相对横坐标    rY 相对纵坐标  rZ 相对高度差  rV相对 速度   ATA   AA
        # if obsll_health :hp
        # if last_action  :a omega
        # 与敌机关系   (7+1) *  n_ememy
        #  visible是否可见
        #  相对距离rD  rX 相对横坐标    rY 相对纵坐标   rV相对 速度   ATA   AA
        #  if obsll_health :hp

        #   ]
        # ]

        # self.viewer = rendering.Viewer(x_range+100, y_range+100)

        # 飞机的动作空间
        # [  [a momega]
        #    [a momega]  ]
        self.own_feats = 1 + n_actions  # 个体信息特征 Hp lastAction
        self.n_state_allay = 9 + n_actions
        self.n_state_enemy = 9

        # 观测环境，数组存放所有的智能体部分观测环境
        self.agent_num = self.n_agents  # set the number of agents(aircrafts), here set to two
        self.obs_dim = self.get_obs_ally_size()  # set the observation dimension of agents
        self.action_dim = self.n_actions  # set the action dimension of agents, here set to a five-dimensional

        # 可视化设置
        self.viewer = None
        self.look = look

    def step(self, action):
        """

        :param action:
        :return:
        """
        print('--------step()----------')
        if self.look:  # 可视化设置
            self.render()

        # 记录更新前的状态 ,用来计算奖励
        self.state_before = copy.deepcopy(self.flight)


        print(f'--更新场景')
        # 根据动作，更新位置
        action_all = np.zeros([self.n_total, self.n_actions])

        # 由动作网络产生的 我方机动动作
        for i in range(self.n_agents):
            action_all[i] = action[i]

        #敌方机动动作
        for e in range(self.n_enemies):
            # 采取随机动作
            # action_all[e + self.n_agents] = [np.clip(np.random.randn(), -0.5, 0.5),
            #                                  np.clip(np.random.randn(), -0.1, 0.1)]
            # 不采取动作
            action_all[e + self.n_agents] = [0,0,0]

        # 每个智能体 执行动作
        for i in range(self.n_total):
            print(f'{i}:  action: {action_all[i]}')
            self.flight[i].last_action = action_all[i]  #保存动作指令
            self.flight[i].send_action(action_all[i])   #执行动作，根据运动方程更新位置

        # 更新场景 ， 判断是否碰撞与越界,更新Hp
        self.update_scene()
        #更新场景前后对比

        #输出信息
        for i in range(self.n_total):
            print(
                f' flight {i} \n pos_x: {self.flight[i].fdm_pos_x}; pos_y: {self.flight[i].fdm_pos_y}; pos_z: {self.flight[i].fdm_pos_z}; v:{self.flight[i].fdm_v};gamma:{self.flight[i].fdm_gamma}; varphi: {self.flight[i].fdm_varphi} hp:{self.flight[i].fdm_hp}')

        #


        # 得到新的场景的观测
        agent_obs = self.get_state_ally()

        print(f'------计算奖励')
        # 根据更新后的场景，给予奖励
        rewards = self.reward()

        # 判断是否结束
        print(f'------判断结束')
        dones = self.done()

        # 获得观测信息/state信息
        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []

        for i in range(self.n_agents):
            sub_agent_obs.append(agent_obs[i])
            sub_agent_reward.append([rewards[i]])
            sub_agent_done.append(dones[i])
            sub_agent_info.append({})

        # print(f'sub_agent_reward: {sub_agent_reward}')
        print(f'观测： sub_agent_obs{len(sub_agent_obs[0])}, {sub_agent_obs}')
        print(f'奖励： sub_agent_reward {len(sub_agent_reward)} , {sub_agent_reward}')
        print(f'结束： sub_agent_done{sub_agent_done}')
        print(f'其他信息： sub_agent_info{sub_agent_info}, ')

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]

    def reset(self):
        """
        self.n_agents设定为2个智能体时，返回值为一个list，每个list里面为一个shape = (self.obs_dim, )的观测数据

        """
        print('reset()')
        # self.__init__()
        # 重新设置姿态
        # 初始化 四个飞机的姿态
        init_pos_aly_0 = [0, 0, 86, 2, 0, 0]
        init_pos_aly_1 = [0, 15, 86, 2, 0, 0]
        init_pos_enemy_0 = [110, 0, 82, 1, np.pi, 0]
        init_pos_enemy_1 = [110, 50, 80, 1, np.pi, 0]
        init_pos_enemy_2 = [110, 100, 80, 1, np.pi, 0]

        init_state = np.ones([self.n_total, self.n_move_feats])
        init_state[0] = init_pos_aly_0
        init_state[1] = init_pos_aly_1
        init_state[2] = init_pos_enemy_0
        init_state[3] = init_pos_enemy_1
        #init_state[4] = init_pos_enemy_2

        # 每一个飞机都是一个飞机动力模拟结构

        for i in range(0, self.n_total):
            self.flight[i].fdm_pos_x = init_state[i][0]
            self.flight[i].fdm_pos_y = init_state[i][1]
            self.flight[i].fdm_pos_z = init_state[i][2]
            self.flight[i].fdm_v = init_state[i][3]
            self.flight[i].fdm_gamma = init_state[i][4]
            self.flight[i].fdm_varphi = init_state[i][5]

        agent_obs = self.get_state_ally()
        sub_agent_obs = []
        for agent_id in range(self.n_agents):
            sub_obs = agent_obs[agent_id]
            sub_agent_obs.append(sub_obs)
        print(f'reset 维度： sub_agent_obs: {len(sub_agent_obs)}, {len(sub_agent_obs[0])}')
        print(f'rest ： sub_agent_obs: {sub_agent_obs}')
        return sub_agent_obs

    def render(self, mode="human", close=False):
        screen_width = 1000
        screen_height = 1000

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        # if mode == 'human':
        #     mode = "rgb_array"

        if self.viewer is None:  # 初始化
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            #
            # 定义坐标轴
            # line_x_0 = rendering.Line((10, 10), (10, 600))
            # line_y_0 = rendering.Line((10, 10), (600, 10))
            # line_x_0.set_color(255, 0, 0)
            # line_y_0.set_color(0, 0, 255)
            # self.viewer.add_geom(line_x_0)
            # self.viewer.add_geom(line_y_0)

            line_y = rendering.Line((0, 300), (600, 300))
            line_x = rendering.Line((300, 0), (300, 600))
            line_x.set_color(255, 0, 0)
            line_y.set_color(0, 0, 255)
            self.viewer.add_geom(line_x)
            self.viewer.add_geom(line_y)

            # 绘制我方飞机
            self.plane_allay = []
            for i in range(self.n_agents):
                # 图像
                plane_ally_0 = rendering.make_polygon([(0, 10), (0, -10), (20, 0)], filled=True)  # 1 # 绘制三角形
                plane_ally_0.set_color(255, 0, 0)  # 给元素添加颜色
                # 变换对象
                plane_0 = rendering.Transform()  # 添加变换属性
                plane_ally_0.add_attr(plane_0)  # 将变换属性与图像绑定
                self.plane_allay.append(plane_0)  # 保存变换对象
                self.viewer.add_geom(plane_ally_0)

            self.plane_enemy = []
            for i in range(self.n_enemies):
                # 图像
                plane_ally_0 = rendering.make_polygon([(0, 10), (0, -10), (20, 0)], filled=True)  # 1 # 绘制三角形
                plane_ally_0.set_color(0, 255, 0)  # 给元素添加颜色
                # 变换对象
                plane_0 = rendering.Transform()  # 添加变换属性
                plane_ally_0.add_attr(plane_0)  # 将变换属性与图像绑定
                self.plane_enemy.append(plane_0)
                self.viewer.add_geom(plane_ally_0)



        for i in range(self.n_agents):  # 联盟设置
            x_ally = self.flight[i].fdm_pos_x/100 + 300  # x
            y_ally = self.flight[i].fdm_pos_y/100 + 300  # y
            #theta_ally = self.flight[i].fdm_varphi  # theta
            self.plane_allay[i].set_translation(x_ally, y_ally)  # 设置位置变换
            #self.plane_allay[i].set_rotation(theta_ally)  # 设置角度变换
        #
        for e in range(self.n_enemies):  # 对手设置
            x_enemy = self.flight[e+ self.n_agents].fdm_pos_x/100 + 300  # x
            y_enemy = self.flight[e + self.n_agents].fdm_pos_y/100 + 300  # y
            #theta_ally = self.flight[i + self.n_agents].fdm_varphi  # theta
            self.plane_enemy[e].set_translation(x_enemy, y_enemy)  # 设置位置变换
            #self.plane_enemy[i].set_rotation(theta_ally)  # 设置角度变换



        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def update_scene(self):
        """
        更新整个场景
        判断是否碰撞，越界，达成攻击效果，更新血量
        1 判断越界，越界血量为0
        2 判断攻击 达成攻击态势（距离+角度） ，对手血量为0
                但如果只达距离，而没有形成角度，给对手带来流血效果 ？  ：目的是促进接近

        :return:
        """
        print('  --update_scene--')

        for i in range(self.n_agents):  #

            flight = self.flight[i]
            flight_pos_x = flight.fdm_pos_x
            flight_pos_y = flight.fdm_pos_y
            flight_pos_z = flight.fdm_pos_z
            flight_gamma = flight.fdm_gamma
            flight_varphi = flight.fdm_varphi

            flight_hp = flight.fdm_hp

            if not (- self.x_range <= flight_pos_x <= self.x_range and - self.y_range <= flight_pos_y <= self.y_range  and  0< flight_pos_z <= self.z_range):  # 如果越界，血量为0
                self.flight[i].fdm_hp = 0

            # 如果达成攻击态势，目标结果hp为0
            # 攻击态势，当前飞机和敌机的距离， 角度

            e_id = i + self.n_agents
            flight_e = self.flight[e_id]
            e_pos_x = flight_e.fdm_pos_x
            e_pos_y = flight_e.fdm_pos_y
            e_pos_z = flight_e.fdm_pos_z

            dist = self.distance(flight_pos_x, flight_pos_y, flight_pos_z,   e_pos_x, e_pos_y , e_pos_z)
            ATA = self.ATA(flight_pos_x, flight_pos_y, flight_pos_z,flight_gamma,flight_varphi, e_pos_x, e_pos_y,e_pos_z)
            print(f'dist:{dist}; flight.attack_range: {flight.attack_range};  {dist < flight.attack_range}')
            print(f'ATA:{ATA}; { 0 <= ATA < np.pi / 2} ')
            if dist < flight.attack_range :  # 我机达成攻击态势，敌机血量为0
                self.flight[e_id].fdm_hp = 0


    def reward(self):
        """
        计算联盟的奖励
        参数 state_before 表示前一状态

        :return:
        """


        dist_b = np.ones([self.n_agents , self.n_enemies])  # 我方n架对敌方m架形成的 两两之间的距离关系 ， 更新位置之前
        dist_n = np.ones([self.n_agents , self.n_enemies])  # 我方n架对敌方m架形成的 两两之间的距离关系 ， 现在

        ATA_b = np.ones([self.n_agents , self.n_enemies])  # 我方n架对敌方m架形成的 两两之间的ATA关系 ， 更新位置之前
        ATA_n = np.ones([self.n_agents , self.n_enemies])  # 我方n架对敌方m架形成的 两两之间的ATA关系 ， 现在


        for i in range(self.n_agents):
            for e in range(self.n_enemies):
                dist_b[i][e] = self.distance(self.state_before[i].fdm_pos_x, self.state_before[i].fdm_pos_y, self.state_before[i].fdm_pos_z,
                                          self.state_before[e + self.n_agents].fdm_pos_x,
                                          self.state_before[e + self.n_agents].fdm_pos_y,
                                          self.state_before[e + self.n_agents].fdm_pos_z)
                dist_n[i][e] = self.distance(self.flight[i].fdm_pos_x, self.flight[i].fdm_pos_y, self.flight[i].fdm_pos_z,
                                          self.flight[e + self.n_agents].fdm_pos_x,
                                          self.flight[e + self.n_agents].fdm_pos_y,
                                          self.flight[e + self.n_agents].fdm_pos_z)

                print(f'distance  {i}--{e} ： before: {dist_b[i][e]} ; now :{dist_n[i][e]}   ')

                ATA_b[i][e] = self.ATA(self.state_before[i].fdm_pos_x, self.state_before[i].fdm_pos_y, self.state_before[i].fdm_pos_z,
                                    self.state_before[i].fdm_gamma, self.state_before[i].fdm_varphi,
                                    self.state_before[e + self.n_agents].fdm_pos_x,
                                    self.state_before[e + self.n_agents].fdm_pos_y,
                                    self.state_before[e + self.n_agents].fdm_pos_z)
                ATA_n[i][e] = self.ATA(self.flight[i].fdm_pos_x, self.flight[i].fdm_pos_y, self.flight[i].fdm_pos_z,
                                    self.flight[i].fdm_gamma,self.flight[i].fdm_varphi,
                                    self.flight[e + self.n_agents].fdm_pos_x,
                                    self.flight[e + self.n_agents].fdm_pos_y,
                                    self.flight[e + self.n_agents].fdm_pos_z)
                print(f'ATA  {i}--{e} ： before: {dist_b[i][e]} ; now :{dist_n[i][e]}   ')



        # 首先计算 血量奖励  , 血量奖励只关于最终结果
        reward_hp = 0
        for e in range(self.n_enemies):
            e_id = e + self.n_agents
            if self.flight[e_id].fdm_hp == 0:
                reward_hp += 3000
        print(f'reward_hp: {reward_hp}')

        # 其次 距离奖励 ,角度奖励
        reward_distance = np.zeros(self.n_agents)
        reward_angle = np.zeros(self.n_agents)

        # 这里的奖励只设置为 i 直接对j 进行评价，即预先选定我方1的目标只为敌方1
        for i in range(self.n_agents):  # 每个智能体的奖励
            # 每个智能体只与自己标号相同的enmey进行对比,如 智能体1 对 敌人1 ,这个时候只有当我方无人机数量小于敌方无人机数量才可以

            # 如果距离相对于上一步减小
            if dist_b[i][i] > dist_n[i][i]:
                reward_distance[i] += 0.5
            else:
                reward_distance[i] -= 0.5
            print(f'{i} reward_distance 1: {reward_distance}')

            # 相对距离的奖励
            # 当我方与敌方距离 小于 sight_range  奖励为  >1
            # 当我方与敌方距离  sight_range —— 2 * sight_range      奖励为  <1
            # 当我方距离与敌方距离 大于 2*sight_range   奖励 <0
            reward_distance[i] += 2 - (dist_n[i][i] / self.flight[i].sight_range)
            print(f'{i} reward_distance 2: {(dist_n[i][i] / self.flight[i].sight_range)}')

            print(f'{i} reward_distance total: {reward_distance[i]}')

            # 攻击态势角度奖励，如果我方飞机与飞机角度相差不超过Pi，给予奖励
            if ATA_b[i][i] > ATA_n[i][i]:
                reward_angle[i] += 0.1
            else:
                reward_angle[i] -= 0.5

            print(f'{i} reward_angle 1: {reward_angle}')

            # 当距离接近之后，再考虑角度,当距离小于 slight_range 进行调整角度
            if dist_n[i][i] <= self.flight[i].sight_range:
                reward_angle[i] += (np.pi / 2 - ATA_n[i][i]) / 2
            print(f'{i}  reward_angle 2: {(np.pi / 2 - ATA_n[i][i]) / 2}')

            print(f'{i}   reward_angle total: {reward_angle}')



        # 求总奖励
        reward_total = np.zeros(self.n_agents)
        for i in range(self.n_agents):
            reward_total[i] += reward_hp + reward_distance[i] + reward_angle[i]

        print(f'reward_total: {reward_total}')
        return reward_total

    def done(self):
        """"
        判断是否结束
        """
        dones = [False] * self.n_agents
        done_flag = False
        # 当任何一方血量为0 ,整个游戏结束
        for i in range(self.n_total):
            if self.flight[i].fdm_hp == 0:
                done_flag = True

        for i in range(self.n_agents):
            dones[i] = done_flag

        return dones

    def get_obs_ally_size(self):
        size_obs_pos = self.n_move_feats
        size_obs_hp = self.own_feats
        size_obs_allay = (self.n_agents - 1) * self.n_state_allay
        size_obs_enemy = self.n_enemies * self.n_state_enemy
        size_total = size_obs_pos + size_obs_hp + size_obs_allay + size_obs_enemy

        return size_total

    def get_obs_ally(self):
        """
        得到联盟飞机的所有个体观测信息
        :return:
        """
        size_obs_pos = self.n_move_feats
        size_obs_hp = self.own_feats
        size_obs_allay = (self.n_agents - 1) * self.n_state_allay
        size_obs_enemy = self.n_enemies *  self.n_state_enemy
        size_total = size_obs_pos + size_obs_hp + size_obs_allay + size_obs_enemy

        obs_all = np.zeros([self.n_agents, size_total])

        # 我方两架飞机的部分观测信息
        for i in range(self.n_agents):

            agent_id = i  #
            # obs_id = np.arange(size_total)  # 飞机编号对应的所有观测信息
            move_feats = np.zeros(size_obs_pos, dtype=np.float32)  # 位置，速度信息 x,y,v,theta
            own_feats = np.zeros(size_obs_hp, dtype=np.float32)  # 血量，last_action
            ally_feats = np.zeros([self.n_agents - 1, self.n_state_allay], dtype=np.float32)  # 和联盟的相对位置关系
            enemy_feats = np.zeros([self.n_enemies, self.n_state_enemy], dtype=np.float32)  # 和对手的位置关系

            # 根据飞机编号 获得 pos_feats
            flight = self.flight[agent_id]
            if flight.fdm_hp > 0:  # 当飞机血量大于0的时候才记录信息
                x = flight.fdm_pos_x
                y = flight.fdm_pos_y
                z = flight.fdm_pos_z
                v = flight.fdm_v
                gamma = flight.fdm_gamma
                varphi = flight.fdm_varphi
                sight_range = flight.sight_range

                # 对运动信息pos_feats进行归一化处理
                move_feats[0] = x / self.x_range
                move_feats[1] = y / self.y_range
                move_feats[2] = z / self.z_range
                move_feats[3] = v / self.vel_range
                move_feats[4] = gamma / np.pi
                move_feats[5] = varphi / np.pi

                # 飞机的 hp 与last_action
                own_feats[0] = flight.fdm_hp
                own_feats[1:] = flight.last_action

                # 获得联邦状态信息  联邦是在 n_angents 中 除了当前编号i 以外的所有飞机
                al_ids = [al_id for al_id in range(self.n_agents) if al_id != agent_id]

                for a, al_id in enumerate(al_ids):
                    flight_al = self.flight[al_id]  # 根据联盟编号获得飞机对象
                    # 联盟飞机的位置信息
                    al_x = flight_al.fdm_pos_x
                    al_y = flight_al.fdm_pos_x
                    al_z = flight_al.fdm_pos_z
                    dist = self.distance(x, y, z, al_x, al_y, al_z)

                    if dist <= sight_range and flight_al.fdm_hp > 0:
                        al_v = flight_al.fdm_v
                        al_gamma = flight_al.fdm_gamma
                        al_varphi = flight_al.fdm_varphi

                        ally_feats[a, 0] = 1  # visible

                        ally_feats[a, 1] = dist / sight_range  # distance
                        ally_feats[a, 2] = (al_x - x) / sight_range  # relative X
                        ally_feats[a, 3] = (al_y - y) / sight_range  # relative Y
                        ally_feats[a, 4] = (al_z - z) / sight_range  # relative Z
                        ally_feats[a, 5] = (al_v - v) / self.vel_range  # relative V

                        ally_feats[a, 6] = self.ATA(x, y, z, gamma, varphi, al_x, al_y, al_z) / np.pi  # ATA
                        ally_feats[a, 7] = self.AA(x, y, z, al_x, al_y, al_z, al_gamma, al_varphi) / np.pi  # AA

                        ally_feats[a, 8] = flight_al.fdm_hp  # health
                        ally_feats[a, 9:] = flight_al.last_action  # last_action

                # 获得对手数据，在飞机编号中为 n_ally，；；，n_ally+n_ememies，即敌机0 对应飞机编号3
                for e in range(self.n_enemies):
                    enemy_id = e + self.n_agents
                    flight_enemy = self.flight[enemy_id]

                    enemy_x = flight_enemy.fdm_pos_x
                    enemy_y = flight_enemy.fdm_pos_y
                    enemy_z = flight_enemy.fdm_pos_z

                    dist = self.distance(x, y, z, enemy_x, enemy_y, enemy_z)

                    if flight_enemy.fdm_hp > 0  and  dist <= sight_range:  # visible and alive
                        enemy_v = flight_enemy.fdm_v
                        enemy_gamma = flight_enemy.fdm_gamma
                        enemy_varphi = flight_enemy.fdm_varphi

                        enemy_feats[e][0] = 1  # visible

                        enemy_feats[e, 1] = dist / sight_range  # distance
                        enemy_feats[e, 2] = (enemy_x - x) / sight_range  # relative X
                        enemy_feats[e, 3] = (enemy_y - y) / sight_range  # relative Y
                        enemy_feats[e, 4] = (enemy_z - z) / sight_range  # relative Z

                        enemy_feats[e, 5] = (enemy_v - v) / self.vel_range  # relative V

                        enemy_feats[e, 6] = self.ATA(x, y, z, gamma, varphi, enemy_x, enemy_y, enemy_z) / np.pi  # ATA
                        enemy_feats[e, 7] = self.AA(x, y, z, enemy_x, enemy_y, enemy_z, enemy_gamma,
                                                   enemy_varphi) / np.pi  # AA

                        enemy_feats[e][8] = flight_enemy.fdm_hp

            obs_id = np.concatenate((
                move_feats.flatten(),
                own_feats.flatten(),
                ally_feats.flatten(),
                enemy_feats.flatten()

            ))
            obs_all[i] = obs_id

        return obs_all

    def get_state_ally(self):
        """
        得到联盟飞机的所有个体全局信息，即去掉了观测范围的限制
        :return:
        """
        size_obs_pos = self.n_move_feats
        size_obs_hp = self.own_feats
        size_obs_allay = (self.n_agents - 1) * self.n_state_allay
        size_obs_enemy = self.n_enemies * self.n_state_enemy
        size_total = size_obs_pos + size_obs_hp + size_obs_allay + size_obs_enemy

        state_all = np.zeros([self.n_agents, size_total])

        # 我方两架飞机的部分观测信息
        for i in range(self.n_agents):

            agent_id = i  #
            move_feats = np.zeros(size_obs_pos, dtype=np.float32)  # 位置，速度信息 x,y,v,theta
            own_feats = np.zeros(size_obs_hp, dtype=np.float32)  # 血量，last_action
            ally_feats = np.zeros([self.n_agents - 1, self.n_state_allay], dtype=np.float32)  # 和联盟的相对位置关系
            enemy_feats = np.zeros([self.n_enemies, self.n_state_enemy], dtype=np.float32)  # 和对手的位置关系

            # 根据飞机编号 获得 pos_feats
            flight = self.flight[agent_id]
            if flight.fdm_hp > 0:  # 当飞机血量大于0的时候才记录信息
                x = flight.fdm_pos_x
                y = flight.fdm_pos_y
                z = flight.fdm_pos_z
                v = flight.fdm_v
                gamma = flight.fdm_gamma
                varphi = flight.fdm_varphi
                sight_range = flight.sight_range

                # 对运动信息pos_feats进行归一化处理
                move_feats[0] = x / self.x_range
                move_feats[1] = y / self.y_range
                move_feats[2] = z / self.z_range
                move_feats[3] = v / self.vel_range
                move_feats[4] = gamma / np.pi
                move_feats[5] = varphi / np.pi


                # 飞机的 hp 与last_action
                own_feats[0] = flight.fdm_hp
                own_feats[1:] = flight.last_action

                # 获得联邦状态信息  联邦是在 n_angents 中 除了当前编号i 以外的所有飞机
                al_ids = [al_id for al_id in range(self.n_agents) if al_id != agent_id]
                for a, al_id in enumerate(al_ids):
                    flight_al = self.flight[al_id]  # 根据联盟编号获得飞机对象
                    # 联盟飞机的位置信息
                    al_x = flight_al.fdm_pos_x
                    al_y = flight_al.fdm_pos_x
                    al_z = flight_al.fdm_pos_z
                    dist = self.distance(x, y,z, al_x, al_y,al_z)

                    if flight_al.fdm_hp > 0:
                        al_v = flight_al.fdm_v
                        al_gamma = flight_al.fdm_gamma
                        al_varphi = flight_al.fdm_varphi

                        ally_feats[a, 0] = 1  # visible

                        ally_feats[a, 1] = dist / sight_range  # distance
                        ally_feats[a, 2] = (al_x - x) / sight_range  # relative X
                        ally_feats[a, 3] = (al_y - y) / sight_range  # relative Y
                        ally_feats[a, 4] = (al_z - z) / sight_range  # relative Z
                        ally_feats[a, 5] = (al_v - v) / self.vel_range  # relative V

                        ally_feats[a, 6] = self.ATA(x, y, z, gamma,varphi, al_x, al_y,al_z) / np.pi  # ATA
                        ally_feats[a, 7] = self.AA(x, y,z, al_x, al_y, al_z, al_gamma,al_varphi) / np.pi  # AA

                        ally_feats[a, 8] = flight_al.fdm_hp  # health
                        ally_feats[a, 9:] = flight_al.last_action  # last_action

                # 获得对手数据，在飞机编号中为 n_ally，；；，n_ally+n_ememies，即敌机0 对应飞机编号3
                for e in range(self.n_enemies):

                    flight_enemy = self.flight[e + self.n_agents]
                    enemy_x = flight_enemy.fdm_pos_x
                    enemy_y = flight_enemy.fdm_pos_y
                    enemy_z = flight_enemy.fdm_pos_z

                    dist = self.distance(x, y,z, enemy_x, enemy_y,enemy_z)

                    if flight_enemy.fdm_hp > 0:  # visible and alive
                        enemy_v = flight_enemy.fdm_v
                        enemy_gamma = flight_enemy.fdm_gamma
                        enemy_varphi =flight_enemy.fdm_varphi

                        enemy_feats[e][0] = 1  # visible

                        enemy_feats[e, 1] = dist / sight_range  # distance
                        enemy_feats[e, 2] = (enemy_x - x) / sight_range  # relative X
                        enemy_feats[e, 3] = (enemy_y - y) / sight_range  # relative Y
                        enemy_feats[e, 4] = (enemy_z - z) / sight_range  # relative Z

                        enemy_feats[e, 5] = (enemy_v - v) / self.vel_range  # relative V

                        enemy_feats[e, 6] = self.ATA(x, y, z, gamma, varphi, enemy_x, enemy_y, enemy_z) / np.pi  # ATA
                        enemy_feats[e, 7] = self.AA(x, y, z, enemy_x, enemy_y, enemy_z, enemy_gamma,
                                                    enemy_varphi) / np.pi  # AA

                        enemy_feats[e][8] = flight_enemy.fdm_hp

            obs_id = np.concatenate((
                move_feats.flatten(),
                own_feats.flatten(),
                ally_feats.flatten(),
                enemy_feats.flatten()

            ))
            state_all[i] = obs_id

        return state_all

    @staticmethod
    def distance(x1, y1,z1, x2, y2,z2):
        """Distance between two points."""
        return math.hypot(x2 - x1, y2 - y1 ,  z2-z1  )

    @staticmethod
    def ATA(x, y, z, gamma,varphi , x1, y1 ,z1):
        """
        计算两机位置矢量与 速度的夹角 我机速度与位置矢量夹角
        :param x:
        :param y:
        :param theta:
        :param x1:
        :param y1:
        :param theta1:
        :return:
        """
        # 首先计算 两机位置矢量与 x轴(向量【1，0】)的夹角
        pos_vector = np.array([x1 - x, y1 - y, z1-z])
        v_vector = np.array([np.cos(gamma)*np.sin(varphi),np.cos(gamma)*np.cos(varphi) ,np.sin(gamma)])
        if np.linalg.norm(pos_vector) != 0:
            cos_radian = np.dot(pos_vector, v_vector) / (np.linalg.norm(pos_vector))
            # cos theta = 向量a 点乘 向量b  /   向量a的模 乘以 向量b的模
            cos_radian = np.clip(cos_radian, -1, 1)  # 进行裁剪
            radian = np.arccos(cos_radian)  # arccos结果为[0,pi]的弧度制
        else:
            radian = 0

        ATA = np.clip(radian,0.01,np.pi)
        return ATA


    @staticmethod
    def AA(x, y, z, x1, y1 ,z1,gamma,varphi ):
        """
        计算两机位置矢量与 速度的夹角 我机速度与位置矢量夹角
        :param x:
        :param y:
        :param theta:
        :param x1:
        :param y1:
        :param theta1:
        :return:
        """
        # 首先计算 两机位置矢量与 x轴(向量【1，0】)的夹角
        pos_vector = np.array([x1 - x, y1 - y, z1-z])
        v_vector = np.array([np.cos(gamma)*np.sin(varphi),np.cos(gamma)*np.cos(varphi) ,np.sin(gamma)])

        if np.linalg.norm(pos_vector) != 0:
            cos_radian = np.dot(pos_vector, v_vector) / (np.linalg.norm(pos_vector))
        # cos theta = 向量a 点乘 向量b  /   向量a的模 乘以 向量b的模
            cos_radian = np.clip(cos_radian, -1, 1)  # 进行裁剪

            radian = np.arccos(cos_radian)  # arccos结果为[0,pi]的弧度制
        else:
            radian = 0
        AA = np.clip(radian,0.01,np.pi)
        return AA

