# -*- coding: utf-8 -*-

"""
@author     : zuti
@software   : PyCharm
@file       : Env_2D_2v2.py
@time       : 2023/4/8 22:15
@desc       ：

2维可视化 飞机飞行gym环境
"""
import copy
import math

import numpy as np
from envs.JsbSimEnv.jsbsimFdm import JsbsimFdm


from gym import Env
from gym.error import DependencyNotInstalled
from gym.spaces import Box


class EnvCore(Env):

    def  __init__(self,
                 look=True,
                 n_agents=2, n_enemies=2,  # 飞机数量
                 n_actions=4,  # 每个飞机的动作

                 ):
        """

        # 可以观测所有人[联盟+对手]的血量 ，  联盟队友的上一步动作
        :param n_agents:
        :param n_enemies:
        :param init_state: [x,y,v,theta] *4
        :param mtime:
        """

        print('Env_JSbSIM  __init__')

        # 智能体数量设置
        self.n_agents = n_agents  # 智能体数量
        self.n_enemies = n_enemies  # 敌方数量
        self.n_total = self.n_agents + self.n_enemies  # 所有数量
        # 信息数组
        print(f'n_agents:{self.n_agents} ,n_n_enemies:{self.n_enemies} , n_total:{self.n_total} ')




        # 观测环境，数组存放所有的智能体部分观测环境
        self.agent_num = self.n_agents  # set the number of agents(aircrafts), here set to two
        self.action_dim =  n_actions  # set the action dimension of agents, here set to a five-dimensional

        self.n_move_feats = 9  # y,x,h theta, psi ,phi , u,v,w ,hp,
        self.own_feats = 1 + n_actions  # 个体信息特征 Hp lastAction
        # 对每一个同盟、敌人 的观测特征
        self.size_obs_allay = 18
        self.size_obs_enemy = 14  # 观测不包含动作
        self.obs_dim = self.get_obs_ally_size()  # set the observation dimension of agents





        # 随机初始化我方飞机位置

        #init_state = []
        self.flight = []
        # 我方飞机位置限定在【1000,2000】*【5000，5500】
        # for a in range(self.n_agents):
        #
        #     rnd_x =  np.random.uniform(1000, 2000)
        #     rnd_y =  np.random.uniform(1000, 2000)
        #     rand_z = np.random.uniform(10000, 12000)
        #
        #     rnd_theta = 0
        #     rnd_psi = 0
        #     rnd_phi = 0
        #     rnd_v = np.random.uniform(200, 250)
        #
        #
        #     #init_pos_aly = [rnd_x,rnd_eci_y,rand_z,rnd_theta,rnd_psi,rnd_phi,rnd_v]
        #
        #     #init_state.append(init_pos_aly)
        #
        #     # 初始化飞机
        #     flight_a = JsbsimFdm(fdm_id=a, fdm_ic_y=rnd_y, fdm_ic_x=rnd_x, fdm_ic_z=rand_z)
        #     self.flight.append(flight_a)
        flight_a = JsbsimFdm(fdm_id=0, fdm_ic_y=0.01, fdm_ic_x=0.01, fdm_ic_z=10000)
        self.flight.append(flight_a)
        flight_a = JsbsimFdm(fdm_id=1, fdm_ic_y=0.01, fdm_ic_x=0.04, fdm_ic_z=12000)
        self.flight.append(flight_a)


        # 敌方飞机位置限定在【50，150】
        # for e in range(self.n_enemies):
        #     rnd_x = np.random.uniform(5000, 6000)
        #     rnd_y = np.random.uniform(5000, 8000)
        #     rand_z = np.random.uniform(15000, 18000)
        #
        #     rnd_theta = 0
        #     rnd_psi = 0
        #     rnd_phi = 0
        #     rnd_v = np.random.uniform(200, 250)
        #
        #     #init_pos_aly = [rnd_x, rnd_eci_y, rand_z, rnd_theta, rnd_psi, rnd_phi, rnd_v]
        #
        #     #init_state.append(init_pos_aly)
        #
        #     # 初始化飞机
        #     e_id = e + self.n_agents
        #     flight_e = JsbsimFdm(fdm_id=e_id, fdm_ic_y=rnd_y, fdm_ic_x=rnd_x, fdm_ic_z=rand_z)
        #     self.flight.append(flight_e)
        flight_e = JsbsimFdm(fdm_id=2, fdm_ic_y=0.04, fdm_ic_x=0.01, fdm_ic_z=15000)
        self.flight.append(flight_e)
        flight_e = JsbsimFdm(fdm_id=3, fdm_ic_y=0.04, fdm_ic_x=0.04, fdm_ic_z=18000)
        self.flight.append(flight_e)
        #init_state = np.array(init_state).flatten()


        # self.flight_ally=[]
        # self.flight_enemy = []
        # for i in range(0,self.n_agents):
        #     self.flight_ally.append(self.flight[i])
        #
        # for i in range(self.n_agents,self.n_total):
        #     self.flight_enemy.append(self.flight[i])

        #self.state_before = copy.deepcopy(self.flight)
        self.state_before = []
        for f in self.flight:
            state = f.getState()
            self.state_before.append(state)

        print(f'flight init success')



        # 战场作战范围
        self.x_range = 15000
        self.y_range = 15000
        self.h_range =  25000
        self.vel_range = 400

        # 飞机的观测空间  self.observation_space
        # [
        #   [
        # 我方飞机姿态  4
        # x   横坐标  y   纵坐标  v   速度    theta 夹角
        # 战斗信息  1 +2
        # hp  血量  + if last_action [a,omega=
        # 与联盟飞机的关系 （ 7 + 1 +2 ） * （n_agnets-1）
        #  visible是否可见
        #  相对距离rD  rX 相对横坐标    rY 相对纵坐标   rV相对 速度   ATA   AA
        # if obsll_health :hp
        # if last_action  :a omega
        # 与敌机关系   (7+1) *  n_ememy
        #  visible是否可见
        #  相对距离rD  rX 相对横坐标    rY 相对纵坐标   rV相对 速度   ATA   AA
        #  if obsll_health :hp

        #   ]
        # ]


        # 飞机的动作空间
        # [  [a momega]
        #    [a momega]  ]



        # 可视化设置
        self.viewer = None
        self.look = look

    def step(self, action):
        """

        :param action:
        :return:
        """
        print('--------step()----------')
        if True:  # 可视化设置
            self.render()

        # 记录更新前的状态 ,用来计算奖励
        self.state_before =[]
        for f in self.flight :
            state =  f.getState()
            self.state_before.append(state)


        print(f'--更新场景')
        # 根据动作，更新位置
        for a in range(self.n_agents):
            self.flight[a].last_action = action[a]  # 保存最后一步动作
            re = self.flight[a].send_action(action[a]) # 根据动作信息更新位置
            # re = self.flight[a].step(1)

        for e in range(self.n_enemies):
            aileron = np.clip(np.random.randn(), -1, 1)
            elevator= np.clip(np.random.randn(), -1, 1)
            rudder= np.clip(np.random.randn(), -1, 1)
            throttle = np.clip(np.random.randn(), 0, 1)
            #w0 = np.clip(np.random.randn(), -0.1, 0.1)
            action_enemy = [aileron,elevator,rudder,throttle]
            e_id = e+ self.n_agents
            self.flight[e_id].last_action = action_enemy
            re = self.flight[e_id].send_action(action_enemy)
            # re = self.flight[e_id].step(1)



        # 更新场景 ， 判断是否碰撞与越界,更新Hp
        self.update_scene()


        #输出信息
        for i in range(self.n_total):
            print(f'{self.flight[i].getState()}')

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
        # 我方飞机位置限定在【1000,2000】*【5000，5500】
        # for a in range(self.n_agents):
        #     rnd_x = np.random.uniform(1000, 2000)
        #     rnd_y = np.random.uniform(1000, 2000)
        #     rand_z = np.random.uniform(10000, 12000)
        #
        #     rnd_theta = 0
        #     rnd_psi = 0
        #     rnd_phi = 0
        #     rnd_v = np.random.uniform(200, 250)
        #
        #     init_pos_aly = [rnd_x, rnd_y, rand_z, rnd_theta, rnd_psi, rnd_phi, rnd_v]
        #
        #     # init_state.append(init_pos_aly)
        #
        #     # 初始化飞机
        #     flight_a = JsbsimFdm(fdm_id=a, fdm_ic_x=rnd_y, fdm_ic_y=rnd_x, fdm_ic_z=rand_z)
        #     self.flight.append(flight_a)
        #
        # # 敌方飞机位置限定在【50，150】
        # for e in range(self.n_enemies):
        #     rnd_x = np.random.uniform(6000, 8000)
        #     rnd_y = np.random.uniform(6000, 8000)
        #     rand_z = np.random.uniform(15000, 20000)
        #
        #     rnd_theta = 0
        #     rnd_psi = 0
        #     rnd_phi = 0
        #     rnd_v = np.random.uniform(200, 250)
        #
        #     #init_pos_aly = [rnd_x, rnd_eci_y, rand_z, rnd_theta, rnd_psi, rnd_phi, rnd_v]
        #
        #     # init_state.append(init_pos_aly)
        #
        #     # 初始化飞机
        #     e_id = e + self.n_agents
        #     flight_e = JsbsimFdm(fdm_id=e_id, fdm_ic_y=rnd_y, fdm_ic_x=rnd_x, fdm_ic_z=rand_z)
        #     self.flight.append(flight_e)

        flight_a = JsbsimFdm(fdm_id=0, fdm_ic_y=0.01, fdm_ic_x=0.01, fdm_ic_z=10000)
        self.flight.append(flight_a)
        flight_a = JsbsimFdm(fdm_id=1, fdm_ic_y=0.01, fdm_ic_x=0.04, fdm_ic_z=12000)
        self.flight.append(flight_a)
        flight_e = JsbsimFdm(fdm_id=2, fdm_ic_y=0.04, fdm_ic_x=0.01, fdm_ic_z=15000)
        self.flight.append(flight_e)
        flight_e = JsbsimFdm(fdm_id=3, fdm_ic_y=0.04, fdm_ic_x=0.04, fdm_ic_z=18000)
        self.flight.append(flight_e)


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
            # x_ally =  self.state_before[i]["position/eci-x-ft"]*2 + 300  # x
            # y_ally = self.state_before[i]["position/eci-y-ft"]*2 + 300  # y
            x_ally = self.state_before[i]["position/lat-gc-deg"] * 2000 + 300  # x
            y_ally = self.state_before[i]["position/long-gc-deg"] * 2000 + 300  # y

            theta_ally = self.state_before[i]["attitude/psi-rad"]  # theta
            self.plane_allay[i].set_translation(x_ally, y_ally)  # 设置位置变换
            self.plane_allay[i].set_rotation(theta_ally)  # 设置角度变换
        #
        for i in range(self.n_enemies):  # 对手设置
            x_ally = self.state_before[i+self.n_agents]["position/lat-gc-deg"]*2000 + 300  # x
            y_ally = self.state_before[i+self.n_agents]["position/long-gc-deg"]*2000 + 300  # y
            theta_ally = self.state_before[i + self.n_agents]["attitude/psi-rad"]   # theta
            self.plane_enemy[i].set_translation(x_ally, y_ally)  # 设置位置变换
            self.plane_enemy[i].set_rotation(theta_ally)  # 设置角度变换



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
            # eci_x = flight.fdm["position/eci-x-ft"]
            # eci_y = flight.fdm["position/eci-y-ft"]
            # eci_z = flight.fdm["position/eci-z-ft"]
            lat = flight.fdm["position/lat-gc-deg"]
            long = flight.fdm["position/long-gc-deg"]
            h = flight.fdm["position/h-sl-ft"]


            # 速度
            v_north = flight.fdm["velocities/v-north-fps"]
            v_east = flight.fdm["velocities/v-east-fps"]
            v_down = flight.fdm["velocities/v-down-fps"]

            if not (
                    0 <= lat <= self.x_range and 0 <= long <= self.y_range and 0 <= h <= self.h_range):  # 如果越界，血量为0
                self.flight[i].fdm_hp = 0

            # if not (0 <= eci_x <= self.x_range and 0 <= eci_y <= self.y_range and 0 <= eci_z <= self.h_range):  # 如果越界，血量为0
            #     self.flight[i].fdm_hp = 0


            # 如果达成攻击态势，目标结果hp为0
            # 攻击态势，当前飞机和敌机的距离， 角度

            e_id = i + self.n_agents
            flight_e = self.flight[e_id]
            # e_eci_x = flight_e.fdm["position/eci-x-ft"]
            # e_eci_y = flight_e.fdm["position/eci-y-ft"]
            # e_eci_z = flight_e.fdm["position/eci-z-ft"]
            e_lat = flight_e.fdm["position/lat-gc-deg"]
            e_long = flight_e.fdm["position/long-gc-deg"]
            e_h = flight_e.fdm["position/h-sl-ft"]

            dist = self.distance_weidu(lat, long, h, e_lat, e_long, e_h)
            ATA = self.ATA_weidu(lat, long, h, v_north, v_east, v_down, e_lat, e_long, e_h) / np.pi  # ATA

            # dist = self.distance(eci_x, eci_y, eci_z, e_eci_x, e_eci_y, e_eci_z)
            # ATA = self.ATA(eci_x, eci_y, eci_z, v_north, v_east, v_down, e_eci_x , e_eci_y, e_eci_z) / np.pi  # ATA

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


        # 首先计算 血量奖励  , 血量奖励只关于最终结果
        reward_hp = 0
        for f  in self.flight :
            if f.fdm_hp == 0 :
                reward_hp += 3000

        # if  self.flight[1].fdm_hp == 0    :
        #     reward_hp += 3000




        reward_distance = np.zeros(self.n_agents)
        reward_angle = np.zeros(self.n_agents)

        for a in range(self.n_agents):
            a_e =  a+self.n_agents

            flight = self.flight[a]
            # eci_x = flight.fdm["position/eci-x-ft"]
            # eci_y = flight.fdm["position/eci-y-ft"]
            # eci_z = flight.fdm["position/eci-z-ft"]
            lat = flight.fdm["position/lat-gc-deg"]
            long = flight.fdm["position/long-gc-deg"]
            h = flight.fdm["position/h-sl-ft"]

            # 速度
            v_north = flight.fdm["velocities/v-north-fps"]
            v_east = flight.fdm["velocities/v-east-fps"]
            v_down = flight.fdm["velocities/v-down-fps"]

            flight_e = self.flight[a_e]
            # e_eci_x = flight_e.fdm["position/eci-x-ft"]
            # e_eci_y = flight_e.fdm["position/eci-y-ft"]
            # e_eci_z = flight_e.fdm["position/eci-z-ft"]
            e_lat = flight_e.fdm["position/lat-gc-deg"]
            e_long = flight_e.fdm["position/long-gc-deg"]
            e_h = flight_e.fdm["position/h-sl-ft"]

            # 当前状态下位置 a —— e

            # dist_a_e_n = self.distance(eci_x, eci_y, eci_z, e_eci_x, e_eci_y, e_eci_z)
            dist_a_e_n = self.distance_weidu(lat, long, e_h, e_lat, e_long, e_h)


            # 之前状态下位置 a —— e
            # dist_a_e_b = self.distance(self.state_before[a]["position/eci-x-ft"], self.state_before[a]["position/eci-y-ft"],self.state_before[a]["position/eci-z-ft"],
            #                            self.state_before[a_e]["position/eci-x-ft"], self.state_before[a_e]["position/eci-y-ft"],self.state_before[a_e]["position/eci-z-ft"])
            dist_a_e_b = self.distance_weidu(self.state_before[a]["position/lat-gc-deg"],
                                       self.state_before[a]["position/long-gc-deg"],
                                       self.state_before[a]["position/h-sl-ft"],
                                       self.state_before[a_e]["position/lat-gc-deg"],
                                       self.state_before[a_e]["position/long-gc-deg"],
                                       self.state_before[a_e]["position/h-sl-ft"])


            if dist_a_e_b  > dist_a_e_n:  # 当前距离小于之前距离
                reward_distance[a] += 0.5
            else:
                reward_distance[a] -= 0.5

            reward_distance[a]  +=  2  - (dist_a_e_n / 200)
            # 当前状态下ATA a —— e
            # ATA_a_e_n = self.ATA_weidu(eci_x , eci_y, eci_z, v_north, v_east, v_down, e_eci_x, e_eci_y, e_eci_z) / np.pi
            ATA_a_e_n = self.ATA_weidu(lat, long, h, v_north, v_east, v_down, e_lat, e_long, e_h) / np.pi  # ATA

            # ATA_a_e_b = self.ATA(self.state_before[a].fdm_pos_x, self.state_before[a].fdm_pos_y, self.state_before[a].fdm_theta,
            #                      self.state_before[a_e].fdm_pos_x, self.state_before[a_e].fdm_pos_y)

            if ATA_a_e_n <= np.pi / 2:
                reward_angle[a] += 0.1
            else:
                reward_angle[a] -= 0.5

            if reward_distance[a] >= 1:
                reward_angle[a] += ((np.pi / 2) - (ATA_a_e_n)) / 2


        print(f'reward_hp: {reward_hp}')
        print(f'reward_distance: {reward_distance}')
        print(f'reward_angle: {reward_angle}')

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
        size_obs_allay = (self.n_agents - 1) * self.size_obs_allay
        size_obs_enemy = self.n_enemies *  self.size_obs_enemy
        size_total = size_obs_pos + size_obs_hp + size_obs_allay + size_obs_enemy

        return size_total

    # def get_obs_ally(self):
    #     """
    #     得到联盟飞机的所有个体全局信息，即去掉了观测范围的限制
    #     :return:
    #     """
    #     size_obs_pos = self.n_move_feats
    #     size_obs_hp = self.own_feats
    #     size_obs_allay = (self.n_agents - 1) * self.size_obs_allay
    #     size_obs_enemy = self.n_enemies * self.size_obs_enemy
    #     size_total = size_obs_pos + size_obs_hp + size_obs_allay + size_obs_enemy
    #
    #     state_all = np.zeros([self.n_agents, size_total])
    #
    #     # 我方两架飞机的部分观测信息
    #     for i in range(self.n_agents):
    #
    #         agent_id = i  #
    #         # obs_id = np.arange(size_total)  # 飞机编号对应的所有观测信息
    #         move_feats = np.zeros(size_obs_pos, dtype=np.float32)  # 位置，速度信息 x,y,v,theta
    #         own_feats = np.zeros(size_obs_hp, dtype=np.float32)  # 血量，last_action
    #         ally_feats = np.zeros([self.n_agents - 1, self.size_obs_allay], dtype=np.float32)  # 和联盟的相对位置关系
    #         enemy_feats = np.zeros([self.n_enemies, self.size_obs_enemy], dtype=np.float32)  # 和对手的位置关系
    #
    #         # 根据飞机编号 获得 pos_feats
    #         flight = self.flight[agent_id]
    #         if flight.fdm_hp > 0:  # 当飞机血量大于0的时候才记录信息
    #             # 位置 x y h
    #             x = flight.fdm["position/x-gc-deg"]
    #             y = flight.fdm["position/y-gc-deg"]
    #             h = flight.fdm["position/h-sl-ft"]
    #             # 角度 弧度制
    #             theta = flight.fdm["attitude/theta-rad"]
    #             psi = flight.fdm["attitude/psi-rad"]
    #             phi = flight.fdm["attitude/phi-rad"]
    #
    #             # 速度
    #             v_u = flight.fdm["velocities/u-fps"]
    #             v_v = flight.fdm["velocities/v-fps"]
    #             v_w = flight.fdm["velocities/w-fps"]
    #
    #             # 视线距离
    #             sight_range = flight.sight_range
    #
    #             # 对运动信息pos_feats进行归一化处理
    #             # 位置
    #             move_feats[0] = x / self.x_range
    #             move_feats[1] = y / self.y_range
    #             move_feats[2] = h / self.h_range
    #             # 姿态
    #             move_feats[3] = theta / np.pi
    #             move_feats[4] = psi / np.pi
    #             move_feats[5] = phi / np.pi
    #             # 速度
    #             move_feats[6] = v_u / self.vel_range
    #             move_feats[7] = v_v / self.vel_range
    #             move_feats[8] = v_w / self.vel_range
    #
    #             # 飞机的 hp 与last_action
    #             own_feats[0] = flight.fdm_hp
    #             own_feats[1:] = flight.last_action
    #
    #             # 获得联邦状态信息  联邦是在 n_angents 中 除了当前编号i 以外的所有飞机
    #             al_ids = [al_id for al_id in range(self.n_agents) if al_id != agent_id]
    #             for a, al_id in enumerate(al_ids):
    #                 flight_al = self.flight[al_id]  # 根据联盟编号获得飞机对象
    #                 # 联盟飞机的位置信息
    #                 # 位置 x y h
    #                 al_long = flight_al.fdm["position/x-gc-deg"]
    #                 al_lat = flight_al.fdm["position/y-gc-deg"]
    #                 al_h = flight_al.fdm["position/h-sl-ft"]
    #                 dist = self.distance(x, y, h, al_long, al_lat, al_h)
    #
    #                 if flight_al.fdm_hp > 0:
    #                     al_theta = flight_al.fdm["attitude/theta-rad"]
    #                     al_psi = flight_al.fdm["attitude/psi-rad"]
    #                     al_phi = flight_al.fdm["attitude/phi-rad"]
    #
    #                     al_u = flight_al.fdm["velocities/u-fps"]
    #                     al_v = flight_al.fdm["velocities/u-fps"]
    #                     al_w = flight_al.fdm["velocities/u-fps"]
    #
    #                     ally_feats[a, 0] = 1  # visible
    #
    #                     ally_feats[a, 1] = dist / sight_range  # distance
    #
    #                     ally_feats[a, 2] = (al_long - x) / sight_range  # relative X
    #                     ally_feats[a, 3] = (al_lat - y) / sight_range  # relative Y
    #                     ally_feats[a, 4] = (al_h - h) / sight_range  # relative H
    #
    #                     ally_feats[a, 5] = (al_theta - theta) / np.pi  # relative  theta
    #                     ally_feats[a, 6] = (al_psi - psi) / np.pi  # relative  psi
    #                     ally_feats[a, 7] = (al_phi - phi) / np.pi  # relative phi
    #
    #                     ally_feats[a, 8] = (al_u - v_u) / self.vel_range  # relative V_x
    #                     ally_feats[a, 9] = (al_v - v_v) / self.vel_range  # relative V_y
    #                     ally_feats[a, 10] = (al_w - v_w) / self.vel_range  # relative V_z
    #
    #                     ally_feats[a, 11] = self.ATA(x, y, h, v_u, v_v, v_w, al_long, al_lat, al_h) / np.pi  # ATA
    #                     ally_feats[a, 12] = self.AA(x, y, h, al_long, al_lat, al_h, al_u, al_v, al_w) / np.pi  # AA
    #
    #                     ally_feats[a, 13] = flight_al.fdm_hp  # health
    #                     ally_feats[a, 14:] = flight_al.last_action  # last_action
    #
    #             # 获得对手数据，在飞机编号中为 n_ally，；；，n_ally+n_ememies，即敌机0 对应飞机编号3
    #             for e in range(self.n_enemies):
    #                 enemy_id = e + self.n_agents
    #                 flight_e = self.flight[enemy_id]  # 根据联盟编号获得飞机对象
    #                 # 联盟飞机的位置信息
    #                 # 位置 x y h
    #                 e_long = flight_e.fdm["position/x-gc-deg"]
    #                 e_lat = flight_e.fdm["position/y-gc-deg"]
    #                 e_h = flight_e.fdm["position/h-sl-ft"]
    #                 dist = self.distance(x, y, h, e_long, e_lat, e_h)
    #
    #                 if flight_e.fdm_hp > 0:  # visible and alive
    #
    #                     e_theta = flight_e.fdm["attitude/theta-rad"]
    #                     e_psi = flight_e.fdm["attitude/psi-rad"]
    #                     e_phi = flight_e.fdm["attitude/phi-rad"]
    #
    #                     e_u = flight_e.fdm["velocities/u-fps"]
    #                     e_v = flight_e.fdm["velocities/u-fps"]
    #                     e_w = flight_e.fdm["velocities/u-fps"]
    #
    #                     enemy_feats[e, 0] = 1  # visible
    #
    #                     enemy_feats[e, 1] = dist / sight_range  # distance
    #
    #                     enemy_feats[e, 2] = (e_long - x) / sight_range  # relative X
    #                     enemy_feats[e, 3] = (e_lat - y) / sight_range  # relative Y
    #                     enemy_feats[e, 4] = (e_h - h) / sight_range  # relative H
    #
    #                     enemy_feats[e, 5] = (e_theta - theta) / np.pi  # relative  theta
    #                     enemy_feats[e, 6] = (e_psi - psi) / np.pi  # relative  psi
    #                     enemy_feats[e, 7] = (e_phi - phi) / np.pi  # relative phi
    #
    #                     enemy_feats[e, 8] = (e_u - v_u) / self.vel_range  # relative V_x
    #                     enemy_feats[e, 9] = (e_v - v_v) / self.vel_range  # relative V_y
    #                     enemy_feats[e, 10] = (e_w - v_w) / self.vel_range  # relative V_z
    #
    #                     enemy_feats[e, 11] = self.ATA(x, y, h, v_u, v_v, v_w, e_long, e_lat, e_h) / np.pi  # ATA
    #                     enemy_feats[e, 12] = self.AA(x, y, h, e_long, e_lat, e_h, e_u, e_v, e_w) / np.pi  # AA
    #
    #                     enemy_feats[e, 13] = flight_e.fdm_hp  # health
    #
    #         obs_id = np.concatenate((
    #             move_feats.flatten(),
    #             own_feats.flatten(),
    #             ally_feats.flatten(),
    #             enemy_feats.flatten()
    #
    #         ))
    #         state_all[i] = obs_id
    #
    #     return state_all

    def get_state_ally(self):
        """
        得到联盟飞机的所有个体全局信息，即去掉了观测范围的限制
        :return:
        """
        size_obs_pos = self.n_move_feats
        size_obs_hp = self.own_feats
        size_obs_allay = (self.n_agents - 1) * self.size_obs_allay
        size_obs_enemy = self.n_enemies * self.size_obs_enemy
        size_total = size_obs_pos + size_obs_hp + size_obs_allay + size_obs_enemy

        state_all = np.zeros([self.n_agents, size_total])

        # 我方两架飞机的部分观测信息
        for i in range(self.n_agents):

            agent_id = i  #
            # obs_id = np.arange(size_total)  # 飞机编号对应的所有观测信息
            move_feats = np.zeros(size_obs_pos, dtype=np.float32)  # 位置，速度信息 x,y,v,theta
            own_feats = np.zeros(size_obs_hp, dtype=np.float32)  # 血量，last_action
            ally_feats = np.zeros([self.n_agents - 1, self.size_obs_allay], dtype=np.float32)  # 和联盟的相对位置关系
            enemy_feats = np.zeros([self.n_enemies, self.size_obs_enemy], dtype=np.float32)  # 和对手的位置关系

            # 根据飞机编号 获得 pos_feats
            flight = self.flight[agent_id]
            if flight.fdm_hp > 0:  # 当飞机血量大于0的时候才记录信息
                # 位置 x y h
                long = flight.fdm["position/eci-x-ft"]
                lat = flight.fdm["position/eci-y-ft"]
                h = flight.fdm["position/eci-z-ft"]
                # 角度 弧度制
                theta = flight.fdm["attitude/theta-rad"]
                psi = flight.fdm["attitude/psi-rad"]
                phi = flight.fdm["attitude/phi-rad"]


               #速度
                v_u = flight.fdm["velocities/u-fps"]
                v_v = flight.fdm["velocities/v-fps"]
                v_w = flight.fdm["velocities/w-fps"]

                # 视线距离
                sight_range = flight.sight_range


                # 对运动信息pos_feats进行归一化处理
                # 位置
                move_feats[0] = long / self.x_range
                move_feats[1] = lat / self.y_range
                move_feats[2] = h / self.h_range
                # 姿态
                move_feats[3] = theta / np.pi
                move_feats[4] = psi / np.pi
                move_feats[5] = phi / np.pi
                # 速度
                move_feats[6] = v_u / self.vel_range
                move_feats[7] = v_v / self.vel_range
                move_feats[8] = v_w / self.vel_range





                # 飞机的 hp 与last_action
                own_feats[0] = flight.fdm_hp
                own_feats[1:] = flight.last_action

                # 获得联邦状态信息  联邦是在 n_angents 中 除了当前编号i 以外的所有飞机
                al_ids = [al_id for al_id in range(self.n_agents) if al_id != agent_id]
                for a, al_id in enumerate(al_ids):
                    flight_al = self.flight[al_id]  # 根据联盟编号获得飞机对象
                    # 联盟飞机的位置信息
                    # 位置 x y h
                    al_long = flight_al.fdm["position/eci-x-ft"]
                    al_lat = flight_al.fdm["position/eci-y-ft"]
                    al_h = flight_al.fdm["position/eci-z-ft"]
                    dist = self.distance(long, lat, h,al_long, al_lat,al_h)

                    if flight_al.fdm_hp > 0:

                        al_theta = flight_al.fdm["attitude/theta-rad"]
                        al_psi = flight_al.fdm["attitude/psi-rad"]
                        al_phi = flight_al.fdm["attitude/phi-rad"]

                        al_u =  flight_al.fdm["velocities/u-fps"]
                        al_v = flight_al.fdm["velocities/u-fps"]
                        al_w = flight_al.fdm["velocities/u-fps"]


                        ally_feats[a, 0] = 1  # visible

                        ally_feats[a, 1] = dist / sight_range  # distance

                        ally_feats[a, 2] = (al_long - long) / sight_range  # relative X
                        ally_feats[a, 3] = (al_lat - lat) / sight_range  # relative Y
                        ally_feats[a, 4] = (al_h - h) / sight_range  # relative H

                        ally_feats[a, 5] = (al_theta - theta) / np.pi  # relative  theta
                        ally_feats[a, 6] = (al_psi - psi) / np.pi   # relative  psi
                        ally_feats[a, 7] = (al_phi - phi) / np.pi   # relative phi

                        ally_feats[a, 8] = (al_u - v_u) / self.vel_range  # relative V_x
                        ally_feats[a, 9] = (al_v - v_v) / self.vel_range  # relative V_y
                        ally_feats[a, 10] = (al_w - v_w) / self.vel_range  # relative V_z



                        ally_feats[a, 11] = self.ATA(long,lat,h,v_u,v_v,v_w,al_long,al_lat,al_h) / np.pi  # ATA
                        ally_feats[a, 12] = self.AA(long,lat,h,al_long,al_lat,al_h,al_u,al_v,al_w) / np.pi  # AA

                        ally_feats[a, 13] = flight_al.fdm_hp  # health
                        ally_feats[a, 14:] = flight_al.last_action  # last_action

                # 获得对手数据，在飞机编号中为 n_ally，；；，n_ally+n_ememies，即敌机0 对应飞机编号3
                for e in range(self.n_enemies):
                    enemy_id = e + self.n_agents
                    flight_e = self.flight[enemy_id]  # 根据联盟编号获得飞机对象
                    # 联盟飞机的位置信息
                    # 位置 x y h
                    e_long = flight_e.fdm["position/eci-x-ft"]
                    e_lat = flight_e.fdm["position/eci-y-ft"]
                    e_h = flight_e.fdm["position/eci-z-ft"]
                    dist = self.distance(long, lat, h, e_long, e_lat, e_h)

                    if flight_e.fdm_hp > 0:  # visible and alive

                        e_theta = flight_e.fdm["attitude/theta-rad"]
                        e_psi = flight_e.fdm["attitude/psi-rad"]
                        e_phi = flight_e.fdm["attitude/phi-rad"]

                        e_u = flight_e.fdm["velocities/u-fps"]
                        e_v = flight_e.fdm["velocities/u-fps"]
                        e_w = flight_e.fdm["velocities/u-fps"]

                        enemy_feats[e, 0] = 1  # visible

                        enemy_feats[e, 1] = dist / sight_range  # distance

                        enemy_feats[e, 2] = (e_long - long) / sight_range  # relative X
                        enemy_feats[e, 3] = (e_lat - lat) / sight_range  # relative Y
                        enemy_feats[e, 4] = (e_h - h) / sight_range  # relative H

                        enemy_feats[e, 5] = (e_theta - theta) / np.pi  # relative  theta
                        enemy_feats[e, 6] = (e_psi - psi) / np.pi  # relative  psi
                        enemy_feats[e, 7] = (e_phi - phi) / np.pi  # relative phi

                        enemy_feats[e, 8] = (e_u - v_u) / self.vel_range  # relative V_x
                        enemy_feats[e, 9] = (e_v - v_v) / self.vel_range  # relative V_y
                        enemy_feats[e, 10] = (e_w - v_w) / self.vel_range  # relative V_z

                        enemy_feats[e, 11] = self.ATA(long, lat, h, v_u, v_v, v_w, e_long, e_lat, e_h) / np.pi  # ATA
                        enemy_feats[e, 12] = self.AA(long, lat, h, e_long, e_lat, e_h, e_u, e_v, e_w) / np.pi  # AA

                        enemy_feats[e, 13] = flight_e.fdm_hp  # health






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
        return math.hypot(x2 - x1, y2 - y1,z2-z1)

    @staticmethod
    def ATA(long,lat,h,v_u,v_v,v_w,al_long,al_lat,al_h):
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
        pos_vector = np.array([al_long - long, al_lat - lat, al_h-h])
        v_vector = np.array([v_u,v_v,v_w])
        cos_radian = np.dot(pos_vector, v_vector) / (np.linalg.norm(pos_vector) * np.linalg.norm(v_vector) )
        # cos theta = 向量a 点乘 向量b  /   向量a的模 乘以 向量b的模
        cos_radian = np.clip(cos_radian, -1, 1)  # 进行裁剪

        radian = np.arccos(cos_radian)  # arccos结果为[0,pi]的弧度制

        ATA = np.clip(radian,0.01,np.pi)
        return ATA

    @staticmethod
    def AA(long,lat,h,al_long,al_lat,al_h,al_u,al_v,al_w):
        """
        计算两机位置矢量与 速度的夹角 ，敌方速度矢量与位置矢量夹角
        :param x:
        :param y:
        :param theta:
        :param x1:
        :param y1:
        :return:
        """
        # 首先计算 两机位置矢量与 x轴(向量【1，0】)的夹角
        pos_vector = np.array([al_long - long, al_lat - lat, al_h-h])
        v_vector = np.array([al_u,al_v,al_w])
        cos_radian = np.dot(pos_vector, v_vector) / (
            np.linalg.norm(pos_vector))  # cos theta = 向量a 点乘 向量b  /   向量a的模 乘以 向量b的模

        cos_radian = np.clip(cos_radian, -1, 1)  # 进行裁剪

        radian = np.arccos(cos_radian)  # arccos结果为[0,pi]的弧度制

        AA = radian
        AA =  np.clip(AA, 0.01, np.pi)
        return AA

    @staticmethod
    def VV(theta0, theta1):
        """
        计算两个飞机的航线角
        """
        # 首先计算 两机位置矢量与 x轴(向量【1，0】)的夹角
        v_vector_0 = np.array([np.cos(theta0), np.sin(theta0)])
        v_vector_1 = np.array([np.cos(theta1), np.sin(theta1)])
        cos_radian = np.dot(v_vector_0, v_vector_1)  # cos theta = 向量a 点乘 向量b  /   向量a的模 乘以 向量b的模

        cos_radian = np.clip(cos_radian, -1, 1)  # 进行裁剪

        radian = np.arccos(cos_radian)  # arccos结果为[0,pi]的弧度制
        return radian

    @staticmethod
    def distance_weidu(x1, y1,z1, x2, y2,z2):
        """Distance between two points."""
        return math.hypot( (x2 - x1)*111195 , (y2 - y1)*111195, (z2-z1)*111195)
    @staticmethod
    def ATA_weidu(long,lat,h,v_u,v_v,v_w,al_long,al_lat,al_h):
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
        pos_vector = np.array([(al_long - long)*111.195, (al_lat - lat)*111.195, (al_h-h)*111.195])
        v_vector = np.array([v_u,v_v,v_w])
        cos_radian = np.dot(pos_vector, v_vector) / (np.linalg.norm(pos_vector) * np.linalg.norm(v_vector) )
        # cos theta = 向量a 点乘 向量b  /   向量a的模 乘以 向量b的模
        cos_radian = np.clip(cos_radian, -1, 1)  # 进行裁剪

        radian = np.arccos(cos_radian)  # arccos结果为[0,pi]的弧度制

        ATA = np.clip(radian,0.01,np.pi)
        return ATA