# -*- coding: utf-8 -*-

"""
@author     : zuti
@software   : PyCharm
@file       : testFdm.py
@time       : 24/05/2023 15:51
@desc       ：

"""
import time

import jsbsim
import numpy as np

# from envs.JsbSimEnv.jsbsimFdm import JsbsimFdm
#
# flight = []
# # 我方飞机位置限定在【1000,2000】*【5000，5500】
# for a in range(2):
#     rnd_eci_x = 0
#     rnd_eci_y = np.random.uniform(1000, 2000)
#     rnd_eci_z = np.random.uniform(10000, 12000)
#
#     rnd_theta = 0
#     rnd_psi = 0
#     rnd_phi = 0
#     rnd_v = np.random.uniform(200, 250)
#
#     # init_pos_aly = [rnd_lat,rnd_eci_y,rand_h,rnd_theta,rnd_psi,rnd_phi,rnd_v]
#
#     # init_state.append(init_pos_aly)
#
#     # 初始化飞机
#     flight_a = JsbsimFdm(fdm_id=a, fdm_ic_x=rnd_eci_x, fdm_ic_y=rnd_eci_y, fdm_ic_z=rnd_eci_z)
#     flight.append(flight_a)
#
# flight = flight[0]
# # 位置 x y h
# x = flight.fdm["position/eci-x-ft"]
# y = flight.fdm["position/eci-y-ft"]
# z = flight.fdm["position/eci-z-ft"]
# # 角度 弧度制
# theta = flight.fdm["attitude/theta-rad"]
# psi = flight.fdm["attitude/psi-rad"]
# phi = flight.fdm["attitude/phi-rad"]
#
#
# #速度
# v_u = flight.fdm["velocities/u-fps"]
# v_v = flight.fdm["velocities/v-fps"]
# v_w = flight.fdm["velocities/w-fps"]
#
# print(f'{x , y, z}')
#
# result =  flight.step()

fdm = jsbsim.FGFDMExec(None)
fdm.load_model('f16')
fdm.set_output_directive('./data_output/flightgear2.xml')

# 设置第一架飞机的初始位置
fdm.set_property_value('position/latitude-deg', 0.01)
fdm.set_property_value('position/longitude-deg', 0.03)
fdm.set_property_value('position/altitude-ft', 10000.0)
fdm.set_property_value('ic/eci-z-ft', 10000.0)


# 设置其他初始状态，如姿态、速度等


# fdm["ic/eci-x-ft"] = 1000
# fdm["ic/eci-y-ft"] = 1000
# fdm["ic/eci-z-ft"] = 1000
#
# # Attitude Initialization 姿态初始化
# fdm["ic/psi-true-deg"] = 0
# fdm["ic/theta-deg"] = 0
# fdm["ic/phi-deg"] = 0
#
fdm['ic/vc-kts'] = 200
#
print('1----------------')
print(f' position/latitude-deg:{fdm["position/latitude-deg"]}')
print(f' position/latitude-deg:{fdm["position/longitude-deg"]}')
print(f' position/h-sl-f:{fdm["position/altitude-ft"]}')
fdm.run_ic()
fdm["propulsion/starter_cmd"] = 1
fdm["propulsion/refuel"] = 1
#
#
# print('2----------------')
# print(f' ic/eci-x-ft:{fdm["ic/eci-x-ft"]}')
# print(f'""ic/eci-z-ft": {fdm["ic/eci-z-ft"]}')
# print(f'"position/eci-z-ft": {fdm["position/eci-z-ft"]}')
result = fdm.run()
fdm["propulsion/active_engine"] = True
fdm["propulsion/set-running"] = -1

print(result)
# print('3----------------')
# print(f' ic/eci-x-ft:{fdm["ic/eci-x-ft"]}')
# print(f'""ic/eci-z-ft": {fdm["ic/eci-z-ft"]}')
# print(f'"position/eci-z-ft": {fdm["position/eci-z-ft"]}')

while True :
    fdm["fcs/aileron-cmd-norm"] = 1
    fdm["fcs/elevator-cmd-norm"] =0
    # fdm["fcs/rudder-cmd-norm"] = 0
    fdm.set_property_value("fcs/rudder-cmd-norm", 1)
    fdm.set_property_value("fcs/throttle-cmd-norm",1)
    #fdm["fcs/throttle-cmd-norm"] = 1
    for _ in range(100):
        result = fdm.run()

    print(f' position/latitude-deg:{fdm.get_property_value("position/altitude-ft")}')
    print(f' position/latitude-deg:{fdm["position/longitude-deg"]}')
    print(f' position/latitude-deg:{fdm["position/latitude-deg"]}')

    time.sleep(0.01)
    # print('3----------------')
    # print(f' ic/eci-x-ft:{fdm["ic/eci-x-ft"]}')
    # print(f'""ic/eci-z-ft": {fdm["ic/eci-z-ft"]}')
    # print(f'"position/eci-z-ft": {fdm["position/eci-z-ft"]}')
    # print(f'"position/eci-x-ft": {fdm["position/eci-x-ft"]}')
    # print(f'"position/eci-y-ft": {fdm["position/eci-y-ft"]}')