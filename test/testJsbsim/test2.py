# -*- coding: utf-8 -*-

"""
@author     : zuti
@software   : PyCharm
@file       : test2.py
@time       : 18/06/2023 05:39
@desc       ：

"""
import jsbsim

.

# FDM Initialization 空气动力学模型初始化
fdm = jsbsim.FGFDMExec(None)

# Aircraft Loading 加载飞机模型
fdm.load_model('f16')

# FlightGear Visualization 可视化

fdm.set_output_directive('./data_output/flightgear{}.xml'.format(0))

# Velocity Initialization 速度初始化
fdm['ic/vc-kts'] = fdm_ic_v

# Position Initialization 位置初始化
fdm["ic/lat-gc-deg"] = fdm_ic_lat
fdm["ic/long-gc-deg"] = fdm_ic_long
fdm["ic/h-sl-ft"] = fdm_ic_h

# Attitude Initialization 姿态初始化
fdm["ic/psi-true-deg"] = fdm_ic_psi
fdm["ic/theta-deg"] = fdm_ic_theta
fdm["ic/phi-deg"] = fdm_ic_phi

##########################
## Model Initialization ##
## 模型初始化            ##
fdm.run_ic()  ##
##########################

# Turning on the Engine 启动引擎
fdm["propulsion/starter_cmd"] = 1

# First but not Initial 第一帧设置（初始状态）
fdm.run()
fdm["propulsion/active_engine"] = True
fdm["propulsion/set-running"] = -1

# Number of Frames 帧数
nof = 1

