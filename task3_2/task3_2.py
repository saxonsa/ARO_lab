import subprocess, math, time, sys, os, numpy as np
import matplotlib.pyplot as plt
import pybullet as bullet_simulation
import pybullet_data

# setup paths and load the core
abs_path = os.path.dirname(os.path.realpath(__file__))
root_path = abs_path + '/..'
core_path = root_path + '/core'
sys.path.append(core_path)
from Pybullet_Simulation import Simulation

# specific settings for this task

taskId = 3.2

try:
    if sys.argv[1] == 'nogui':
        gui = False
    else:
        gui = True
except:
    gui = True

pybulletConfigs = {
    "simulation": bullet_simulation,
    "pybullet_extra_data": pybullet_data,
    "gui": gui,
    "panels": False,
    "realTime": False,
    "controlFrequency": 1000,
    "updateFrequency": 250,
    "gravity": -9.81,
    "gravityCompensation": 1.,
    "floor": True,
    "cameraSettings": (1.2, 90, -22.8, (-0.12, -0.01, 0.99))
}
robotConfigs = {
    "robotPath": core_path + "/nextagea_description/urdf/NextageaOpen.urdf",
    "robotPIDConfigs": core_path + "/PD_gains.yaml",
    "robotStartPos": [0, 0, 0.85],
    "robotStartOrientation": [0, 0, 0, 1],
    "fixedBase": True,
    "colored": True
}

sim = Simulation(pybulletConfigs, robotConfigs)

##### Please leave this function unchanged, feel free to modify others #####
def getReadyForTask():
    global finalTargetPos
    global taleId, cubeId, targetId, obstacle
    finalTargetPos = np.array([0.35,0.38,1.0])
    # compile target urdf
    urdf_compiler_path = core_path + "/urdf_compiler.py"
    subprocess.call([urdf_compiler_path,
                     "-o", abs_path+"/lib/task_urdfs/task3_2_target_compiled.urdf",
                     abs_path+"/lib/task_urdfs/task3_2_target.urdf"])

    sim.p.resetJointState(bodyUniqueId=1, jointIndex=12, targetValue=-0.4)
    sim.p.resetJointState(bodyUniqueId=1, jointIndex=6, targetValue=-0.4)

    # load the table in front of the robot
    tableId = sim.p.loadURDF(
        fileName              = abs_path+"/lib/task_urdfs/table/table_taller.urdf",
        basePosition          = [0.8, 0, 0],             
        baseOrientation       = sim.p.getQuaternionFromEuler([0,0,math.pi/2]),                                  
        useFixedBase          = True,             
        globalScaling         = 1.4
    )
    cubeId = sim.p.loadURDF(
        fileName              = abs_path+"/lib/task_urdfs/cubes/task3_2_dumb_bell.urdf", 
        basePosition          = [0.5, 0, 1.1],            
        baseOrientation       = sim.p.getQuaternionFromEuler([0,0,0]),                                  
        useFixedBase          = False,             
        globalScaling         = 1.4
    )
    sim.p.resetVisualShapeData(cubeId, -1, rgbaColor=[1,1,0,1])
    
    targetId = sim.p.loadURDF(
        fileName              = abs_path+"/lib/task_urdfs/task3_2_target_compiled.urdf",
        basePosition          = finalTargetPos,             
        baseOrientation       = sim.p.getQuaternionFromEuler([0,0,math.pi/4]), 
        useFixedBase          = True,             
        globalScaling         = 1
    )
    obstacle = sim.p.loadURDF(
        fileName              = abs_path+"/lib/task_urdfs/cubes/task3_2_obstacle.urdf",
        basePosition          = [0.43,0.275,0.9],             
        baseOrientation       = sim.p.getQuaternionFromEuler([0,0,math.pi/4]), 
        useFixedBase          = True,             
        globalScaling         = 1
    )

    for _ in range(300):
        # sim.tick()
        time.sleep(1./1000)

    return tableId, cubeId, targetId


def solution():

    paths = [
        {
            'left': 'LARM_JOINT5',
            'leftTargetPosition': np.array([0.45, 0.015, 1.08]),
            'leftOrientation': np.array([0.98, 0, 1]),
            'right': 'RARM_JOINT5',
            'rightTargetPosition': np.array([0.45, -0.015, 1.08]),
            'rightOrientation': np.array([0.98, 0, 1]),
            'iterNum': 50
        },
        {
            'left': 'LARM_JOINT5',
            'leftTargetPosition': np.array([0.45, 0.1, 1.11]),
            'leftOrientation': np.array([0.99, 0, 1.1]),
            'right': 'RARM_JOINT5',
            'rightTargetPosition': np.array([0.45, -0.1, 1.11]),
            'rightOrientation': np.array([0.99, 0, 1.1]),
            'iterNum': 25
        },
        {
            'left': 'LARM_JOINT5',
            'leftTargetPosition': np.array([0.45, 0.5, 1.11]),
            'leftOrientation': np.array([0.99, 0, 1.1]),
            'right': 'RARM_JOINT5',
            'rightTargetPosition': np.array([0.45, 0.3, 1.11]),
            'rightOrientation': np.array([0.99, 0, 1.1]),
            'iterNum': 25
        },
        # {
        #     'left': 'LARM_JOINT5',
        #     'leftTargetPosition': np.array([0.45, 0.015, 1.1]),
        #     'leftOrientation': np.array([0.99, 0, 1.00]),
        #     'right': 'RARM_JOINT5',
        #     'rightTargetPosition': np.array([0.45, -0.015, 1.1]),
        #     'rightOrientation': np.array([0.99, 0, 1.00]),
        #     'iterNum': 25
        # },
        # {
        #     'left': 'LARM_JOINT5',
        #     'leftTargetPosition': np.array([0.45, 0.015, 1.10]),
        #     'leftOrientation': np.array([0.98, 0, 0.96]),
        #     'right': 'RARM_JOINT5',
        #     'rightTargetPosition': np.array([0.45, -0.02, 1.10]),
        #     'rightOrientation': np.array([0.98, 0, 0.96]),
        #     'iterNum': 25
        # },
        # {
        #     'left': 'LARM_JOINT5',
        #     'leftTargetPosition': np.array([0.45, 0.015, 1.15]),
        #     'leftOrientation': np.array([0.98, 0, 0.96]),
        #     'right': 'RARM_JOINT5',
        #     'rightTargetPosition': np.array([0.45, -0.02, 1.15]),
        #     'rightOrientation': np.array([0.98, 0, 0.96]),
        #     'iterNum': 25
        # },

    ]

    for path in paths:
        sim.clamp(leftEndEffector=path['left'], leftTargetPosition=path['leftTargetPosition'], leftOrientation=path['leftOrientation'], rightEndEffector=path['right'], rightTargetPosition=path['rightTargetPosition'], rightOrientation=path['rightOrientation'], iterNum=path['iterNum'])


tableId, cubeId, targetId = getReadyForTask()
solution()


try:
    time.sleep(float(sys.argv[1]))
except:
    time.sleep(10)
