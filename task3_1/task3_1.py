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

taskId = 3.1

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
    "cameraSettings": (1.07, 90.0, -52.8, (0.07, 0.01, 0.76))
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
    # compile urdfs
    finalTargetPos = np.array([0.7, 0.00, 0.91])
    urdf_compiler_path = core_path + "/urdf_compiler.py"
    subprocess.call([urdf_compiler_path,
                     "-o", abs_path+"/lib/task_urdfs/task3_1_target_compiled.urdf",
                     abs_path+"/lib/task_urdfs/task3_1_target.urdf"])

    sim.p.resetJointState(bodyUniqueId=1, jointIndex=12, targetValue=-0.4)
    sim.p.resetJointState(bodyUniqueId=1, jointIndex=6, targetValue=-0.4)
    # load the table in front of the robot
    tableId = sim.p.loadURDF(
        fileName            = abs_path+"/lib/task_urdfs/table/table_taller.urdf",
        basePosition        = [0.8, 0, 0],
        baseOrientation     = sim.p.getQuaternionFromEuler([0, 0, math.pi/2]),
        useFixedBase        = True,
        globalScaling       = 1.4
    )
    cubeId = sim.p.loadURDF(
        fileName            = abs_path+"/lib/task_urdfs/cubes/cube_small.urdf",
        basePosition        = [0.33, 0, 1.0],
        baseOrientation     = sim.p.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase        = False,
        globalScaling       = 1.4
    )
    sim.p.resetVisualShapeData(cubeId, -1, rgbaColor=[1, 1, 0, 1])

    targetId = sim.p.loadURDF(
        fileName            = abs_path+"/lib/task_urdfs/task3_1_target_compiled.urdf",
        basePosition        = finalTargetPos,
        baseOrientation     = sim.p.getQuaternionFromEuler([0, 0, math.pi]),
        useFixedBase        = True,
        globalScaling       = 1
    )
    for _ in range(200):
        # sim.tick()
        time.sleep(1./1000)

    print('distance!!', sim.p.getBasePositionAndOrientation(cubeId)[0])
    

    return tableId, cubeId, targetId


def solution():

    paths = [
        {
            'left': 'LARM_JOINT5',
            'leftTargetPosition': [0.20, 0.23, 1.03],
            'right': 'RARM_JOINT5',
            'rightTargetPosition': [0.20, -0.23, 1.03],
            'iterNum': 25
        },
        {
            'left': 'LARM_JOINT5',
            'leftTargetPosition': [0.24, 0.12, 1.03],
            'right': 'RARM_JOINT5',
            'rightTargetPosition': [0.24, -0.12, 1.03],
            'iterNum': 25
        },
        {
            'left': 'LARM_JOINT5',
            'leftTargetPosition': [0.24, 0.10, 0.98],
            'right': 'RARM_JOINT5',
            'rightTargetPosition': [0.24, -0.10, 0.98],
            'iterNum': 10
        },
            {
            'left': 'LARM_JOINT5',
            'leftTargetPosition': [0.22, 0.10, 0.97],
            'right': 'RARM_JOINT5',
            'rightTargetPosition': [0.22, -0.10, 0.97],
            'iterNum': 25
        },
        {
            'left': 'LARM_JOINT5',
            'leftTargetPosition': [0.22, 0.085, 0.95],
            'right': 'RARM_JOINT5',
            'rightTargetPosition': [0.22, -0.085, 0.95],
            'iterNum': 10
        },
        {
            'left': 'LARM_JOINT5',
            'leftTargetPosition': [0.594, 0.0415, 0.9265],
            'right': 'RARM_JOINT5',
            'rightTargetPosition': [0.594, -0.0415, 0.9265], # 594 0415 9265
            'iterNum': 32
        },
    ]

    for path in paths:
        sim.selfDockingToPosition(leftEndEffector=path['left'], leftTargetPosition=path['leftTargetPosition'], rightEndEffector=path['right'], rightTargetPosition=path['rightTargetPosition'], iterNum=path['iterNum'])
    
    # test current position: 0.01973340798318625
    cubic_position = sim.p.getBasePositionAndOrientation(cubeId)[0]
    target_position = sim.p.getBasePositionAndOrientation(targetId)[0]
    print('distance', np.linalg.norm(np.asarray(cubic_position) - np.asarray(target_position)))

tableId, cubeId, targetId = getReadyForTask()

solution()
try:
    time.sleep(float(sys.argv[1]))
except:
    time.sleep(10)
