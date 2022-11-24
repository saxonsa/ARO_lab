import subprocess, math, time, sys, os, numpy as np
from turtle import right
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
    
    # lift it up
    paths = [
        {
            'left': 'LARM_JOINT5',
            'leftTargetPosition': np.array([0.50497461, 0.23, 1.05]),
            'leftOrientation': None,
            'right': 'RARM_JOINT5',
            'rightTargetPosition': np.array([0.50497461, -0.23, 1.05]),
            'rightOrientation': None,
            'iterNum': 10
        },
        {
            'left': 'LARM_JOINT5',
            'leftTargetPosition': np.array([0.42, 0.23, 1.05]),
            'leftOrientation': None,
            'right': 'RARM_JOINT5',
            'rightTargetPosition': np.array([0.42, -0.23, 1.05]),
            'rightOrientation': None,
            'iterNum': 10
        },
        {
            'left': 'LARM_JOINT5',
            'leftTargetPosition': np.array([0.42, 0.08, 1.05]),
            'leftOrientation': np.array([1, 0, 1]),
            'right': 'RARM_JOINT5',
            'rightTargetPosition': np.array([0.42, -0.08, 1.05]),
            'rightOrientation': np.array([1, 0, 1]),
            'iterNum': 10
        },
        {
            'left': 'LARM_JOINT5',
            'leftTargetPosition': np.array([0.42, 0.08, 1.18]),
            'leftOrientation': None,
            'right': 'RARM_JOINT5',
            'rightTargetPosition': np.array([0.42, -0.08, 1.18]),
            'rightOrientation': None,
            'iterNum': 20
        },
        {
            'left': 'LARM_JOINT5',
            'leftTargetPosition': np.array([0.48, 0.07, 1.18]),
            'leftOrientation': None,
            'right': 'RARM_JOINT5',
            'rightTargetPosition': np.array([0.48, -0.07, 1.18]),
            'rightOrientation': None,
            'iterNum': 20
        },
    ]
    for path in paths:
        sim.clamp(leftEndEffector=path['left'], leftTargetPosition=path['leftTargetPosition'], leftOrientation=path['leftOrientation'], rightEndEffector=path['right'], rightTargetPosition=path['rightTargetPosition'], rightOrientation=path['rightOrientation'], iterNum=path['iterNum'])

    path_forward = [
        {
            'left': 'LARM_JOINT5',
            'leftTargetPosition': np.array([0.50, 0.08, 1.18]),
            'leftOrientation': None,
            'right': 'RARM_JOINT5',
            'rightTargetPosition': np.array([0.50, -0.08, 1.18]),
            'rightOrientation': None,
            'iterNum': 10
        },
    ]
    for path in path_forward:
        sim.clamp(leftEndEffector=path['left'], leftTargetPosition=path['leftTargetPosition'], leftOrientation=path['leftOrientation'], rightEndEffector=path['right'], rightTargetPosition=path['rightTargetPosition'], rightOrientation=path['rightOrientation'], iterNum=path['iterNum'])

    # Interpolate the path through cubic spline function
    currentLeftPosition = sim.getJointPosition(jointName='LARM_JOINT5').T[0][:2]
    currentRightPosition = sim.getJointPosition(jointName='RARM_JOINT5').T[0][:2]
    targetPosition = finalTargetPos[:2]
    obstaclePosition = np.array([0.40,0.275,0.9][:2])
    left_points = [currentLeftPosition, np.add(obstaclePosition, np.array([-0.03, 0.02])), np.add(targetPosition, np.array([-0.03, 0.02]))]
    right_points = [currentRightPosition, np.add(obstaclePosition, np.array([-0.05, 0.18])), np.add(targetPosition, np.array([-0.05, 0.18]))]
    xs_left, y_left = sim.cubic_interpolation(left_points)
    xs_right, y_right = sim.cubic_interpolation(right_points)

    xs_left = np.flip(xs_left)
    y_left = np.flip(y_left)

    xs_right = np.flip(xs_right)    
    y_right = np.flip(y_right)

    paths_twist_1 = []
    for num in range(3):
        paths_twist_1.append(
            {
                'left': 'LARM_JOINT5',
                'leftTargetPosition': np.array(np.add(np.array([xs_left[num], y_left[num], 1.18]), np.array([0.0, 0.0, -0.01]))),
                'leftOrientation': None,
                'right': 'RARM_JOINT5',
                'rightTargetPosition': np.array(np.add(np.array([xs_right[num], y_right[num], 1.18]), np.array([0.01, 0.03, -0.01]))),
                'rightOrientation': None,
                'iterNum': 15
            }
        )

    for path in paths_twist_1:
        sim.clamp(leftEndEffector=path['left'], leftTargetPosition=path['leftTargetPosition'], leftOrientation=path['leftOrientation'], rightEndEffector=path['right'], rightTargetPosition=path['rightTargetPosition'], rightOrientation=path['rightOrientation'], iterNum=path['iterNum'])
    
    currentLeftPosition = sim.getJointPosition(jointName='LARM_JOINT5').T[0][:2]
    currentRightPosition = sim.getJointPosition(jointName='RARM_JOINT5').T[0][:2]
    targetPosition = finalTargetPos[:2]
    obstaclePosition = np.array([0.40,0.275,0.9][:2])
    left_points = [currentLeftPosition, np.add(obstaclePosition, np.array([-0.03, 0.02])), np.add(targetPosition, np.array([-0.03, 0.02]))]
    right_points = [currentRightPosition, np.add(obstaclePosition, np.array([-0.05, 0.18])), np.add(targetPosition, np.array([-0.05, 0.18]))]
    xs_left, y_left = sim.cubic_interpolation(left_points)
    xs_right, y_right = sim.cubic_interpolation(right_points)

    xs_left = np.flip(xs_left)
    y_left = np.flip(y_left)

    xs_right = np.flip(xs_right)    
    y_right = np.flip(y_right)

    paths_twist_2 = []
    for num in range(5):
        paths_twist_2.append(
            {
                'left': 'LARM_JOINT5',
                'leftTargetPosition': np.array([xs_left[num], y_left[num], 1.18]),
                'leftOrientation': None,
                'right': 'RARM_JOINT5',
                'rightTargetPosition': np.array(np.add(np.array([xs_right[num], y_right[num], 1.18]), np.array([-0.02, 0.03, 0]))),
                'rightOrientation': None,
                'iterNum': 10
            }
        )

    for path in paths_twist_2:
        sim.clamp(leftEndEffector=path['left'], leftTargetPosition=path['leftTargetPosition'], leftOrientation=path['leftOrientation'], rightEndEffector=path['right'], rightTargetPosition=path['rightTargetPosition'], rightOrientation=path['rightOrientation'], iterNum=path['iterNum'])


    paths_twist_3 = [
        {
            'left': 'LARM_JOINT5',
            'leftTargetPosition': np.array([0.34, 0.36, 1.18]),
            'leftOrientation': None,
            'right': 'RARM_JOINT5',
            'rightTargetPosition': np.array([0.44, 0.16, 1.17]),
            'rightOrientation': None,
            'iterNum': 10
        },
        {
            'left': 'LARM_JOINT5',
            'leftTargetPosition': np.array([0.335, 0.38, 1.18]),
            'leftOrientation': None,
            'right': 'RARM_JOINT5',
            'rightTargetPosition': np.array([0.438, 0.16, 1.17]),
            'rightOrientation': None,
            'iterNum': 10
        },
        {
            'left': 'LARM_JOINT5',
            'leftTargetPosition': np.array([0.33, 0.38, 1.18]),
            'leftOrientation': None,
            'right': 'RARM_JOINT5',
            'rightTargetPosition': np.array([0.435, 0.18, 1.17]),
            'rightOrientation': None,
            'iterNum': 10
        },
        {
            'left': 'LARM_JOINT5',
            'leftTargetPosition': np.array([0.325, 0.385, 1.18]),
            'leftOrientation': None,
            'right': 'RARM_JOINT5',
            'rightTargetPosition': np.array([0.433, 0.18, 1.17]),
            'rightOrientation': None,
            'iterNum': 10
        },
        {
            'left': 'LARM_JOINT5',
            'leftTargetPosition': np.array([0.31, 0.37, 1.18]),
            'leftOrientation': None,
            'right': 'RARM_JOINT5',
            'rightTargetPosition': np.array([0.43, 0.20, 1.17]),
            'rightOrientation': None,
            'iterNum': 10
        },
        {
            'left': 'LARM_JOINT5',
            'leftTargetPosition': np.array([0.30, 0.38, 1.18]),
            'leftOrientation': None,
            'right': 'RARM_JOINT5',
            'rightTargetPosition': np.array([0.42, 0.22, 1.17]),
            'rightOrientation': None,
            'iterNum': 10
        },
        {
            'left': 'LARM_JOINT5',
            'leftTargetPosition': np.array([0.30, 0.37, 1.18]),
            'leftOrientation': None,
            'right': 'RARM_JOINT5',
            'rightTargetPosition': np.array([0.417, 0.25, 1.17]),
            'rightOrientation': None,
            'iterNum': 10
        },
        {
            'left': 'LARM_JOINT5',
            'leftTargetPosition': np.array([0.28, 0.41, 1.18]),
            'leftOrientation': None,
            'right': 'RARM_JOINT5',
            'rightTargetPosition': np.array([0.43, 0.30, 1.17]),
            'rightOrientation': None,
            'iterNum': 1
        },
        {
            'left': 'LARM_JOINT5',
            'leftTargetPosition': np.array([0.28, 0.41, 1.18]),
            'leftOrientation': None,
            'right': 'RARM_JOINT5',
            'rightTargetPosition': np.array([0.43, 0.30, 1.17]),
            'rightOrientation': None,
            'iterNum': 1
        },
        {
            'left': 'LARM_JOINT5',
            'leftTargetPosition': np.array([0.28, 0.41, 1.18]),
            'leftOrientation': None,
            'right': 'RARM_JOINT5',
            'rightTargetPosition': np.array([0.43, 0.30, 1.17]),
            'rightOrientation': None,
            'iterNum': 1
        },
        {
            'left': 'LARM_JOINT5',
            'leftTargetPosition': np.array([0.28, 0.41, 1.18]),
            'leftOrientation': None,
            'right': 'RARM_JOINT5',
            'rightTargetPosition': np.array([0.43, 0.30, 1.17]),
            'rightOrientation': None,
            'iterNum': 1
        },
        {
            'left': 'LARM_JOINT5',
            'leftTargetPosition': np.array([0.28, 0.41, 1.18]),
            'leftOrientation': None,
            'right': 'RARM_JOINT5',
            'rightTargetPosition': np.array([0.43, 0.31, 1.17]),
            'rightOrientation': None,
            'iterNum': 1
        },
        {
            'left': 'LARM_JOINT5',
            'leftTargetPosition': np.array([0.28, 0.41, 1.18]),
            'leftOrientation': None,
            'right': 'RARM_JOINT5',
            'rightTargetPosition': np.array([0.43, 0.31, 1.17]),
            'rightOrientation': None,
            'iterNum': 1
        },
        {
            'left': 'LARM_JOINT5',
            'leftTargetPosition': np.array([0.278, 0.42, 1.18]),
            'leftOrientation': None,
            'right': 'RARM_JOINT5',
            'rightTargetPosition': np.array([0.426, 0.34, 1.17]),
            'rightOrientation': None,
            'iterNum': 1
        },
    ]


    for path in paths_twist_3:
        print('path_twist3')
        sim.clamp(leftEndEffector=path['left'], leftTargetPosition=path['leftTargetPosition'], leftOrientation=path['leftOrientation'], rightEndEffector=path['right'], rightTargetPosition=path['rightTargetPosition'], rightOrientation=path['rightOrientation'], iterNum=path['iterNum'])


    # # put it down
    # path_down = [
    #     {
    #         'left': 'LARM_JOINT5',
    #         'leftTargetPosition': np.array([0.30, 0.37, 1.15]),
    #         'leftOrientation': None,
    #         'right': 'RARM_JOINT5',
    #         'rightTargetPosition': np.array([0.417, 0.25, 1.14]),
    #         'rightOrientation': None,
    #         'iterNum': 10
    #     },
    # ]
    # for path in path_down:
    #     sim.clamp(leftEndEffector=path['left'], leftTargetPosition=path['leftTargetPosition'], leftOrientation=path['leftOrientation'], rightEndEffector=path['right'], rightTargetPosition=path['rightTargetPosition'], rightOrientation=path['rightOrientation'], iterNum=path['iterNum'])



    # for interpolation in interpolation_points:
    #     interpolationLeft = np.asarray(interpolation)
    
    # fig, ax = plt.subplots(figsize=(6.5, 4))
    # ax.plot(xs, interpolation_points)
    # plt.show()
    # xs_right, y_right = sim.
    cubic_position = sim.p.getBasePositionAndOrientation(cubeId)[0]
    target_position = sim.p.getBasePositionAndOrientation(targetId)[0]
    print('distance', np.linalg.norm(np.asarray(cubic_position) - np.asarray(target_position)))

    
tableId, cubeId, targetId = getReadyForTask()
solution()


try:
    time.sleep(float(sys.argv[1]))
except:
    time.sleep(10)

