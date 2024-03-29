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
taskId = 1

try:
    if sys.argv[1] == 'nogui':
        gui = False
    else:
        gui = True
except:
    gui = True


### You may want to change the code from here
pybulletConfigs = {
    "simulation": bullet_simulation,
    "pybullet_extra_data": pybullet_data,
    "gui": gui,
    "panels": False,
    "realTime": False,
    "controlFrequency": 1000,
    "updateFrequency": 250,
    "gravity": -9.81,
    "gravityCompensation": .8,
    "floor": True,
    "cameraSettings": (1.07, 90.0, -52.8, (0.07, 0.01, 0.76))
}
robotConfigs = {
    "robotPath": core_path + "/nextagea_description/urdf/NextageaOpen.urdf",
    "robotPIDConfigs": core_path + "/PD_gains.yaml",
    "robotStartPos": [0, 0, 0.85],
    "robotStartOrientation": [0, 0, 0, 1],
    "fixedBase": True,
    "colored": False
}
sim = Simulation(pybulletConfigs, robotConfigs)


# test move_with_PD ==========================
endEffector = "RARM_JOINT5"
targetPosition = np.array([0.55, -0.070, 0.93])  # x,y,z coordinates in world frame

pltDistance, _, pltTime = sim.move_with_PD_dual_joint(leftEndEffector=endEffector, leftTargetPosition=targetPosition, speed=0.01, threshold=1e-3, maxIter=100, debug=False, verbose=False)

# Now plot some graphs
task2_figure_name = "move_with_PD.png"
task2_savefig = True
# ...

fig = plt.figure(figsize=(6, 4))

print('pltTime', pltTime)
print('plotEffPosition', pltDistance)
plt.plot(pltTime, pltDistance, color='blue')
plt.xlabel("Time s")
plt.ylabel("Distance to target position")

plt.suptitle("task2 move with PD", size=16)
plt.tight_layout()
plt.subplots_adjust(left=0.15)

if task2_savefig:
    fig.savefig(task2_figure_name)
plt.show()

try:
    time.sleep(float(sys.argv[1]))
except:
    time.sleep(10)

