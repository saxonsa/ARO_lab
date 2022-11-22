from configparser import Interpolation
from scipy.spatial.transform import Rotation as npRotation
from scipy.special import comb
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import numpy as np
import math
import re
import time
import yaml
import pybullet as bullet_simulation

from Pybullet_Simulation_base import Simulation_base

# TODO: Rename class name after copying this file
class Simulation(Simulation_base):
    """A Bullet simulation involving Nextage robot"""

    def __init__(self, pybulletConfigs, robotConfigs, refVect=None):
        """Constructor
        Creates a simulation instance with Nextage robot.
        For the keyword arguments, please see in the Pybullet_Simulation_base.py
        """
        super().__init__(pybulletConfigs, robotConfigs)
        if refVect:
            self.refVector = np.array(refVect)
        else:
            self.refVector = np.array([1,0,0])

    ########## Task 1: Kinematics ##########
    # Task 1.1 Forward Kinematics
    jointRotationAxis = {
        'base_to_dummy': np.zeros(3),  # Virtual joint
        'base_to_waist': np.zeros(3),  # Fixed joint
        # TODO: modify from here
        'CHEST_JOINT0': np.array([0, 0, 1]),
        'HEAD_JOINT0': np.array([0, 0, 1]),
        'HEAD_JOINT1': np.array([0, 1, 0]),
        'LARM_JOINT0': np.array([0, 0, 1]),
        'LARM_JOINT1': np.array([0, 1, 0]),
        'LARM_JOINT2': np.array([0, 1, 0]),
        'LARM_JOINT3': np.array([1, 0, 0]),
        'LARM_JOINT4': np.array([0, 1, 0]),
        'LARM_JOINT5': np.array([0, 0, 1]),
        'RARM_JOINT0': np.array([0, 0, 1]),
        'RARM_JOINT1': np.array([0, 1, 0]),
        'RARM_JOINT2': np.array([0, 1, 0]),
        'RARM_JOINT3': np.array([1, 0, 0]),
        'RARM_JOINT4': np.array([0, 1, 0]),
        'RARM_JOINT5': np.array([0, 0, 1]),
        'RHAND'      : np.array([0, 0, 0]),
        'LHAND'      : np.array([0, 0, 0])
    }

    frameTranslationFromParent = {
        'base_to_dummy': np.zeros(3),  # Virtual joint
        # 'base_to_waist': np.zeros(3),  # Fixed joint
        # TODO: modify from here
        'base_to_waist': np.array([0, 0, 0.85]),
        'CHEST_JOINT0': np.array([0, 0, 0.267]),
        'HEAD_JOINT0': np.array([0, 0, 0.302]),
        'HEAD_JOINT1': np.array([0, 0, 0.066]),
        'LARM_JOINT0': np.array([0.04, 0.135, 0.1015]),
        'LARM_JOINT1': np.array([0, 0, 0.066]),
        'LARM_JOINT2': np.array([0, 0.095, -0.25]),
        'LARM_JOINT3': np.array([0.1805, 0, -0.03]),
        'LARM_JOINT4': np.array([0.1495, 0, 0]),
        'LARM_JOINT5': np.array([0, 0, -0.1335]),
        'RARM_JOINT0': np.array([0.04, -0.135, 0.1015]),
        'RARM_JOINT1': np.array([0, 0, 0.066]),
        'RARM_JOINT2': np.array([0, -0.095, -0.25]),
        'RARM_JOINT3': np.array([0.1805, 0, -0.03]),
        'RARM_JOINT4': np.array([0.1495, 0, 0]),
        'RARM_JOINT5': np.array([0, 0, -0.1335]),
        'RHAND'      : np.array([0, 0, 0]), # optional
        'LHAND'      : np.array([0, 0, 0]) # optional
    }

    chain_dict = {
        'CHEST_JOINT0': ['CHEST_JOINT0'],
        'HEAD_JOINT0': ['CHEST_JOINT0', 'HEAD_JOINT0'],
        'HEAD_JOINT1': ['CHEST_JOINT0', 'HEAD_JOINT0', 'HEAD_JOINT1'],
        'LARM_JOINT0': ['CHEST_JOINT0', 'LARM_JOINT0'],
        'LARM_JOINT1': ['CHEST_JOINT0', 'LARM_JOINT0', 'LARM_JOINT1'],
        'LARM_JOINT2': ['CHEST_JOINT0', 'LARM_JOINT0', 'LARM_JOINT1', 'LARM_JOINT2'],
        'LARM_JOINT3': ['CHEST_JOINT0', 'LARM_JOINT0', 'LARM_JOINT1', 'LARM_JOINT2', 'LARM_JOINT3'],
        'LARM_JOINT4': ['CHEST_JOINT0', 'LARM_JOINT0', 'LARM_JOINT1', 'LARM_JOINT2', 'LARM_JOINT3', 'LARM_JOINT4'],
        'LARM_JOINT5': ['CHEST_JOINT0', 'LARM_JOINT0', 'LARM_JOINT1', 'LARM_JOINT2', 'LARM_JOINT3', 'LARM_JOINT4',
                        'LARM_JOINT5'],
        'RARM_JOINT0': ['CHEST_JOINT0', 'RARM_JOINT0'],
        'RARM_JOINT1': ['CHEST_JOINT0', 'RARM_JOINT0', 'RARM_JOINT1'],
        'RARM_JOINT2': ['CHEST_JOINT0', 'RARM_JOINT0', 'RARM_JOINT1', 'RARM_JOINT2'],
        'RARM_JOINT3': ['CHEST_JOINT0', 'RARM_JOINT0', 'RARM_JOINT1', 'RARM_JOINT2', 'RARM_JOINT3'],
        'RARM_JOINT4': ['CHEST_JOINT0', 'RARM_JOINT0', 'RARM_JOINT1', 'RARM_JOINT2', 'RARM_JOINT3', 'RARM_JOINT4'],
        'RARM_JOINT5': ['CHEST_JOINT0', 'RARM_JOINT0', 'RARM_JOINT1', 'RARM_JOINT2', 'RARM_JOINT3', 'RARM_JOINT4',
                        'RARM_JOINT5']
    }

    plot_time = []
    plot_distance = []
    plot_time_dp = []
    plot_distance_dp = []
    ignore_joints = ['base_to_dummy', 'base_to_waist', 'RHAND', 'LHAND']

    def getJointRotationalMatrix(self, jointName=None, theta=None):
        """
            Returns the 3x3 rotation matrix for a joint from the axis-angle representation,
            where the axis is given by the revolution axis of the joint and the angle is theta.
        """
        if jointName is None:
            raise Exception("[getJointRotationalMatrix] \
                Must provide a joint in order to compute the rotational matrix!")

        # TODO modify from here

        theta_x, theta_y, theta_z = self.jointRotationAxis[jointName] * theta

        R_x = np.matrix([[1, 0, 0],
                        [0, np.cos(theta_x), -np.sin(theta_x)],
                        [0, np.sin(theta_x), np.cos(theta_x)]])

        R_y = np.matrix([[np.cos(theta_y), 0, np.sin(theta_y)],
                        [0, 1, 0],
                        [-np.sin(theta_y), 0, np.cos(theta_y)]])

        R_z = np.matrix([[np.cos(theta_z), -np.sin(theta_z), 0],
                        [np.sin(theta_z), np.cos(theta_z), 0],
                        [0, 0, 1]])
        
        return R_z * R_y * R_x

        # Hint: the output should be a 3x3 rotational matrix as a numpy array
        #return np.matrix()

    def getTransformationMatrices(self): 
        """
            Returns the homogeneous transformation matrices for each joint as a dictionary of matrices.
        """
        transformationMatrices = {}

        # TODO modify from here
        
        for joint_name, joint_pos in self.frameTranslationFromParent.items():
            if joint_name == 'RHAND' or joint_name == 'LHAND':
                continue
            joint_angle = self.getJointPos(jointName=joint_name)  # float -- the angle of the joint in radians

            transformationMatrices[joint_name] = self.constructTransformationMatrix(joint_name, joint_angle, joint_pos)
            # if theta[joint_name] is None:
            #     transformationMatrices[joint_name] = self.constructTransformationMatrix(joint_name, 0, joint_pos)
            # transformationMatrices[joint_name] = self.constructTransformationMatrix(joint_name, theta[joint_name], joint_pos)

        # Hint: the output should be a dictionary with joint names as keys and
        # their corresponding homogeneous transformation matrices as values.
        return transformationMatrices
    
    def constructTransformationMatrix(self, jointName, theta, jointPos):
        """
            Manually Added function which concat rotation matrix 
            and translation of a joint to form a transformation matrix.
kukaId
            input: 
                @jointName and @theta - used to get the rotational matix from getJointRotationalMatrix
                @jointPos - another critical element to form transformation matrix
        """
        # 1-d -> 2-d
        jointPos_2dim = np.array([jointPos])
        
        rotationMatrix = self.getJointRotationalMatrix(jointName, theta)
        AugmentedRotationMatrix = np.concatenate((rotationMatrix, jointPos_2dim.T), axis=1)
        tm = np.concatenate((AugmentedRotationMatrix, np.array([[0, 0, 0, 1]])), axis=0)
        return tm

    def deconstructTransformationMatrix(self, transformationMatrix):
        """
            Manually added funtion which extract joint position and rotation matrix from transformation matrix

        """
        pos = np.array([transformationMatrix[0,3], transformationMatrix[1,3], transformationMatrix[2,3]])
        rotmat = np.squeeze(np.asarray(transformationMatrix[0:3][:,0:3]))

        return np.array([pos]).T, rotmat

    def getJointLocationAndOrientation(self, jointName):
        """
            Returns the position and rotation matrix of a given joint using Forward Kinematics
            according to the topology of the Nextage robot.
        """
        # Remember to multiply the transformation matrices following the kinematic chain for each arm.
        #TODO modify from here

        transformationMatrices = self.getTransformationMatrices()

        if jointName == "base_to_dummy" or jointName == "base_to_waist":
            return self.frameTranslationFromParent[jointName], self.getJointRotationalMatrix(jointName)

        if jointName == "LHAND" or jointName == "RHAND":
            raise Exception("[getJointLocationAndOrientation] Sorry, currently we do not support this joint!")

        joint_chain = self.chain_dict[jointName]
        tm_joint = transformationMatrices['base_to_waist']

        for sub_joint in joint_chain:
            tm_joint = tm_joint * transformationMatrices[sub_joint]

        return self.deconstructTransformationMatrix(tm_joint)

        # Hint: return two numpy arrays, a 3x1 array for the position vector,
        # and a 3x3 array for the rotation matrix
        #return pos, rotmat
        #pass

    def getJointPosition(self, jointName):
        """Get the position of a joint in the world frame, leave this unchanged please."""
        return self.getJointLocationAndOrientation(jointName)[0]

    def getJointOrientation(self, jointName, ref=None):
        """Get the orientation of a joint in the world frame, leave this unchanged please."""
        if ref is None:
            return np.array(self.getJointLocationAndOrientation(jointName)[1] @ self.refVector).squeeze()
        else:
            return np.array(self.getJointLocationAndOrientation(jointName)[1] @ ref).squeeze()

    def getJointAxis(self, jointName):
        """Get the orientation of a joint in the world frame, leave this unchanged please."""
        return np.array(self.getJointLocationAndOrientation(jointName)[1] @ self.jointRotationAxis[jointName]).squeeze()

    def jacobianMatrix(self, endEffector):
        """Calculate the Jacobian Matrix for the Nextage Robot."""
        # TODO modify from here
        # You can implement the cross product yourself or use calculateJacobian().
        # Hint: you should return a numpy array for your Jacobian matrix. The
        # size of the matrix will depend on your chosen convention. You can have
        # a 3xn or a 6xn Jacobian matrix, where 'n' is the number of joints in
        # your kinematic chain.

        J = []  # 3 * n
        joint_chain = self.chain_dict[endEffector]
        pos_eff = self.getJointPosition(endEffector).T[0]

        for jointName in joint_chain:
            if jointName == endEffector:
                rotationAxis = self.jointRotationAxis[jointName]
                J.append(np.cross(rotationAxis, pos_eff))
                # J.append(np.cross(rotationAxis, [0, 0, 0]))
                continue
            rotationAxis = self.jointRotationAxis[jointName]  # dim: 1 * 3
            jointPos = self.getJointPosition(jointName).T[0]  # pos, dim: 1 * 3, ([a, b, c])
            J.append(np.cross(rotationAxis, (pos_eff - jointPos)))

        J = np.array(J).T

        return J

    def jacobianMatrix6(self, endEffector):
        
        J = []  # 3 * n
        joint_chain = self.chain_dict[endEffector]
        pos_eff = self.getJointPosition(endEffector).T[0]
        
        for jointName, _ in self.frameTranslationFromParent.items():
            if jointName in self.ignore_joints:
                continue
            elif jointName not in joint_chain:
                J.append(np.array([0, 0, 0]))
            else:
                rotationAxis = self.jointRotationAxis[jointName]  # dim: 1 * 3
                jointPos = self.getJointPosition(jointName).T[0]  # pos, dim: 1 * 3, ([a, b, c])
                J.append(np.cross(rotationAxis, (pos_eff - jointPos)))

        # for jointName in joint_chain:
        #     if jointName == endEffector:
        #         rotationAxis = self.jointRotationAxis[jointName]
        #         J.append(np.cross(rotationAxis, pos_eff))
        #         # J.append(np.cross(rotationAxis, [0, 0, 0]))
        #         continue
        #     rotationAxis = self.jointRotationAxis[jointName]  # dim: 1 * 3
        #     jointPos = self.getJointPosition(jointName).T[0]  # pos, dim: 1 * 3, ([a, b, c])
        #     J.append(np.cross(rotationAxis, (pos_eff - jointPos)))

        J = np.array(J).T

        J_O = []

        for jointName, _ in self.frameTranslationFromParent.items():
            if jointName in self.ignore_joints:
                continue
            elif jointName not in joint_chain:
                J_O.append(np.array([0, 0, 0]))
            else:
                rotationAxis = self.jointRotationAxis[jointName]
                a_effector = self.jointRotationAxis[endEffector]
                jointOrientation = np.cross(rotationAxis, a_effector)
                J_O.append(jointOrientation)

        # for jointName in joint_chain:
        #     # endeffect axies 
        #     rotationAxis = self.jointRotationAxis[jointName]
        #     a_effector = self.jointRotationAxis[endEffector]
        #     jointOrientation = np.cross(rotationAxis, a_effector)
        #     J_O.append(jointOrientation)
        
        J_O = np.array(J_O).T

        J_6_dim = np.concatenate((J, J_O), axis=0)

        return J_6_dim
    
    # Task 1.2 Inverse Kinematics

    def inverseKinematics(self, endEffector, targetPosition, orientation, interpolationSteps, maxIterPerStep, threshold):
        """Your IK solver \\
        Arguments: \\
            endEffector: the jointName the end-effector \\
            targetPosition: final destination the the end-effector \\
            orientation: the desired orientation of the end-effector
                         together with its parent link \\
            interpolationSteps: number of interpolation steps
            maxIterPerStep: maximum iterations per step
            threshold: accuracy threshold
        Return: \\
            Vector of x_refs
        """
        # TODO add your code here
        current_q = []
        eff_config, eff_orientation, eff_pos = None, None, None

        for joint_name, _ in self.frameTranslationFromParent.items():
            if joint_name in self.ignore_joints:
                continue
            else:
                joint_angle = self.getJointPos(jointName=joint_name) 
                current_q.append(joint_angle)

        # for joint_name in self.chain_dict[endEffector]:
        #     if joint_name == 'RHAND' or joint_name == 'LHAND':
        #         continue
        #     joint_angle = self.getJointPos(jointName=joint_name) 
        #     current_q.append(joint_angle)

        eff_pos = self.getJointPosition(jointName=endEffector).T[0]  # dim: 3 * 1
        eff_config = eff_pos
        self.plot_distance.append(np.linalg.norm(eff_pos - targetPosition))
        self.plot_time.append(time.process_time())

        eff_orientation = self.getJointOrientation(jointName=endEffector)
        eff_config = np.concatenate((eff_pos, eff_orientation), axis=0)

        traj = [current_q]
        step_positions=np.linspace(start=eff_pos, stop=targetPosition, num=interpolationSteps)

        for step in range(1, interpolationSteps):
            dy, J = None, None

            if orientation is not None:
                current_target = np.concatenate((step_positions[step], orientation), axis=0)
            else:
                current_target = np.concatenate((step_positions[step], eff_orientation), axis=0)
            dy = current_target - eff_config
            J = self.jacobianMatrix6(endEffector)

            d_theta = np.linalg.pinv(J).dot(dy)

            current_q = current_q + d_theta
            traj.append(current_q)

            # i = 0
            # for joint_name in self.chain_dict[endEffector]:
            #     self.jointTargetPos[joint_name] = current_q[i]
            #     i += 1
            i = 0
            for joint_name, _ in self.frameTranslationFromParent.items():
                if joint_name in self.ignore_joints:
                    continue
                else:
                    self.jointTargetPos[joint_name] = current_q[i]
                    i += 1
            

            self.tick_without_PD(endEffector)
            print(np.linalg.norm(eff_pos - targetPosition))
            
            eff_pos = self.getJointPosition(jointName=endEffector).T[0]
            eff_orientation = self.getJointOrientation(jointName=endEffector)
            eff_config = np.concatenate((eff_pos, eff_orientation), axis=0)

            self.plot_distance.append(np.linalg.norm(eff_pos - targetPosition))
            self.plot_time.append(time.process_time())
            if np.linalg.norm(eff_pos - targetPosition) < threshold:
                break

        return traj
        # Hint: return a numpy array which includes the reference angular
        # positions for all joints after performing inverse kinematics.
        # pass

    def move_without_PD(self, endEffector, targetPosition, speed=0.01, orientation=None,
        threshold=1e-3, maxIter=3000, debug=False, verbose=False):
        """
        Move joints using Inverse Kinematics solver (without using PD control).
        This method should update joint states directly.
        Return:
            pltTime, pltDistance arrays used for plotting
        """
        #TODO add your code here
        # iterate through joints and update joint states based on IK solver

        _ = self.inverseKinematics(endEffector, targetPosition, orientation, interpolationSteps=50, maxIterPerStep=maxIter, threshold=threshold)

        return np.array(self.plot_time), np.array(self.plot_distance)
        #return pltTime, pltDistance
        #pass

    def tick_without_PD(self, endEffector):
        """Ticks one step of simulation without PD control. """
        # TODO modify from here
        # Iterate through all joints and update joint states.
            # For each joint, you can use the shared variable self.jointTargetPos.
        # i = 0
        # for joint_name in self.chain_dict[endEffector]:
        #     self.p.resetJointState(self.robot, self.jointIds[joint_name], self.jointTargetPos[joint_name])
        #     i += 1
        
        for joint_name, _ in self.frameTranslationFromParent.items():
            if joint_name not in self.ignore_joints:
                self.p.resetJointState(self.robot, self.jointIds[joint_name], self.jointTargetPos[joint_name])

        self.p.stepSimulation()
        self.drawDebugLines()
        time.sleep(self.dt)


    ########## Task 2: Dynamics ##########
    # Task 2.1 PD Controller
    def calculateTorque(self, x_ref, x_real, dx_ref, dx_real, integral, kp, ki, kd):
        """ This method implements the closed-loop control \\
        Arguments: \\
            x_ref - the target position \\
            x_real - current position \\
            dx_ref - target velocity \\
            dx_real - current velocity \\
            integral - integral term (set to 0 for PD control) \\
            kp - proportional gain \\
            kd - derivetive gain \\
            ki - integral gain \\
        Returns: \\
            u(t) - the manipulation signal
        """
        # TODO: Add your code here
        u = kp * (x_ref - x_real) + kd * (dx_ref - dx_real)

        return u

    # Task 2.2 Joint Manipulation
    def moveJoint(self, joint, targetPosition, targetVelocity, verbose=False):
        """ This method moves a joint with your PD controller. \\
        Arguments: \\
            joint - the name of the joint \\
            targetPos - target joint position \\
            targetVel - target joint velocity
        """

        pltTorque = []
        x_list = []
        x_target = []
        x_velocity = []

        def toy_tick(x_ref, x_real, dx_ref, dx_real, integral):
            # loads your PID gains
            jointController = self.jointControllers[joint]
            kp = self.ctrlConfig[jointController]['pid']['p']
            ki = self.ctrlConfig[jointController]['pid']['i']
            kd = self.ctrlConfig[jointController]['pid']['d']

            ### Start your code here: ###
            # Calculate the torque with the above method you've made
            torque = self.calculateTorque(x_ref, x_real, dx_ref, dx_real, integral, kp, ki, kd)
            ### To here ###

            pltTorque.append(torque)

            # send the manipulation signal to the joint
            self.p.setJointMotorControl2(
                bodyIndex=self.robot,
                jointIndex=self.jointIds[joint],
                controlMode=self.p.TORQUE_CONTROL,
                force=torque
            )
            # calculate the physics and update the world
            self.p.stepSimulation()
            time.sleep(self.dt)
        
        # manually added here ----
        for i in range(1000):
            # x_last =  x_list[i-1] if i != 0 else x_list[i]
            if i == 0:
                x_last = self.getJointPos(joint)
            else:
                x_last = x_list[i-1]
            
            x_real = self.getJointPos(joint)
            x_list.append(x_real)
            dx_real = (x_real - x_last) / self.dt # current pos - last pos / time
            x_target.append(targetPosition)
            x_velocity.append(dx_real)

            # -------
            targetPosition, targetVelocity = float(targetPosition), float(targetVelocity)

            # disable joint velocity controller before apply a torque
            self.disableVelocityController(joint)

            # manually added here ----
            toy_tick(targetPosition, x_real, targetVelocity, dx_real, integral=0)
            # -------

        pltTime = np.arange(1000) * self.dt


        # logging for the graph
        pltTime, pltTarget, pltTorque, pltTorqueTime, pltPosition, pltVelocity = \
        pltTime, x_target, pltTorque, pltTime, x_list, x_velocity

        return pltTime, pltTarget, pltTorque, pltTorqueTime, pltPosition, pltVelocity

    def move_with_PD(self, endEffector, targetPosition=None, path=None, speed=0.01, orientation=None,
        threshold=1e-3, maxIter=100, debug=False, verbose=False):
        """
        Move joints using inverse kinematics solver and using PD control.
        This method should update joint states using the torque output from the PD controller.
        Return:
            pltTime, pltDistance arrays used for plotting
        """
        #TODO add your code here
        # Iterate through joints and use states from IK solver as reference states in PD controller.
        # Perform iterations to track reference states using PD controller until reaching
        # max iterations or position threshold.

        # Hint: here you can add extra steps if you want to allow your PD
        # controller to converge to the final target position after performing
        # all IK iterations (optional).

        #initial parameters
        iterNum = 100
        eff_pos = self.getJointPosition(jointName=endEffector).T[0]  # dim: 3 * 1
        self.plot_distance_dp.append(np.linalg.norm(eff_pos - targetPosition))
        self.plot_time_dp.append(time.process_time())

        # inverse kinematics
        current_q = []

        for joint_name in self.chain_dict[endEffector]:
            if joint_name == 'RHAND' or joint_name == 'LHAND':
                continue
            joint_angle = self.getJointPos(jointName=joint_name) 
            current_q.append(joint_angle)

        step_positions = np.linspace(start=eff_pos, stop=targetPosition, num=iterNum)

        if path is not None:
            step_positions = path
            iterNum = len(path)
        
        old_x_real = list()

        for step in range(1, iterNum):
            current_target = step_positions[step]

            dy = current_target - eff_pos
            J = self.jacobianMatrix(endEffector)
            d_theta = np.linalg.pinv(J).dot(dy)

            # last_q = current_q
            current_q += d_theta

            # moving by DP
            new_x_real = list()
            # for _ in range(200):
            #     if len(new_x_real) == 0:
            #         old_x_real = current_q
            #     else:
            #         old_x_real = new_x_real
            if len(new_x_real) == 0:
                old_x_real = current_q
            else:
                old_x_real = new_x_real
            new_x_real = self.tick(endEffector, current_q, old_x_real)

            # check the position of effector after moving with PD
            eff_pos = self.getJointPosition(jointName=endEffector).T[0]

            # print(np.linalg.norm(eff_pos - current_target))
            print(np.linalg.norm(eff_pos - targetPosition))

            self.plot_distance_dp.append(np.linalg.norm(eff_pos - targetPosition))
            self.plot_time_dp.append(time.process_time())

            if np.linalg.norm(eff_pos - targetPosition) < threshold:
                print('stop')
                break

        return np.array(self.plot_time_dp), np.array(self.plot_distance_dp)
    
    def move_with_PD_dual_joint(self, leftEndEffector, leftTargetPosition, leftOrientation=None, rightEndEffector=None, rightTargetPosition=None, \
            rightOrientation=None, speed=0.01, threshold=1e-3, maxIter=100, debug=False, verbose=False, iterNum=25):

        eff_left_pos = self.getJointPosition(jointName=leftEndEffector).T[0]
        eff_left_orientation = self.getJointOrientation(jointName=leftEndEffector)
        eff_left_config = np.concatenate((eff_left_pos, eff_left_orientation), axis=0)

        if rightEndEffector is not None:

            eff_right_pos = self.getJointPosition(jointName=rightEndEffector).T[0]
            eff_right_orientation = self.getJointOrientation(jointName=rightEndEffector)
            eff_right_config = np.concatenate((eff_right_pos, eff_right_orientation), axis=0)

        # plot
        plot_left_distance, plot_right_distance, plot_time = [], [], []

        # initialize current_q
        left_current_q, right_current_q = list(), list()
        total_current_q = []

        for joint_name, _ in self.frameTranslationFromParent.items():
            if joint_name not in self.ignore_joints:
                joint_angle = self.getJointPos(jointName=joint_name)
                left_current_q.append(joint_angle)
                right_current_q.append(joint_angle)
                total_current_q.append(joint_angle)
        

        left_step_positions = np.linspace(start=eff_left_pos, stop=leftTargetPosition, num=iterNum)

        if rightEndEffector is not None:
            right_step_positions = np.linspace(start=eff_right_pos, stop=rightTargetPosition, num=iterNum)
        left_new_x_real, left_old_x_real, right_new_x_real, right_old_x_real = [], [], [], []
        total_new_x_real, total_old_x_real = [], []

        for step in range(1, iterNum):
            if leftOrientation is not None:
                left_current_target = np.concatenate((left_step_positions[step], leftOrientation), axis=0)
            else:
                left_current_target = np.concatenate((left_step_positions[step], eff_left_orientation), axis=0)

            if rightOrientation is not None:
                right_current_target = np.concatenate((right_step_positions[step], rightOrientation), axis=0)
            else:
                if rightEndEffector is not None:
                    right_current_target = np.concatenate((right_step_positions[step], eff_right_orientation), axis=0)


            dy_left = left_current_target - eff_left_config
            J_left = self.jacobianMatrix6(leftEndEffector)
            d_theta_left = np.linalg.pinv(J_left).dot(dy_left)
            left_current_q += d_theta_left
            total_current_q += d_theta_left

            if rightEndEffector is not None:
                dy_right = right_current_target - eff_right_config
                J_right = self.jacobianMatrix6(rightEndEffector)
                d_theta_right = np.linalg.pinv(J_right).dot(dy_right)
                right_current_q += d_theta_right
                total_current_q += d_theta_right

            # moving by DP
            # for _ in range(100):
            #     if len(left_new_x_real) == 0:
            #         left_old_x_real = left_current_q
            #     else:
            #         left_old_x_real = left_new_x_real
                
            #     if rightEndEffector is not None:
            #         if len(right_new_x_real) == 0:
            #             right_old_x_real = right_current_q
            #         else:
            #             right_old_x_real = right_new_x_real

            #     left_new_x_real = self.tick(left_current_q, left_old_x_real)

            #     if rightEndEffector is not None:
            #         right_new_x_real = self.tick(right_current_q, right_old_x_real)

            for _ in range(60):
                if len(total_new_x_real) == 0:
                    total_old_x_real = total_current_q
                else:
                    total_old_x_real = total_new_x_real

                total_new_x_real = self.tick(total_current_q, total_old_x_real)
            
            # check the position of effector after moving with PD
            eff_left_pos = self.getJointPosition(jointName=leftEndEffector).T[0]
            eff_left_orientation = self.getJointOrientation(jointName=leftEndEffector)
            eff_left_config = np.concatenate((eff_left_pos, eff_left_orientation), axis=0)

            if rightEndEffector is not None:
                eff_right_pos = self.getJointPosition(jointName=rightEndEffector).T[0]
                eff_right_orientation = self.getJointOrientation(jointName=rightEndEffector)
                eff_right_config = np.concatenate((eff_right_pos, eff_right_orientation), axis=0)

            # print(np.linalg.norm(eff_pos - current_target))
            print('left hand', np.linalg.norm(eff_left_pos - leftTargetPosition))
            if rightEndEffector is not None:
                print('right hand', np.linalg.norm(eff_right_pos - rightTargetPosition))

            plot_left_distance.append(np.linalg.norm(eff_left_pos - leftTargetPosition))
            if rightEndEffector is not None:
                plot_right_distance.append(np.linalg.norm(eff_right_pos - rightTargetPosition))
            plot_time.append(time.process_time())

            if rightEndEffector is not None:
                if np.linalg.norm(eff_left_pos - leftTargetPosition) < threshold:
                    print('stop')
                    break
            else:
                if np.linalg.norm(eff_left_pos - leftTargetPosition) < threshold:
                    print('stop')
                    break

        return plot_left_distance, plot_right_distance, plot_time


    def move_with_PD6(self, endEffector, targetPosition=None, orientation=None, path=None, speed=0.01,
        threshold=1e-3, maxIter=100, debug=False, verbose=False):
        """
        Move joints using inverse kinematics solver and using PD control.
        This method should update joint states using the torque output from the PD controller.
        Return:
            pltTime, pltDistance arrays used for plotting
        """
        #TODO add your code here
        # Iterate through joints and use states from IK solver as reference states in PD controller.
        # Perform iterations to track reference states using PD controller until reaching
        # max iterations or position threshold.

        # Hint: here you can add extra steps if you want to allow your PD
        # controller to converge to the final target position after performing
        # all IK iterations (optional).

        #initial parameters
        iterNum = 20
        eff_pos = self.getJointPosition(jointName=endEffector).T[0]  # dim: 1 * 3
        eff_orientation = self.getJointOrientation(jointName=endEffector) # dim: 1 * 3
        eff_config = np.concatenate((eff_pos, eff_orientation), axis=0)
        self.plot_distance_dp.append(np.linalg.norm(eff_pos - targetPosition[0:3]))
        self.plot_time_dp.append(time.process_time())

        # inverse kinematics
        current_q = []

        for joint_name, _ in self.frameTranslationFromParent.items():
            if joint_name not in self.ignore_joints:
                joint_angle = self.getJointPos(jointName=joint_name)
                current_q.append(joint_angle)
        # for joint_name in self.chain_dict[endEffector]:
        #     if joint_name == 'RHAND' or joint_name == 'LHAND':
        #         continue
        #     joint_angle = self.getJointPos(jointName=joint_name) 
        #     current_q.append(joint_angle)
        
        step_positions = np.linspace(start=eff_pos, stop=targetPosition, num=iterNum)
        if path is not None:
            step_positions = path
            iterNum = len(path)
        
        new_x_real, old_x_real = [], []

        for step in range(1, iterNum):
            if orientation is not None:
                current_target = np.concatenate((step_positions[step], orientation), axis=0)
            else:
                current_target = np.concatenate((step_positions[step], eff_orientation), axis=0)


            dy = current_target - eff_config
            J = self.jacobianMatrix6(endEffector)
            d_theta = np.linalg.pinv(J).dot(dy)

            # last_q = current_q
            current_q += d_theta

            # moving by DP
            for _ in range(60):
                if len(new_x_real) == 0:
                    old_x_real = current_q
                else:
                    old_x_real = new_x_real

                new_x_real = self.tick(current_q, old_x_real)

            # check the position of effector after moving with PD
            eff_pos = self.getJointPosition(jointName=endEffector).T[0]
            eff_orientation = self.getJointOrientation(jointName=endEffector)
            eff_config = np.concatenate((eff_pos, eff_orientation), axis=0)

            # print(np.linalg.norm(eff_pos - current_target))
            print(np.linalg.norm(eff_pos - targetPosition))

            self.plot_distance_dp.append(np.linalg.norm(eff_pos - targetPosition))
            self.plot_time_dp.append(time.process_time())

            if np.linalg.norm(eff_pos - targetPosition) < threshold:
                print('stop')
                break
            
        return np.array(self.plot_time_dp), np.array(self.plot_distance_dp)

    def tick(self, theta_list, old_x_real):
        """Ticks one step of simulation using PD control."""
        # Iterate through all joints and update joint states using PD control.
        # for joint in self.joints:
        new_x_real = []

        i = 0

        for joint, _ in self.frameTranslationFromParent.items():
            if joint not in self.ignore_joints:
                
        # for joint in self.chain_dict[endEffector]:
                integral = 0
                x_ref = theta_list[i]  # target angle 
                
                x_real = self.getJointPos(joint) # joint angle 
                new_x_real.append(x_real)

                dx_ref = 0 # target velocity

                # dx_real1 = self.getJointVel(joint) # joint velocity
                # dx_real2 = (x_real - last_q_list[i]) / self.dt
                dx_real = (x_real - old_x_real[i]) / self.dt
                i += 1
                

                # skip dummy joints (world to base joint)
                jointController = self.jointControllers[joint]
                if jointController == 'SKIP_THIS_JOINT':
                    continue

                # disable joint velocity controller before apply a torque
                self.disableVelocityController(joint)

                # loads your PID gains
                kp = self.ctrlConfig[jointController]['pid']['p']
                ki = self.ctrlConfig[jointController]['pid']['i']
                kd = self.ctrlConfig[jointController]['pid']['d']

                ### Implement your code from here ... ###
                # TODO: obtain torque from PD controller
                torque = self.calculateTorque(x_ref, x_real, dx_ref, dx_real, integral, kp, ki, kd)
                ### ... to here ###

                self.p.setJointMotorControl2(
                    bodyIndex=self.robot,
                    jointIndex=self.jointIds[joint],
                    controlMode=self.p.TORQUE_CONTROL,
                    force=torque
                )

                # Gravity compensation
                # A naive gravitiy compensation is provided for you
                # If you have embeded a better compensation, feel free to modify
                compensation = self.jointGravCompensation[joint]
                self.p.applyExternalForce(
                    objectUniqueId=self.robot,
                    linkIndex=self.jointIds[joint],
                    forceObj=[0, 0, -compensation],
                    posObj=self.getLinkCoM(joint),
                    flags=self.p.WORLD_FRAME
                )
                # Gravity compensation ends here

        self.p.stepSimulation()
        self.drawDebugLines()
        time.sleep(self.dt)

        return new_x_real

    ########## Task 3: Robot Manipulation ##########
    def cubic_interpolation(self, points, nTimes=100):
        """
        Given a set of control points, return the
        cubic spline defined by the control points,
        sampled nTimes along the curve.
        """
        #TODO add your code here
        # Return 'nTimes' points per dimension in 'points' (typically a 2xN array),
        # sampled from a cubic spline defined by 'points' and a boundary condition.
        # You may use methods found in scipy.interpolate

        # points = [start, target]

        point_num = len(points)
        cs = CubicSpline(range(point_num), points, bc_type='natural')
        point_gap = (point_num - 1) / nTimes

        xs = np.arange(0, (point_num - 1) + point_gap, point_gap)
        return cs(xs)

        #return xpoints, ypoints

    # Task 3.1 Pushing
    # def dockingToPosition(self, leftTargetAngle, rightTargetAngle, angularSpeed=0.005,
    #         threshold=1e-1, maxIter=300, verbose=False):
    #     """A template function for you, you are free to use anything else"""
        
    #     # TODO: Append your code here
    #     points = self.cubic_interpolation()        
    #     pass
    
    def selfDockingToPosition(self, leftEndEffector, leftTargetPosition, leftOrientation=None, rightEndEffector=None, rightTargetPosition=None, \
            rightOrientation=None, speed=0.01, threshold=1e-3, maxIter=100, debug=False, verbose=False, iterNum=25):
        # path = self.cubic_interpolation(points)

        # plot path
        # from mpl_toolkits.mplot3d import Axes3D
        # fig = plt.figure(figsize=(10, 10))
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(path[:,0], path[:,1], path[:,2])
        # plt.show()
        self.move_with_PD_dual_joint(leftEndEffector, leftTargetPosition, leftOrientation=leftOrientation, rightEndEffector=rightEndEffector, rightTargetPosition=rightTargetPosition, rightOrientation=rightOrientation, \
            threshold=1e-3, iterNum=iterNum)


    # Task 3.2 Grasping & Docking
    def clamp(self, leftEndEffector, leftTargetPosition, leftOrientation=None, rightEndEffector=None, rightTargetPosition=None, \
            rightOrientation=None, speed=0.01, threshold=1e-3, maxIter=100, debug=False, verbose=False, iterNum=25):
        """A template function for you, you are free to use anything else"""
        self.move_with_PD_dual_joint(leftEndEffector, leftTargetPosition, leftOrientation=leftOrientation, rightEndEffector=rightEndEffector, rightTargetPosition=rightTargetPosition, rightOrientation=rightOrientation, \
            threshold=1e-3, iterNum=iterNum)

 ### END
