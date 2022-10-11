from scipy.spatial.transform import Rotation as npRotation
from scipy.special import comb
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import numpy as np
import math
import re
import time
import yaml

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
        'base_to_waist': np.zeros(3),  # Fixed joint
        # TODO: modify from here
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

    def getJointRotationalMatrix(self, jointName=None, theta=None):
        """
            Returns the 3x3 rotation matrix for a joint from the axis-angle representation,
            where the axis is given by the revolution axis of the joint and the angle is theta.
        """
        if jointName == None:
            raise Exception("[getJointRotationalMatrix] \
                Must provide a joint in order to compute the rotational matrix!")

        # TODO modify from here

        # manually add: initialize theta to 0 if it is not given

        if theta == None:
            theta = 0
        

        #if jointRotationAxis[jointName]
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
        #pass

    def getTransformationMatrices(self): 
        """
            Returns the homogeneous transformation matrices for each joint as a dictionary of matrices.
        """
        transformationMatrices = {}

        # TODO modify from here
        
        for joint_name, joint_pos in self.frameTranslationFromParent.items():
            if joint_name == 'RHAND' or joint_name == 'LHAND':
                continue
            joint_angle = self.getJointPos(jointName=joint_name)
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

            input: 
                @jointName and @theta - used to get the rotational matix from getJointRotationalMatrix
                @jointPos - another critical element to form transformation matrix
        """
        # 1-d -> 2-d
        jointPos_2dim = np.array([jointPos])
        
        rotationMatrix = self.getJointRotationalMatrix(jointName, theta)
        AugmentedRotationMatrix = np.concatenate((rotationMatrix, jointPos_2dim.T), axis=1)
        return np.concatenate((AugmentedRotationMatrix, np.array([[0, 0, 0, 1]])), axis=0)

    def deconstructTransformationMatrix(self, transformationMatrix):
        """
            Manually added funtion which extract joint position and rotation matrix from transformation matrix
            
        """
        pos = np.array([transformationMatrix[0,3], transformationMatrix[1,3], transformationMatrix[2,3]])
        rotmat = np.matrix(transformationMatrix[0:3][:,0:3])

        return np.array([pos]).T, rotmat

    def getJointLocationAndOrientation(self, jointName):
        """
            Returns the position and rotation matrix of a given joint using Forward Kinematics
            according to the topology of the Nextage robot.
        """
        # Remember to multiply the transformation matrices following the kinematic chain for each arm.
        #TODO modify from here

        # fix 0, 1
        # body support: 2 - 3 - 4
        # left arm: 5 - 6 - 7 - 8 - 9 - 10
        # right arm: 11 - 12 - 13 - 14 - 15 - 16

        transformationMatrices = self.getTransformationMatrices()

        ARM_name = ["LARM_JOINT", "RARM_JOINT"]

        if jointName == "base_to_dummy" or jointName == "base_to_waist":
            return self.frameTranslationFromParent[jointName], self.getJointRotationalMatrix(jointName)

        elif jointName == "LHAND" or jointName == "RHAND":
            raise Exception("[getJointLocationAndOrientation] Sorry, currently we do not support this joint!")
        
        elif jointName == "CHEST_JOINT0":
            return self.deconstructTransformationMatrix(transformationMatrices[jointName])

        elif jointName == "HEAD_JOINT0":
            return self.deconstructTransformationMatrix(transformationMatrices["CHEST_JOINT0"] * transformationMatrices[jointName])

        elif jointName == "HEAD_JOINT1":
            tm_head_joint = transformationMatrices["CHEST_JOINT0"] * transformationMatrices["HEAD_JOINT0"] * transformationMatrices["HEAD_JOINT1"] 
            return self.deconstructTransformationMatrix(tm_head_joint)
        
        else:
            arm_index = int(jointName[-1])
            tm_side = transformationMatrices["CHEST_JOINT0"]
            if jointName[:-1] == ARM_name[0]:
                # left arm
                for i in range(0, arm_index+1, 1):
                    tm_side = tm_side * transformationMatrices["LARM_JOINT{joint_index}".format(joint_index=i)]
                return self.deconstructTransformationMatrix(tm_side)
            else:
                # right arm
                for i in range(0, arm_index+1, 1):
                    tm_side = tm_side * transformationMatrices["RARM_JOINT{joint_index}".format(joint_index=i)]
                return self.deconstructTransformationMatrix(tm_side)
            
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
            rotationAxis = self.jointRotationAxis[jointName]  # dim: 1 * 3
            # if jointName == endEffector:
            #     J.append(np.cross(rotationAxis, pos_eff))
            #     continue
            jointPos = self.getJointPosition(jointName).T[0]  # pos, dim: 1 * 3, ([a, b, c])
            J.append(np.cross(rotationAxis, (pos_eff - jointPos)))

        J = np.array(J).T

        return J

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
        # Hint: return a numpy array which includes the reference angular
        # positions for all joints after performing inverse kinematics.
        pass

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

        #return pltTime, pltDistance
        pass

    def tick_without_PD(self):
        """Ticks one step of simulation without PD control. """
        # TODO modify from here
        # Iterate through all joints and update joint states.
            # For each joint, you can use the shared variable self.jointTargetPos.

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
        pass

    # Task 2.2 Joint Manipulation
    def moveJoint(self, joint, targetPosition, targetVelocity, verbose=False):
        """ This method moves a joint with your PD controller. \\
        Arguments: \\
            joint - the name of the joint \\
            targetPos - target joint position \\
            targetVel - target joint velocity
        """
        def toy_tick(x_ref, x_real, dx_ref, dx_real, integral):
            # loads your PID gains
            jointController = self.jointControllers[joint]
            kp = self.ctrlConfig[jointController]['pid']['p']
            ki = self.ctrlConfig[jointController]['pid']['i']
            kd = self.ctrlConfig[jointController]['pid']['d']

            ### Start your code here: ###
            # Calculate the torque with the above method you've made
            torque = 0.0
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

        targetPosition, targetVelocity = float(targetPosition), float(targetVelocity)

        # disable joint velocity controller before apply a torque
        self.disableVelocityController(joint)
        # logging for the graph
        pltTime, pltTarget, pltTorque, pltTorqueTime, pltPosition, pltVelocity = [], [], [], [], [], []

        return pltTime, pltTarget, pltTorque, pltTorqueTime, pltPosition, pltVelocity

    def move_with_PD(self, endEffector, targetPosition, speed=0.01, orientation=None,
        threshold=1e-3, maxIter=3000, debug=False, verbose=False):
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

        #return pltTime, pltDistance
        pass

    def tick(self):
        """Ticks one step of simulation using PD control."""
        # Iterate through all joints and update joint states using PD control.
        for joint in self.joints:
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
            torque = 0.0  # TODO: fix me
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

        #return xpoints, ypoints
        pass

    # Task 3.1 Pushing
    def dockingToPosition(self, leftTargetAngle, rightTargetAngle, angularSpeed=0.005,
            threshold=1e-1, maxIter=300, verbose=False):
        """A template function for you, you are free to use anything else"""
        # TODO: Append your code here
        pass

    # Task 3.2 Grasping & Docking
    def clamp(self, leftTargetAngle, rightTargetAngle, angularSpeed=0.005, threshold=1e-1, maxIter=300, verbose=False):
        """A template function for you, you are free to use anything else"""
        # TODO: Append your code here
        pass

 ### END
