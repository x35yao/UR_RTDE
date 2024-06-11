import time

import numpy as np
from scipy.spatial.transform import Rotation as R


VELOCITY = 0.05
ACCELERATION = 0.05


# defaltQ = [0.0003591522399801761, -1.1898253599749964, -1.743985954915182, -1.7710626761065882, 1.556787133216858,
#            -4.738660995160238]
defaltQ = [0.00034716803929768503, -1.1898253599749964, -1.743997875844137, -1.8, 1.5560801029205322, -4.7386489550219935]## defalt joint position of the robot

defaltTCP = [0.3228618875769767, -0.1118356963537865, 0.29566077338942987, 3.134481837517487, 0.04104053340885737, -0.033028778010714535] ## defalt robot pose

def move_to_defalt_pose(rtde_c, asynchronous = True):
    rtde_c.moveJ(defaltQ, speed = 0.4, acceleration = 0.4, asynchronous = asynchronous)  ## Back to defalt pose
    return

def move_to_defalt_ori(rtde_r, rtde_c, asynchronous = True):
    target = rtde_r.getActualTCPPose()
    target[3:] = defaltTCP[3:]
    rtde_c.moveL(target, speed = 0.4, acceleration = 0.4, asynchronous = asynchronous)  ## Back to defalt pose
    return

def move_foward(rtde_r, rtde_c, D, reference_frame, asynchronous = True):
    target = rtde_r.getActualTCPPose()
    if reference_frame == 'base':
        target[0] += D
    elif reference_frame == 'tool':
        rotmat = R.from_rotvec(target[3:]).as_matrix()
        t = rotmat @ np.array([D, 0, 0])
        target[0:3] += t
    rtde_c.moveL(target, VELOCITY, ACCELERATION, asynchronous = asynchronous)
    return

def move_backward(rtde_r, rtde_c, D, reference_frame, asynchronous = True):
    target = rtde_r.getActualTCPPose()
    if reference_frame == 'base':
        target[0] += -D
    elif reference_frame == 'tool':
        rotmat = R.from_rotvec(target[3:]).as_matrix()
        t = rotmat @ np.array([-D, 0, 0])
        target[0:3] += t
    rtde_c.moveL(target, VELOCITY, ACCELERATION, asynchronous = asynchronous)
    return

def move_left(rtde_r, rtde_c, D, reference_frame, asynchronous = True):
    target = rtde_r.getActualTCPPose()
    if reference_frame == 'base':
        target[1] += D
    elif reference_frame == 'tool':
        rotmat = R.from_rotvec(target[3:]).as_matrix()
        t = rotmat @ np.array([0, -D, 0])
        target[0:3] += t
    rtde_c.moveL(target, VELOCITY, ACCELERATION, asynchronous = asynchronous)
    return

def move_right(rtde_r, rtde_c, D, reference_frame, asynchronous = True):
    target = rtde_r.getActualTCPPose()
    if reference_frame == 'base':
        target[1] += -D
    elif reference_frame == 'tool':
        rotmat = R.from_rotvec(target[3:]).as_matrix()
        t = rotmat @ np.array([0, D, 0])
        target[0:3] += t
    rtde_c.moveL(target, VELOCITY, ACCELERATION, asynchronous = asynchronous)
    return

def move_up(rtde_r, rtde_c, D, reference_frame, asynchronous = True):
    target = rtde_r.getActualTCPPose()
    if reference_frame == 'base':
        target[2] += D
    elif reference_frame == 'tool':
        rotmat = R.from_rotvec(target[3:]).as_matrix()
        t = rotmat @ np.array([0, 0, -D])
        target[0:3] += t
    rtde_c.moveL(target, VELOCITY, ACCELERATION, asynchronous = asynchronous)
    return

def move_down(rtde_r, rtde_c, D, reference_frame, asynchronous = True):
    target = rtde_r.getActualTCPPose()
    if reference_frame == 'base':
        target[2] += -D
    elif reference_frame == 'tool':
        rotmat = R.from_rotvec(target[3:]).as_matrix()
        t = rotmat @ np.array([0, 0, D])
        target[0:3] += t
    rtde_c.moveL(target, VELOCITY, ACCELERATION, asynchronous = asynchronous)
    return

def rotate_wrist_clockwise(rtde_r, rtde_c, asynchronous = True):
    target = rtde_r.getActualQ()
    target[-1] += 0.05
    rtde_c.moveJ(target, asynchronous = asynchronous)
    return

def rotate_wrist_counterclockwise(rtde_r, rtde_c, asynchronous = True):
    target = rtde_r.getActualQ()
    target[-1] += -0.05
    rtde_c.moveJ(target, asynchronous = asynchronous)
    return

def screw(rtde_r, rtde_c, asynchronous = True):
    target = rtde_r.getActualQ()
    target[-1] += 3 * np.pi
    rtde_c.moveJ(target, asynchronous = asynchronous)
    return

def unscrew(rtde_r, rtde_c, asynchronous = True):
    target = rtde_r.getActualQ()
    target[-1] += - 3 * np.pi
    rtde_c.moveJ(target, asynchronous = asynchronous)
    return

def wrist2_plus(rtde_r, rtde_c, asynchronous = True):
    target = rtde_r.getActualQ()
    target[-2] += 0.3
    rtde_c.moveJ(target, asynchronous = asynchronous)
    return

def wrist2_minus(rtde_r, rtde_c, asynchronous = True):
    target = rtde_r.getActualQ()
    target[-2] += -0.3
    rtde_c.moveJ(target, asynchronous = asynchronous)
    return

def wrist1_plus(rtde_r, rtde_c, asynchronous = True):
    target = rtde_r.getActualQ()
    target[-3] += 0.3
    rtde_c.moveJ(target, asynchronous = asynchronous)
    return

def wrist1_minus(rtde_r, rtde_c, asynchronous = True):
    target = rtde_r.getActualQ()
    target[-3] += -0.3
    rtde_c.moveJ(target, asynchronous = asynchronous)
    return

def servoL(poses, rtde_c, dt, vel = 0.05, acc = 0.05, lookahead_time = 0.1, gain = 300):
    print(len(poses))
    for i, pose in enumerate(poses):
        print(i)
        t_start = rtde_c.initPeriod()
        rtde_c.servoL(pose, vel, acc, dt, lookahead_time, gain)
        rtde_c.waitPeriod(t_start)
    rtde_c.servoStop()
    return

# def servoStop(rtde):
#     '''
#     :param rtde: rtde_io.RTDEIOInterface() object.
#     :return: True
#     '''
#     rtde.setInputIntRegister(0, 16)  #### value 16 corresponds to stop servo
#     return

# def get_horizontal(rtde_c):
#     speed = 0.4
#     acc = 0.4
#     blend = 0.01
#     a = [0.000323199579725042, -1.1898968855487269, -1.7439616362201136, -1.7710745970355433, 0.14936089515686035, -4.738744918500082, speed, acc, blend]
#     b = [0.00038312069955281913, -1.1898253599749964, -1.7439501921283167, -1.771050278340475, -1.571493927632467, -4.744859878216879, speed, acc, blend]
#     c = [0.00038312069955281913, -1.1898253599749964, -1.7439501921283167,  -3.305073086415426, -1.5715177694903772, -4.744823877011434, speed, acc, blend]
#     pose = []
#     pose.append(a)
#     pose.append(b)
#     pose.append(c)
#     rtde_c.moveJ(pose, asynchronous = True)
#     return

# def get_vertical(rtde_c):
#     speed = 0.4
#     acc = 0.4
#     blend = 0
#     a = [0.00034716803929768503, -1.189801041279928, -1.7439616362201136, -2.443040196095602, -0.014967266713277638,
#          -4.677885238324301, speed, acc, blend]
#     b = [0.0003351838095113635, -1.189789120350973, -1.7438300291644495, -1.7987802664386194, 1.560789942741394, -4.7387089172946375, speed, acc, blend]
#     pose = []
#     pose.append(a)
#     pose.append(b)
#     rtde_c.moveJ(pose)
#     return

# def get_vertical(rtde_r, rtde_c, asynchronous = True):
#     # speed = 0.4
#     # acc = 0.4
#     # blend = 0
#     target = rtde_r.getActualQ()
#     # a = target[:3] + [-2.6525519529925745, -0.6156352202044886, -1.782320801411764, speed, acc, blend]
#     # b = target[:3] + [-1.7710626761065882, 1.556787133216858, -4.738660995160238, speed, acc, blend]
#     target[3:] = [-2.6525519529925745, -0.6156352202044886, -1.782320801411764]
#     rtde_c.moveJ(target, speed = 0.4, acceleration = 0.4, asynchronous = False)
#     target = rtde_r.getActualQ()
#     target[3:] = [-1.7710626761065882, 1.556787133216858, -4.738660995160238]
#     rtde_c.moveJ(target, speed=0.4, acceleration=0.4, asynchronous=False)
#     # pose = [a, b]
#     return

def get_vertical(rtde_r, rtde_c, asynchronous = True):
    target = rtde_r.getActualQ()
    target[3:] = [-1.7710626761065882, 1.556787133216858, -4.738660995160238]
    rtde_c.moveJ(target, speed = 0.4, acceleration = 0.4, asynchronous = asynchronous)
    return

def get_horizontal(rtde_r, rtde_c, asynchronous = True):
    target = rtde_r.getActualQ()
    target[3:] = [-3.4431965986834925, -1.623061482106344, -1.5770514647113245]
    rtde_c.moveJ(target, speed = 0.4, acceleration = 0.4, asynchronous = asynchronous)
    return

def stop(rtde_c):
    rtde_c.stopJ(10, True)
    rtde_c.servoStop()
    return

def protective_stop(rtde_c):
    rtde_c.triggerProtectiveStop()
    return





