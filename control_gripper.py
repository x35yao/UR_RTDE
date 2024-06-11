import time

def close_gripper(rtde):
    '''
    :param rtde: rtde_io.RTDEIOInterface() object.
    :return: True
    '''
    rtde.setInputIntRegister(18, 1)  ### Gripper close
    time.sleep(1)
    rtde.setInputIntRegister(18, 2)### Set gripper to defalt state 2
    return

def open_gripper(rtde):
    '''
    :param rtde: rtde_io.RTDEIOInterface() object.
    :return: True
    '''
    rtde.setInputIntRegister(18, 0)  #### Gripper open
    time.sleep(1)
    rtde.setInputIntRegister(18, 2) ### Set gripper to defalt state 2
    return