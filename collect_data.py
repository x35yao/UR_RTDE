import threading
import pygame
import sys
import tcp_client as tc
import datetime
import os
import numpy as np
import nidaqmx
import rtde_receive
from rtde_control import RTDEControlInterface as RTDEControl
import rtde_control
import rtde_io
import time
from control_gripper import open_gripper, close_gripper
from control_robot import move_up, move_down, move_left, move_right, move_foward, move_backward, move_to_defalt_pose, move_to_defalt_ori, rotate_wrist_clockwise, rotate_wrist_counterclockwise, screw, unscrew, get_horizontal, get_vertical, stop, protective_stop, wrist1_plus, wrist2_plus, wrist1_minus, wrist2_minus
import ctypes
import pickle
import cv2

def is_capslock_on():
    return True if ctypes.WinDLL("User32.dll").GetKeyState(0x14) else False


calibration_matrix = np.array([[-0.07071, 0.00779, 0.08539, -3.77588, -0.08876, 3.81759],
                               [-0.14419,   4.32640,   0.02626,  -2.17358,   0.14086,  -2.21257 ],
                               [6.18920,  -0.15334,   6.57294,  -0.27184,   6.62477,  -0.25805],
                               [-0.00169,   0.05222,  -0.18994,  -0.01967,   0.19266, -0.03254],
                               [0.20699,  -0.00403,  -0.11136,   0.04967,  -0.10984,  -0.04233],
                               [0.00134,  -0.11867,   0.00232,  -0.11984,   0.00672,  -0.12104]])


# def record_robot(rtde_r, filename, frequency):
#     global collecting_data
#     rtde_r.startFileRecording(filename)
#     print("Data recording started.")
#     dt = 1 / frequency
#     i = 0
#     while collecting_data:
#         # start = time.time()
#         # if i % 10 == 0:
#         #     # sys.stdout.write("\r")
#         #     # sys.stdout.write("{:3d} samples.".format(i))
#         #     sys.stdout.flush()
 #         # duration = end - start
#         # if duration < dt:
#         #     time.sleep(dt - duration)
#         # i += 1
#         print('recording')
#     rtde_r.stopFileRecording()
#     print("\nData recording stopped.")

def read_force(filename):
    '''
    This file record the gauge values of the force sensor. Force_torque = calibration_matrix @ gauge_values
    '''
    global collecting_data
    gauge_values = []

    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan("dev1/ai0:5")
        while collecting_data:
                gauge_values.append(np.array(task.read(number_of_samples_per_channel=1)).flatten())
    force_torque = (calibration_matrix @ np.array(gauge_values).T).T
    np.save(filename + '.npy', force_torque)

def connect_robot(host, FREQUENCY):
    try:
        rtde_r = rtde_receive.RTDEReceiveInterface(host, FREQUENCY)
        rtde_c = rtde_control.RTDEControlInterface(host, FREQUENCY, RTDEControl.FLAG_CUSTOM_SCRIPT)
        rtde = rtde_io.RTDEIOInterface(host)
        print("Robot connection success")
        return rtde, rtde_c, rtde_r
    except RuntimeError:
        print('Robot connection failure')
        raise
def connect_camera():
    my_camera = tc.command_camera()
    if my_camera.connected == 1:
        print("Camera connection success")
    else:
        print("Camera connection failure")
    return my_camera

def record_wrist_camera(url, filename):
    global collecting_data
    wrist_camera = cv2.VideoCapture(url)
    frame_width = int(wrist_camera.get(3))
    frame_height = int(wrist_camera.get(4))

    out = cv2.VideoWriter(filename + '.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
    while collecting_data:
        ret, frame = wrist_camera.read()
        if ret == True:
            out.write(frame)
        else:
            break
    wrist_camera.release()
    out.release()

def connect_wrist_camera(url):
    wrist_camera = cv2.VideoCapture(url)
    if not wrist_camera.isOpened():
       raise ValueError('Wrist camera cannot be connected, check wrist camera opened and url correct')
    else:
        print('Wrist camera connection success')
    return



if __name__ == "__main__":
    DATE = str(datetime.date.today())
    FOLDER = os.path.join('C:/Users/xyao0/Desktop/project/data/assembly/raw_data', DATE)
    if not os.path.isdir(FOLDER):
        os.mkdir(FOLDER)

    ### Connect to the robot #######
    FREQUENCY = 125
    host = "192.168.3.5"
    rtde, rtde_c, rtde_r = connect_robot(host, FREQUENCY)
    ### The TCP offset when collecting data###
    tmp = rtde_c.getTCPOffset()
    # tcp_offset = [0.0, 0.0, 0.18659, 0.0, 0.0, -3.141590000011358]

    # rtde_c.setTcp([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    ### Connect to the camera ###
    # my_camera = connect_camera()

    ### Connect to the writst camera
    url = 'http://192.168.36.19:8080/video'
    # connect_wrist_camera(url)
    ##### Set up pygame screen ##############
    n_trial = 0
    n_success = 0
    n_failure = 0
    pygame.init()
    size = [500, 700]
    WIN = pygame.display.set_mode(size)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    FPS = 20
    WIN.fill(WHITE)
    pygame.display.update()
    pygame.display.set_caption("Collect data")

    key_ring = {}

    ### These values could change from keyboard to keyboard
    Caps_lock = '1073741881' ### Used to switch between fast mode or slow mode
    SHIFT = '1073742049' ### Hold left shift key to control the robot and the gripper
    left = '1073741904'
    right = '1073741903'
    up = '1073741906'
    down = '1073741905'
    # page_up = '1073741899'
    # page_down = '1073741902'

    key_ring[SHIFT] = 0  # 1073742049 is the left shift key. This will be displayed in the screen  Caps lock = 1 + keys are the command set
    key_ring[Caps_lock] = 1
    key_pressed = 0  # key press and release will happen one after another
    key_released = 0

    font1 = pygame.font.SysFont('aril', 26)
    font2 = pygame.font.SysFont('aril', 35)
    font3 = pygame.font.SysFont('aril', 150)

    text1 = font1.render('Shift should be 1 to accept any keys', True, BLACK, WHITE)
    text3 = font1.render("'c': close the fingers", True, BLACK, WHITE)
    text4 = font1.render("'o': open the fingers", True, BLACK, WHITE)
    text5 = font1.render("'b': begin ", True, BLACK, WHITE)
    text6 = font1.render("'f': fail", True, BLACK, WHITE)
    text7 = font1.render("'s': success", True, BLACK, WHITE)
    text14 = font1.render("'d': defalt robot position", True, BLACK, WHITE)
    text15 = font1.render("'up arrow': move forward", True, BLACK, WHITE)
    text16 = font1.render("'down arrow': move backward", True, BLACK, WHITE)
    text17 = font1.render("'left arrow': move left", True, BLACK, WHITE)
    text18 = font1.render("'right arrow': move right", True, BLACK, WHITE)
    text19 = font1.render("'page up': move up", True, BLACK, WHITE)
    text20 = font1.render("'page down': move down", True, BLACK, WHITE)
    text22 = font1.render("'1': screw", True, BLACK, WHITE)
    text23 = font1.render("'2': unscrew", True, BLACK, WHITE)
    text24 = font1.render("'v': gripper vertical", True, BLACK, WHITE)
    text25 = font1.render("'h': gripper horizontal", True, BLACK, WHITE)
    text28 = font1.render("'7': slow, '8': medium, '9': fast ", True, BLACK, WHITE)
    text30 = font1.render("Press Caps Lock to change ", True, BLACK, WHITE)
    text21 = font1.render(f"Speed mode: ", True, BLACK, WHITE)
    text26 = font1.render(f"Reference frame:", True, BLACK, WHITE)
    text2 = font1.render(f'Shift : ', True, BLACK, WHITE)

    text8 = font2.render("#Trial", True, BLACK, WHITE)
    text9 = font2.render("#Success", True, BLACK, WHITE)
    text10 = font2.render("#Failure", True, BLACK, WHITE)
    text_counterclockwise = font1.render(f'"[" : counterclockwise rotate ', True, BLACK, WHITE)
    text_clockwize = font1.render(f'"]" : clockwise rotate ', True, BLACK, WHITE)
    text_keypoint = font1.render(f'"t" : keypoint timestamp ', True, BLACK, WHITE)
    text_defalt = font1.render(f'"k" : defalt pose timestamp ', True, BLACK, WHITE)
    text_failed_action = font1.render(f'"r" : falied action timestamp ', True, BLACK, WHITE)
    text_stop = font1.render(f'"enter" : stop movement', True, BLACK, WHITE)
    text_protective_stop = font1.render(f'"ESC" : protective stop ', True, BLACK, WHITE)


    clock = pygame.time.Clock()
    run = True
    speed_mode = 'slow'
    collecting_data = False
    reference_frame = 'base' if is_capslock_on() else 'tool'

    Ds = {'fast': 0.05, 'medium': 0.005, 'slow': 0.001}
    speed = Ds[speed_mode]
    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.KEYDOWN:
                key_pressed = event.key
                # print(key_pressed)
                key_ring[str(key_pressed)] = 1
                if key_ring[SHIFT] == 1:  # Left shift is pressed
                    if key_pressed == int(Caps_lock): ### Use Caps lock to change reference frame
                        if reference_frame == 'base':
                            reference_frame = 'tool'
                        else:
                            reference_frame = 'base'
                    if key_pressed == 98: ## Keyboard 'b' to start a demonstration##
                        rtde_r = rtde_receive.RTDEReceiveInterface(host, FREQUENCY)
                        keypoint = {}
                        keypoint['key_time_stamp'] = []
                        keypoint['defalt_pose_time_stamp'] = []
                        keypoint['failed_action_time_stamp'] = []
                        keypoint['wrist_camera_time'] = []
                        keypoint['defalt_pose_time_stamp'].append(rtde_r.getTimestamp())
                        keypoint['key_time_stamp'].append(rtde_r.getTimestamp())
                        collecting_data = True
                        n_trial += 1
                        trial_id = str(datetime.datetime.now().timestamp()).split('.')[0]
                        filename = os.path.join(FOLDER , trial_id)
                        my_camera.start_trial(filename)
                        # record_robot_thread = threading.Thread(target=record_robot, args = [rtde_r, filename, FREQUENCY])
                        rtde_r.startFileRecording(filename)
                        record_force_thread = threading.Thread(target=read_force, args = [filename])
                        # record_robot_thread.start()
                        record_force_thread.start()
                        print('Begin a trial')
                    elif key_pressed == 102 and collecting_data: #### Keyboard 'f' for a failed demonstration####
                        print('Failure')
                        SUCCESS = False
                        n_failure += 1
                        collecting_data = False
                        # record_robot_thread.join()
                        rtde_r.stopFileRecording()
                        record_force_thread.join()
                        # record_record_wrist_camera_thread.join()
                        my_camera.stop_trial()
                        keypoint['key_time_stamp'].append('f')
                        with open(f'{filename}_keypoint.pickle', 'wb') as f1:
                            pickle.dump(keypoint, f1)
                    elif key_pressed == 115 and collecting_data: #### Keyboard 's' for a sucessful demonstration####
                        SUCCESS = True
                        print('Success')
                        collecting_data = False
                        # record_robot_thread.join()
                        rtde_r.stopFileRecording()
                        record_force_thread.join()
                        # record_record_wrist_camera_thread.join()
                        n_success +=1
                        my_camera.stop_trial()
                        keypoint['key_time_stamp'].append('s')
                        with open(f'{filename}_keypoint.pickle', 'wb') as f:
                            pickle.dump(keypoint, f)
                    elif key_pressed == 111: #### Keyboard 'o' to open the gripper ####
                        # print('Open the gripper')
                        open_gripper(rtde)
                    elif key_pressed == 99: #### Keyboard 'c' to close the gripper####
                        # print('Close the gripper')
                        close_gripper(rtde)
                    elif key_pressed == 100: #### Keyboard 'd' to get back to defalt pose #####
                        move_to_defalt_pose(rtde_c)
                    elif key_pressed == 113: #### Keyboard 'q' to get back to defalt orientation #####
                        move_to_defalt_ori(rtde_r, rtde_c)
                    elif key_pressed == 107: ### Keyboard 'k' to mark the start of an action###
                        if collecting_data:
                            keypoint['defalt_pose_time_stamp'].append(rtde_r.getTimestamp())
                    elif key_pressed == 114: #### Keyboard 'r' to mark a failed action#####
                        if collecting_data:
                            keypoint['failed_action_time_stamp'].append(rtde_r.getTimestamp())
                    elif key_pressed == int(up): ### Keyboard up arrow to move forward###
                        move_foward(rtde_r, rtde_c, speed, reference_frame)
                    elif key_pressed == int(down): ### Keyboard down arrow to move backward###
                        move_backward(rtde_r, rtde_c, speed, reference_frame)
                    elif key_pressed == int(left): ### Keyboard left arrow to move left###
                        move_left(rtde_r, rtde_c, speed, reference_frame)
                    elif key_pressed == int(right): ### Keyboard right arrow to move right###
                        move_right(rtde_r, rtde_c, speed, reference_frame)
                    elif key_pressed == 61: ### Keyboard + to move up###
                        move_up(rtde_r, rtde_c, speed, reference_frame)
                    elif key_pressed == 45: ### Keyboard - to move down###
                        move_down(rtde_r, rtde_c, speed, reference_frame)
                    elif key_pressed == 93: #### keyboard right bracket to rotate clockwise####
                        rotate_wrist_clockwise(rtde_r, rtde_c)
                    elif key_pressed == 91: #### keyboard left bracket to rotate counter-clockwize#####
                        rotate_wrist_counterclockwise(rtde_r, rtde_c)
                    elif key_pressed == 49: ### keyboard 1 to assemble###
                        screw(rtde_r, rtde_c)
                    elif key_pressed == 50: ### keyboard 2 to undo assemble###
                        unscrew(rtde_r, rtde_c)
                    elif key_pressed == 118:### keyboard v to get back to vertical pose###
                        get_vertical(rtde_r,rtde_c)
                    elif key_pressed == 104: ### keyboard h to get to horizontal pose####
                        get_horizontal(rtde_r, rtde_c)
                    elif key_pressed == 116: ### keyboard t to mark a keypoint timestamp###
                        if collecting_data:
                            keypoint['key_time_stamp'].append(rtde_r.getTimestamp())
                    elif key_pressed == 51: ### keyboard 3###
                        wrist2_plus(rtde_r, rtde_c)
                    elif key_pressed == 52: ### keyboard 4###
                        wrist2_minus(rtde_r, rtde_c)
                    elif key_pressed == 53: ### keyboard 5###
                        wrist1_plus(rtde_r, rtde_c)
                    elif key_pressed == 54: ### keyboard 6###
                        wrist1_minus(rtde_r, rtde_c)
                    elif key_pressed == 55: ### keyboard 7 to change to slow mode###
                        speed_mode = 'slow'
                        speed = Ds[speed_mode]
                    elif key_pressed == 56: ### keyboard 8 to change to medium speed mode###
                        speed_mode = 'medium'
                        speed = Ds[speed_mode]
                    elif key_pressed == 57: ### keyboard 9 to change to fast speed mode###
                        speed_mode = 'fast'
                        speed = Ds[speed_mode]
                    elif key_pressed == 13: ### keyboard enter to stop the movement###
                        stop(rtde_c)
                    elif key_pressed == 27: ### keyboard space ESC to stop the robot###
                        protective_stop(rtde_c)
                    elif key_pressed == 112: ### keyboard p to take a picture from the wrist cam###
                        wrist_camera = cv2.VideoCapture(url)
                        ret, frame = wrist_camera.read()
                        tmp = str(datetime.datetime.now().timestamp()).split('.')[0] + '.jpeg'
                        fname = os.path.join(FOLDER, tmp)
                        cv2.imwrite(fname, frame)
                    elif key_pressed == 105: ### keyboard i to take a left and a right picture from the wrist cam###
                        if collecting_data:
                            keypoint['wrist_camera_time'].append(rtde_r.getTimestamp())
                        wrist_camera = cv2.VideoCapture(url)
                        ret, frame_left = wrist_camera.read()
                        move_right(rtde_r, rtde_c, 0.03, 'tool')
                        time.sleep(4)
                        wrist_camera = cv2.VideoCapture(url)
                        ret, frame_right = wrist_camera.read()
                        tmp = str(datetime.datetime.now().timestamp()).split('.')[0]
                        # fname_left = os.path.join(FOLDER, filename + '_' + tmp + '_left.jpeg')
                        # fname_right = os.path.join(FOLDER, filename + '_' + tmp + '_right.jpeg')
                        fname_left = os.path.join(FOLDER,  tmp + '_left.jpeg')
                        fname_right = os.path.join(FOLDER,  tmp + '_right.jpeg')
                        cv2.imwrite(fname_left, frame_left)
                        cv2.imwrite(fname_right, frame_right)
                        # cv2.imwrite(fname_left, frame_right)
                        # cv2.imwrite(fname_right, frame_left)
                        move_left(rtde_r, rtde_c, 0.03 , 'tool')
                    # elif key_pressed == 105: ### keyboard i to record a video from the wrist cam###
                    #     record_record_wrist_camera_thread = threading.Thread(target=record_wrist_camera, args=[url, filename])
                    #     record_record_wrist_camera_thread.start()

            elif event.type == pygame.KEYUP:
                key_released = event.key
                key_ring[str(key_released)] = 0
            else:
                pass  # ignoring other non-logitech joystick event types
        text27 = font2.render(f"{speed_mode}               ", True, RED, WHITE)
        text29 = font2.render(f"{reference_frame}    ", True, RED, WHITE)
        text31 = font2.render(f'{key_ring[SHIFT]}', True, RED, WHITE)
        text11 = font3.render(f"{n_trial}", True, RED, WHITE)
        text12 = font3.render(f"{n_success}", True, RED, WHITE)
        text13 = font3.render(f"{n_failure}", True, RED, WHITE)

        ### Shift
        WIN.blit(text1, (150, 0))
        WIN.blit(text2, (10, 0))
        WIN.blit(text31, (70, -2))


        ### Speed mode
        WIN.blit(text21, (10, 30))
        WIN.blit(text27, (130, 26))
        WIN.blit(text28, (250, 30))

        ### Reference frame
        WIN.blit(text26, (10, 60))
        WIN.blit(text29, (170, 54))
        WIN.blit(text30, (250, 60))

        WIN.blit(text3, (10, 90))
        WIN.blit(text4, (10, 120))
        WIN.blit(text5, (10, 150))
        WIN.blit(text6, (10, 180))
        WIN.blit(text7, (10, 210))
        WIN.blit(text14, (10, 240))
        WIN.blit(text22, (10, 270))
        WIN.blit(text23, (10, 300))

        WIN.blit(text15, (250, 90))
        WIN.blit(text16, (250, 120))
        WIN.blit(text17, (250, 150))
        WIN.blit(text18, (250, 180))
        WIN.blit(text19, (250, 210))
        WIN.blit(text20, (250, 240))
        WIN.blit(text24, (250, 270))
        WIN.blit(text25, (250, 300))


        WIN.blit(text8, (10, 340))
        WIN.blit(text9, (10, 475))
        WIN.blit(text10, (10, 600))

        ### trials, successes, and failures
        WIN.blit(text11, (150, 350))
        WIN.blit(text12, (150, 475))
        WIN.blit(text13, (150, 600))

        ### Rotations
        WIN.blit(text_clockwize, (250, 330))
        WIN.blit(text_counterclockwise, (250, 360))

        ## Keypoints
        WIN.blit(text_keypoint, (250, 390))
        WIN.blit(text_defalt, (250, 420))
        WIN.blit(text_failed_action, (250, 450))

        ## Stops
        WIN.blit(text_stop, (250, 480))
        WIN.blit(text_protective_stop, (250, 510))


        pygame.display.update()
    pygame.quit()


