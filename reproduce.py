import threading
# from RTDE import RTDE
# print(dir(RTDE))
# raise
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
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from glob import glob
import yaml
from scipy.spatial.transform import Rotation as R
from assembly import normalize_wrapper, create_tags, homogeneous_transform, get_obj_pose, analyze_images, analyze_videos, TFEncoderDecoder2, process_quaternions
import pandas as pd
import torch
from matplotlib import pyplot as plt
from control_robot import servoL
from process_data.main import svo_to_avi

def is_capslock_on():
    return True if ctypes.WinDLL("User32.dll").GetKeyState(0x14) else False


calibration_matrix = np.array([[-0.07071, 0.00779, 0.08539, -3.77588, -0.08876, 3.81759],
                               [-0.14419,   4.32640,   0.02626,  -2.17358,   0.14086,  -2.21257 ],
                               [6.18920,  -0.15334,   6.57294,  -0.27184,   6.62477,  -0.25805],
                               [-0.00169,   0.05222,  -0.18994,  -0.01967,   0.19266, -0.03254],
                               [0.20699,  -0.00403,  -0.11136,   0.04967,  -0.10984,  -0.04233],
                               [0.00134,  -0.11867,   0.00232,  -0.11984,   0.00672,  -0.12104]])

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

def get_obj_pose_zed():
    video_id = str(datetime.datetime.now().timestamp()).split('.')[0]
    # video_id = '1715130445'
    video_dir = os.path.join(FOLDER, 'zed_videos', video_id)
    os.makedirs(video_dir, exist_ok=True)
    filename = os.path.join(video_dir, video_id)
    my_camera.start_trial(filename)
    time.sleep(1)
    my_camera.stop_trial()
    ## convert svo to avi files ###
    svo_to_avi(filename + '.svo', video_dir)
    vids = glob(os.path.join(video_dir, '*.avi'))
    ### Analyze videos to get the 3d coordinates of markers
    analyze_videos(vids, task_config)
    ## Get object pose ###
    h5_3d_zed = os.path.join(video_dir, '3d_combined', 'markers_trajectory_3d.h5')
    df_pose_zed = get_obj_pose(h5_3d_zed, video_dir, zed_in_base, obj_templates_in_base_for_zed,
                               HT_template_in_base_for_zed, 'zed', window_size=5)
    print('Get object pose from zed camera.')
    return df_pose_zed

def get_obj_pose_wrist():
    current_pose = rtde_r.getActualTCPPose()
    pos = np.array(current_pose[:3]) * scale
    rotmatrix = R.from_rotvec(current_pose[3:]).as_matrix()
    tcp_in_robot = homogeneous_transform(rotmatrix, pos)
    wrist_camera = cv2.VideoCapture(url)
    ret, frame_left = wrist_camera.read()
    move_right(rtde_r, rtde_c, 0.05, 'tool')
    time.sleep(4)
    wrist_camera = cv2.VideoCapture(url)
    ret, frame_right = wrist_camera.read()
    image_id = str(datetime.datetime.now().timestamp()).split('.')[0]
    image_dir = os.path.join(FOLDER, demo_dir, 'wrist_images', image_id)
    os.makedirs(image_dir, exist_ok=True)
    fname_left = os.path.join(image_dir, image_id + '_left.jpeg')
    fname_right = os.path.join(image_dir, image_id + '_right.jpeg')
    cv2.imwrite(fname_left, frame_left)
    cv2.imwrite(fname_right, frame_right)
    move_left(rtde_r, rtde_c, 0.05, 'tool')
    ## Analyze images##
    analyze_images(image_dir, task_config)
    ## Get object pose ###
    wrist_cam_in_base = tcp_in_robot @ wrist_camera_in_tcp
    h5_3d_wrist = glob(os.path.join(image_dir, '3d_combined', f'*.h5'))[0]
    df_pose_wrist = get_obj_pose(h5_3d_wrist, image_dir, wrist_cam_in_base, obj_templates_in_base_for_wrist,
                                 HT_template_in_base_for_wrist, 'wrist', window_size=1)
    print('Get object pose from wrist camera.')
    return df_pose_wrist

def update_obj_sequence(df_pose_zed, df_pose_wrist, all_objs, task_dims, obj_tags):
    obj_pose_seq = []
    if df_pose_zed is None and df_pose_wrist is None:
        raise('No video or image is captured.')
    individuals = list(df_pose_zed.columns)
    for object_ind in all_objs:
        individual = [ind for ind in individuals if object_ind in ind][0]
        obj_pose_zed = df_pose_zed[individual][task_dims].to_numpy()
        try:
            obj_pose_wrist = df_pose_wrist[individual][task_dims].to_numpy()
        except (KeyError, TypeError) as error: ### KeyError if this object is not detected. TypeError if df_pose_wrist is None
            obj_pose_wrist = np.zeros(len(task_dims))
            obj_pose_wrist[:] = np.nan
        if not np.isnan(obj_pose_wrist).any():
            obj_pose = obj_pose_wrist
        else:
            obj_pose = obj_pose_zed
        obj_pose = np.concatenate([obj_pose, obj_tags[object_ind]])
        obj_pose_seq.append(obj_pose)
    return torch.tensor(np.array(obj_pose_seq))


def plot_traj_and_obj_pos(traj_pos, obj_data, colors, all_objs):
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    traj_color = colors['traj']
    line = ax.plot(traj_pos[:, 0], traj_pos[:, 1], (traj_pos[:, 2]),
                   color=traj_color, label=f'traj')
    ax.plot(traj_pos[-1, 0], traj_pos[-1, 1], traj_pos[-1, 2], 'x',
            color=traj_color, label=f'end')
    ax.plot(traj_pos[0, 0], traj_pos[0, 1], traj_pos[0, 2], 'o',
            color=traj_color, label=f'start')

    for i in range(obj_data.shape[0]):
        obj_pos = obj_data[i, :3]
        unique_obj = all_objs[i]
        color = colors[unique_obj]
        ax.plot(obj_pos[0], obj_pos[1], obj_pos[2], 's', color=color)
    plt.show()
    return

def plot_traj_ori(axes, quats):
    for i, ax in enumerate(axes):
        ax.plot(quats[:, i], color='red')
        ax.set_ylim(-np.sqrt(2), np.sqrt(2))

def update_object_pose_zed(demo_dir):
    video_dirs = sorted(os.listdir(os.path.join(demo_dir, 'zed_videos')))
    if len(video_dirs) != 0:
        video_dir = video_dirs[-1]
        df_pose_zed = pd.read_csv(os.path.join(demo_dir, 'zed_videos', video_dir, 'obj_pose_zed.csv'), index_col=0)
    else:
        df_pose_zed = None
    df_pose_wrist = None
    obj_seq = update_obj_sequence(df_pose_zed, df_pose_wrist, all_objs, task_dims, obj_tags)
    return obj_seq

def update_object_pose_wrist(demo_dir):
    video_dirs = sorted(os.listdir(os.path.join(demo_dir, 'zed_videos')))
    if len(video_dirs) != 0:
        video_dir = video_dirs[-1]
        df_pose_zed = pd.read_csv(os.path.join(demo_dir, 'zed_videos', video_dir, 'obj_pose_zed.csv'), index_col=0)
    else:
        df_pose_zed = None
    image_dirs = sorted(os.listdir(os.path.join(demo_dir, 'wrist_images')))
    if len(image_dirs) != 0:
        image_dir = image_dirs[-1]
        df_pose_wrist = pd.read_csv(os.path.join(demo_dir, 'wrist_images', image_dir, 'obj_pose_wrist.csv'),
                                    index_col=0)
    else:
        df_pose_wrist = None
    obj_seq = update_obj_sequence(df_pose_zed, df_pose_wrist, all_objs, task_dims, obj_tags)
    return obj_seq



def predict_traj_and_action(new_model, obj_seq, traj_seq):
    obj_seq_normalized = norm_func(obj_seq.clone())
    obj_seq_normalized = obj_seq_normalized[None, :, :]
    obj_seq_normalized = obj_seq_normalized.to(device)
    traj_seq = traj_seq.to(device)
    predicted_traj_tf, actin_tag_pred = new_model(obj_seq_normalized, traj_seq)
    predicted_traj = predicted_traj_tf.cpu().detach().numpy()[0]
    predicted_traj[:, :3] = predicted_traj[:, :3] * train_std + train_mean
    actin_tag_pred = actin_tag_pred.cpu().detach().numpy()[0]
    return predicted_traj, np.argmax(actin_tag_pred)

def quat_to_rotvec(traj):
    traj_new = np.zeros((len(traj), 6))
    traj_new[:, :3] = traj[:, :3]
    traj_new[:, 3:] = R.from_quat(traj[:, 3:]).as_rotvec()
    return traj_new


if __name__ == "__main__":
    ### Create dirs to reproduce
    DATE = str(datetime.date.today())
    DATE = '2024-05-28'
    FOLDER = os.path.join('C:/Users/xyao0/Desktop/project/assembly/data/reproduce', DATE)
    if not os.path.isdir(FOLDER):
        os.makedirs(FOLDER, exist_ok=True)

    ### Load transformer data
    assembly_dir = 'C:/Users/xyao0/Desktop/project/assembly'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_len = 170
    task_config = 'C:/Users/xyao0/Desktop/project/assembly/data/task_config.yaml'
    with open(task_config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    project_dir = config['project_path']  # Modify this to your need.
    # with open(os.path.join(project_dir, 'data', 'processed', 'wrist_cam_ind.pickle'), 'rb') as f:
    #     wrist_cam_ind = pickle.load(f)

    wrist_cam_ind = {'action_0': 18, 'action_1': 14, 'action_2': 15}
    grasp_inds = {'action_0': 78, 'action_1': 75, 'action_2':76}
    open_inds = {'action_0': 134, 'action_1': 134, 'action_2':-1}

    all_objs = config['objects']
    colors = {'bolt': 'green', 'nut': 'yellow', 'bin': 'black', 'jig': 'purple', 'traj': 'red'}
    task_dims = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
    # instaniate all object type and enable/disable via-points
    unique_objs = ['trajectory'] + sorted(list(set(all_objs)))
    obj_tags = create_tags(unique_objs)
    with open(os.path.join(assembly_dir, 'transformer', 'train_stat.pickle'), 'rb') as f:
        train_stat = pickle.load(f)
    train_mean = train_stat['mean']
    train_std = train_stat['std']
    norm_func = normalize_wrapper(train_mean, train_std)
    n_dims = len(task_dims)
    n_objs = len(unique_objs)
    task_dim = n_dims + n_objs
    traj_seq = torch.zeros((1, max_len, task_dim), dtype = torch.double)
    traj_seq[0, :, n_dims:] = obj_tags['trajectory']
    n_encoder_layers = 3
    n_decoder_layers = 3
    n_tasks = 3
    ### Wrist model
    new_model = TFEncoderDecoder2(task_dim=task_dim, traj_dim=n_dims, n_tasks=n_tasks, n_objs=n_objs - 1,
                                  embed_dim=32, nhead=8, max_len=max_len, num_encoder_layers=n_encoder_layers,
                                  num_decoder_layers=n_decoder_layers, device=device)

    model_ind = 5000
    date = '05_01_2024'
    PATH = os.path.join(assembly_dir, 'transformer', date, f'model_{model_ind}_action.pth')
    new_model.load_state_dict(torch.load(PATH,  map_location=torch.device(device)))
    new_model.eval()
    ### Zed model
    new_model_zed = TFEncoderDecoder2(task_dim=task_dim, traj_dim=n_dims, n_tasks=n_tasks, n_objs=n_objs - 1,
                                  embed_dim=32, nhead=8, max_len=max_len, num_encoder_layers=n_encoder_layers,
                                  num_decoder_layers=n_decoder_layers, device=device)

    model_ind = 4005
    date = '2024-05-28'
    PATH = os.path.join(assembly_dir, 'transformer', date, f'model_{model_ind}_action.pth')
    new_model_zed.load_state_dict(torch.load(PATH, map_location=torch.device(device)))
    new_model_zed.eval()


    ### Load transformations
    with open(os.path.join(project_dir, 'transformations', 'zed_in_base.pickle'), 'rb') as f:
        zed_in_base = pickle.load(f)
    with open(os.path.join(project_dir, 'transformations', 'wrist_cam_in_tcp.pickle'), 'rb') as f:
        wrist_camera_in_tcp = pickle.load(f)
    with open(os.path.join(project_dir, 'transformations', 'HT_template_in_base_for_zed.pickle'), 'rb') as f:
        HT_template_in_base_for_zed = pickle.load(f)
    with open(os.path.join(project_dir, 'transformations', 'HT_template_in_base_for_wrist.pickle'), 'rb') as f:
        HT_template_in_base_for_wrist = pickle.load(f)
    with open(os.path.join(project_dir, 'transformations', 'template_in_base_for_zed.pickle'), 'rb') as f:
        obj_templates_in_base_for_zed = pickle.load(f)
    with open(os.path.join(project_dir, 'transformations', 'template_in_base_for_wrist.pickle'), 'rb') as f:
        obj_templates_in_base_for_wrist = pickle.load(f)

    ### Connect to the robot #######
    FREQUENCY = 1
    dt = 1/ FREQUENCY

    scale = 1000
    host = "192.168.3.5"
    rtde, rtde_c, rtde_r = connect_robot(host, FREQUENCY)

    ### The TCP offset when collecting data###
    # tmp = rtde_c.getTCPOffset()
    # tcp_offset = [0.0, 0.0, 0.18659, 0.0, 0.0, -3.141590000011358]

    # rtde_c.setTcp([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    ### Connect to the camera ###
    # my_camera = connect_camera()

    ### Connect to the writst camera
    url = 'http://192.168.37.19:8080/video'
    connect_wrist_camera(url)

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
                print(key_pressed)
                key_ring[str(key_pressed)] = 1
                demo_dir = FOLDER
                if key_ring[SHIFT] == 1:  # Left shift is pressed
                    if key_pressed == 98:  ## Keyboard 'b' to start a demonstration##
                        demo_id = str(datetime.datetime.now().timestamp()).split('.')[0]
                        demo_dir = os.path.join(FOLDER, demo_id)
                        zed_dir = os.path.join(demo_dir, 'zed_videos')
                        wrist_dir = os.path.join(demo_dir, 'wrist_images')
                        os.makedirs(zed_dir, exist_ok=True)
                        os.makedirs(wrist_dir, exist_ok=True)
                    elif key_pressed == 108:  ## Keyboard 'l' to use the last demonstration##
                        demo_id = sorted(os.listdir(FOLDER))[-1]
                        demo_dir = os.path.join(FOLDER, demo_id)
                    elif key_pressed == 122:  ## Keyboard 'z' to get object pose from zed camera##
                       df_pose_zed = get_obj_pose_zed()
                    if key_pressed == 119:  ## Keyboard 'w' to capture 2 images from wrist camera and get object pose##
                        df_pose_wrist = get_obj_pose_wrist()
                    elif key_pressed == 117: ### Keyboard 'u' to update the object pose sequence
                        obj_seq = update_object_pose(demo_dir)
                    elif key_pressed == 112: ### Keyboard 'p' to predict the trajectory
                        obj_seq_normalized = norm_func(obj_seq.clone())
                        obj_seq_normalized = obj_seq_normalized[None, :, :]
                        obj_seq_normalized = obj_seq_normalized.to(device)
                        traj_seq = traj_seq.to(device)
                        predicted_traj_tf, actin_tag_pred = new_model(obj_seq_normalized, traj_seq)
                        predicted_traj = predicted_traj_tf.cpu().detach().numpy()[0]
                        predicted_traj[:, :3] = predicted_traj[:, :3] * train_std + train_mean
                        # predicted_traj[:, 3:] = process_quaternions(predicted_traj[:, 3:])
                        actin_tag_pred = actin_tag_pred.cpu().detach().numpy()[0]
                    elif key_pressed == 102: ### Keyboard 'f' to plot the trajectory and object pose
                        fig = plt.figure(figsize=(9, 5))
                        ax = fig.add_subplot(1, 1, 1, projection='3d')
                        plot_traj_and_obj_pos(ax, predicted_traj, obj_seq, colors, all_objs)

                        fig2, axes = plt.subplots(4, 1, figsize=(9, 10))
                        plot_traj_ori(axes, predicted_traj[:, 3:])
                        plt.show()
                    elif key_pressed == 101:  ### Keyboard 'e' to execute the action
                        ### Get object pose from zed camera
                        open_gripper(rtde)
                        # df_pose_zed = get_obj_pose_zed()
                        obj_seq = update_object_pose_zed(demo_dir)

                        ### Move above object to take pictures
                        predicted_traj, actin_pred = predict_traj_and_action(new_model_zed, obj_seq, traj_seq)
                        action = f'action_{actin_pred}'
                        print(f'This is {action}!!!!!!!!!!')
                        print(len(predicted_traj))
                        wrist_ind = wrist_cam_ind[action]
                        grasp_ind = grasp_inds[action]
                        open_ind = open_inds[action]
                        plot_traj_and_obj_pos(predicted_traj, obj_seq, colors, all_objs)
                        traj = predicted_traj[:wrist_ind]
                        traj_new = quat_to_rotvec(traj)
                        traj_new[:,:3] = traj_new[:,:3] / scale
                        servoL(traj_new, rtde_c, dt)
                        time.sleep(3)

                        ### Move to grasp object
                        df_pose_wrist = get_obj_pose_wrist()
                        obj_seq = update_object_pose_wrist(demo_dir)
                        predicted_traj, actin_pred = predict_traj_and_action(new_model, obj_seq, traj_seq)
                        plot_traj_and_obj_pos(predicted_traj, obj_seq, colors, all_objs)
                        traj = predicted_traj[wrist_ind:grasp_ind]
                        traj_new = quat_to_rotvec(traj)
                        traj_new[:, :3] = traj_new[:, :3] / scale
                        servoL(traj_new, rtde_c, dt)
                        close_gripper(rtde)

                        ### Move to release object
                        traj = predicted_traj[grasp_ind:open_ind]
                        traj_new = quat_to_rotvec(traj)
                        traj_new[:, :3] = traj_new[:, :3] / scale
                        servoL(traj_new, rtde_c, dt)
                        open_gripper(rtde)

                        ### Move away from object
                        traj = predicted_traj[open_ind:]
                        traj_new = quat_to_rotvec(traj)
                        traj_new[:, :3] = traj_new[:, :3] / scale
                        servoL(traj_new, rtde_c, dt)





                    elif key_pressed == 111:  #### Keyboard 'o' to open the gripper ####
                        # print('Open the gripper')
                        open_gripper(rtde)
                    elif key_pressed == 99:  #### Keyboard 'c' to close the gripper####
                        # print('Close the gripper')
                        close_gripper(rtde)
                    elif key_pressed == 100: #### Keyboard 'd' to get back to defalt pose #####
                        move_to_defalt_pose(rtde_c)
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


