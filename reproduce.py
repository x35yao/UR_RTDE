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
import shutil
import yaml
from scipy.spatial.transform import Rotation as R
from assembly import TFEncoderDecoder4, create_tags, get_obj_pose, homogeneous_transform, svo_to_avi, quat_to_rotvect, rotvect_to_quat, normalize_wrapper, inverse_homogeneous_transform, vec2homo, homo2vec
import pandas as pd
import torch
from matplotlib import pyplot as plt
from control_robot import servoL
import urllib
import subprocess

class Robot():
    def __init__(self, host, frequency):
        self.host = host
        self.frequency = frequency
        self.dt = 1/frequency

    def connect(self):
        try:
            self.rtde_r = rtde_receive.RTDEReceiveInterface(self.host, self.frequency)
            self.rtde_c = rtde_control.RTDEControlInterface(self.host, self.frequency, RTDEControl.FLAG_CUSTOM_SCRIPT)
            self.rtde = rtde_io.RTDEIOInterface(self.host)
            print("Robot connection success")
        except RuntimeError:
            print('Robot connection failure')

    # def get_tcp_pose(self):
    #     return self.rtde_r.getActualTCPPose()

    def get_tcp_pose(self):
        print('This is for testing purposes, change it back to the function above!!!!!!!!')
        return np.array([1,2,3,4,5,6])

    def servoL(self, traj):
        servoL(traj, self.rtde_c, self.dt)

class Zed():
    def __init__(self, destfolder, videotype = 'avi'):
        self.destfolder = destfolder
        self.videotype = videotype

    def connect_camera(self):
        self.cam = tc.command_camera()
        if self.cam.connected == 1:
            print("Camera connection success")
        else:
            print("Camera connection failure, is the server opened?")

    def capture_video(self, destfolder = None, duration = 1):
        if destfolder is None:
            destfolder = self.destfolder
        os.makedirs(destfolder, exist_ok=True)
        fname = os.path.join(destfolder, 'zed')
        self.cam.start_trial(fname)
        time.sleep(duration)
        self.cam.stop_trial()
        svo_to_avi(fname + '.svo2', outdir = destfolder)

    def analyze_video(self, destfolder = None, visualize = False):
        if destfolder is None:
            destfolder = self.destfolder
        env_name = r"C:/Users/xyao0/anaconda3/envs/DLC"
        script_path = "C:/Users/xyao0/Desktop/project/assembly/analyze_videos.py"
        conda_path = "C:/Users/xyao0/anaconda3/ScripDts/conda.exe"
        subprocess.run(["conda", "run", "-p", env_name, "python", script_path, destfolder, self.videotype, str(visualize)], shell=True)

    def clear_folder(self, destfolder=None):
        if destfolder is None:
            destfolder = self.destfolder
            # Check if the folder exists
        if os.path.exists(destfolder) and os.path.isdir(destfolder):
            for item in os.listdir(destfolder):
                item_path = os.path.join(destfolder, item)
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.remove(item_path)  # Remove file or symbolic link
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)  # Remove directory
            print(f"Cleared contents of '{destfolder}'")
        else:
            print(f"The folder '{destfolder}' does not exist or is not a directory.")

class Wrist():
    def __init__(self, destfolder, url_left, url_right, imagetype = 'png'):
        self.url_left = url_left
        self.url_right = url_right
        self.destfolder = destfolder
        self.imagetype = imagetype

    def take_pics(self, destfolder = None):
        if destfolder is None:
            destfolder = self.destfolder
        os.makedirs(destfolder, exist_ok=True)
        wrist_camera_left = urllib.request.urlopen(self.url_left)
        wrist_camera_right = urllib.request.urlopen(self.url_right)
        frame_left = cv2.imdecode(np.array(bytearray(wrist_camera_left.read()), dtype=np.uint8), -1)
        frame_right = cv2.imdecode(np.array(bytearray(wrist_camera_right.read()), dtype=np.uint8), -1)
        fname_left = str(datetime.datetime.now().timestamp()).split('.')[0] + '_left.png'
        fname_right = fname_left.replace('left', 'right')
        path_left = os.path.join(destfolder, fname_left)
        path_right = os.path.join(destfolder, fname_right)
        cv2.imwrite(path_left, frame_left)
        cv2.imwrite(path_right, frame_right)

    def analyze_image(self, destfolder = None, visualize = True):
        if destfolder is None:
            destfolder = self.destfolder
        env_name = r"C:/Users/xyao0/anaconda3/envs/DLC"
        script_path = "C:/Users/xyao0/Desktop/project/assembly/analyze_images.py"
        conda_path = "C:/Users/xyao0/anaconda3/ScripDts/conda.exe"
        subprocess.run(["conda", "run", "-p", env_name, "python", script_path, destfolder, self.imagetype, str(visualize)], shell=True)

    def clear_folder(self, destfolder = None):
        if destfolder is None:
            destfolder = self.destfolder
            # Check if the folder exists
        if os.path.exists(destfolder) and os.path.isdir(destfolder):
            for item in os.listdir(destfolder):
                item_path = os.path.join(destfolder, item)
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.remove(item_path)  # Remove file or symbolic link
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)  # Remove directory
            print(f"Cleared contents of '{destfolder}'")
        else:
            print(f"The folder '{destfolder}' does not exist or is not a directory.")

class Network():
    def __init__(self, max_len, train_stat, device, n_dims = 7, n_tasks = 3, n_objs = 4, embed_dim = 64, n_head = 8, n_encoder_layers = 3, n_decoder_layers = 3 ):
        self.traj_seq_dim = n_dims + n_objs + 1
        self.norm_func = normalize_wrapper(train_stat['mean'], train_stat['std'])
        self.mean = train_stat['mean']
        self.std = train_stat['std']
        self.max_len = max_len
        self.model = TFEncoderDecoder4(task_dim=n_dims, target_dim= self.traj_seq_dim, source_dim= self.traj_seq_dim,
                                      n_tasks=n_tasks,
                                      embed_dim=embed_dim, nhead = n_head, max_len= max_len,
                                      num_encoder_layers=n_encoder_layers,
                                      num_decoder_layers=n_decoder_layers, device=device)
    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    def get_obj_tags(self, all_objs):
        self.obj_tags = create_tags(all_objs)

    def update_obj_sequence(self, df_pose_zed, df_pose_wrist , tcp_pose, all_objs, task_dims = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']):
        '''
        :param df_pose_zed: Current object pose detected by zed cam
        :param df_pose_wrist: Current object pose detected by wrist cam
        :param tcp_pose: Current tcp pose
        :param task_dims: Dimensions of the task
        :return:
        '''
        obj_pose_seq = []
        if df_pose_wrist is None:
            df_pose_wrist = df_pose_zed
        individuals = list(df_pose_zed.columns)
        for object_ind in all_objs:
            if object_ind != 'trajectory': ### Actual object excluding the trajectory object
                individual = [ind for ind in individuals if object_ind in ind][0]
                obj_pose_zed = df_pose_zed[individual][task_dims].to_numpy()
                obj_pose_wrist = df_pose_wrist[individual][task_dims].to_numpy()
                if not np.isnan(obj_pose_wrist).any():
                    obj_pose = obj_pose_wrist
                else:
                    obj_pose = obj_pose_zed
            else: ### The trajectory object
                obj_pose = tcp_pose
            obj_pose = np.concatenate([obj_pose, self.obj_tags[object_ind]])
            obj_pose_seq.append(obj_pose)
        obj_pose_seq = torch.tensor(obj_pose_seq)
        obj_seq_normalized = self.norm_func(obj_pose_seq.clone())
        self.obj_seq = obj_seq_normalized[None, :, :]
        return self.obj_seq

    def predict_traj(self, traj_len):
        self.model.eval()
        traj_hidden = torch.zeros((1, self.max_len, self.traj_seq_dim), dtype=torch.double)
        traj_hidden[0, :, 7] = 1
        padding_mask = torch.zeros(1, self.max_len)
        padding_mask[0, traj_len:] = 1
        padding_mask = padding_mask > 0
        output_seq, _ = self.model(self.obj_seq, traj_hidden, tgt_padding_mask=padding_mask)
        self.output_seq = output_seq.detach().numpy()[0]
        print(self.output_seq.shape, 'ffffff')
        traj = quat_to_rotvect(self.output_seq)
        traj[:, :3] = (traj[:, :3] * self.std + self.mean) / SCALE
        return traj[:traj_len]

    def plot_traj_and_obj_pos(self, traj_pos, obj_data, colors, all_objs):
        fig = plt.figure(figsize=(9, 5))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        line = ax.plot(traj_pos[:, 0], traj_pos[:, 1], (traj_pos[:, 2]),
                       color=colors['traj'], label=f'traj')
        ax.plot(traj_pos[-1, 0], traj_pos[-1, 1], traj_pos[-1, 2], 'x',
                color=colors['traj'], label=f'end')
        ax.plot(traj_pos[0, 0], traj_pos[0, 1], traj_pos[0, 2], 'o',
                color=colors['traj'], label=f'start')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        for i in range(len(obj_data)):
            obj_pos = obj_data[i, :3]
            unique_obj = all_objs[i]
            color = colors[unique_obj]
            ax.plot(obj_pos[0], obj_pos[1], obj_pos[2], 's', color=color)
        return ax


class PoseProcessor():
    def __init__(self, transformation_dir, destfolder_zed, destfolder_wrist):
        with open(os.path.join(transformation_dir, 'zed_in_base.pickle'), 'rb') as f:
            self.zed_in_base = pickle.load(f)
        with open(os.path.join(transformation_dir, 'wrist_cam_in_tcp.pickle'), 'rb') as f:
            self.wrist_camera_in_tcp = pickle.load(f)
        with open(os.path.join(transformation_dir, 'HT_template_in_base_for_zed.pickle'), 'rb') as f:
            self.HT_template_in_base_for_zed = pickle.load(f)
        with open(os.path.join(transformation_dir, 'HT_template_in_base_for_wrist.pickle'), 'rb') as f:
            self.HT_template_in_base_for_wrist = pickle.load(f)
        with open(os.path.join(transformation_dir, 'template_in_base_for_zed.pickle'), 'rb') as f:
            self.obj_templates_in_base_for_zed = pickle.load(f)
        with open(os.path.join(transformation_dir, 'template_in_base_for_wrist.pickle'), 'rb') as f:
            self.obj_templates_in_base_for_wrist = pickle.load(f)
        with open(os.path.join(transformation_dir, 'template_in_obj_for_zed.pickle'), 'rb') as f:
            self.obj_templates_in_obj_for_zed = pickle.load(f)
        with open(os.path.join(transformation_dir, 'template_in_obj_for_wrist.pickle'), 'rb') as f:
            self.obj_templates_in_obj_for_wrist = pickle.load(f)
        self.destfolder_zed = destfolder_zed
        self.destfolder_wrist = destfolder_wrist

    def get_pose_zed(self, h5_3d_file = None):
        if h5_3d_file is None:
            h5_3d_file = os.path.join(self.destfolder_zed, '3d_combined', 'markers_trajectory_3d.h5')
        if not os.path.isfile(h5_3d_file):
            self.obj_pose_zed = None
        else:
            self.obj_pose_zed = get_obj_pose(h5_3d_file, self.destfolder_zed, self.zed_in_base, self.obj_templates_in_base_for_zed, self.HT_template_in_base_for_zed, suffix = 'zed',
                         window_size=5)

    def get_pose_wrist(self, current_pose, h5_3d_file = None):
        if h5_3d_file is None:
            h5_3d_file = os.path.join(self.destfolder_wrist, '3d_combined', 'markers_trajectory_wrist_3d.h5')
        if not os.path.isfile(h5_3d_file):
            self.obj_pose_wrist = None
        else:
            pos = np.array(current_pose[:3])
            rotmatrix = R.from_rotvec(current_pose[3:]).as_matrix()
            tcp_in_robot = homogeneous_transform(rotmatrix, pos)
            wrist_cam_in_base = tcp_in_robot @ self.wrist_camera_in_tcp
            self.obj_pose_wrist = get_obj_pose(h5_3d_file, self.destfolder_wrist, wrist_cam_in_base, self.obj_templates_in_base_for_wrist,
                                             self.HT_template_in_base_for_wrist, suffix='wrist',
                                             window_size=1)

class IBVS():
    def __init__(self, robot, wrist, max_lmbda = 0.1, thresh = 20, max_run = 3, scale = 1000):
        self.wrist = wrist
        self.max_lmbda = max_lmbda
        self.thersh = thresh
        self.max_run = max_run
        self.n_run = 0
        self.finished = False
        self.robot = robot
        self.scale = scale

    def process_traj(self):
        ### Convert tcp poses using wrist camera poses
        cam_poses_path = 'C:/Users/xyao0/Desktop/project/assembly/data/reproduce/ibvs/plan/cam_poses.pickle'
        if os.path.isfile(cam_poses_path):
            with open(cam_poses_path, 'rb') as f:
                tmp = pickle.load(f)
            cam_poses = tmp[:-1]
            is_finished = tmp[-1]
            with open('C:/Users/xyao0/Desktop/project/assembly/transformations/2024-08-28/wrist_cam_in_tcp.pickle',
                      'rb') as f:
                H_cam_in_tcp = pickle.load(f)

            vec_tcp_in_base = np.array(self.robot.get_tcp_pose())
            H_tcp_in_base = vec2homo(vec_tcp_in_base)
            H_cam_in_tcp[:-1, -1] /= self.scale
            H_tcp_in_cam = inverse_homogeneous_transform(H_cam_in_tcp)

            tcp_poses = []
            tcp_poses_vec = []
            for cam_pose in cam_poses:
                tcp_pose = H_tcp_in_base @ H_cam_in_tcp @ cam_pose @ H_tcp_in_cam
                tcp_poses.append(tcp_pose)
                tcp_poses_vec.append(homo2vec(tcp_pose))
            tcp_poses_vec = np.array(tcp_poses_vec)
            os.remove(cam_poses_path) ### delete the file to avoid repeated movement
        else:
            tcp_poses_vec = None ### Return None if camera pose is not found
            is_finished = False
        return tcp_poses_vec, is_finished

    def run(self, current_dir, target_dir, individuals):
        for i in range(self.max_run):
            if self.finished:
                print('IBVS finished due to threshold statisfied.')
                break
            self.n_run = i
            if self.n_run < self.max_run:
                env_name = r"C:/Users/xyao0/anaconda3/envs/rvc3"
                script_path = "C:/Users/xyao0/Desktop/project/rvc/ibvs_stereo.py"
                lmbda = self.max_lmbda * 1 / (self.max_run - self.n_run)
                args = [current_dir, target_dir, str(individuals), f'{lmbda}', str(self.thersh)]
                subprocess.run(["conda", "run", "-p", env_name, "python", script_path] + args, shell=True)

                ### Handel projectory ###
                planned_traj, is_finished = self.process_traj()
                # print(is_finished)
                # self.finished = is_finished
                # fig = plt.figure(figsize=(9, 5))
                # ax = fig.add_subplot(1, 1, 1, projection='3d')
                # ax.plot(planned_traj[0, 0], planned_traj[0, 1], planned_traj[0, 2], 'o', color = 'red')
                # ax.plot(0, 0, 0, 'o', color = 'blue')
                # ax.plot(planned_traj[:,0], planned_traj[:,1], planned_traj[:,2])
                # ax.set_xlabel('x')
                # ax.set_ylabel('y')
                # ax.set_zlabel('z')
                # plt.show()
                # raise
                ### Move robot ###
                self.robot.servoL(planned_traj)

                ### Take pics ###
                if self.n_run < self.max_run - 1:
                    self.wrist.clear_folder()
                    self.wrist.take_pics()
                    self.wrist.analyze_image()

        print('IBVS finished due to max runs')


class Planner():
    def __init__(self, net, robot, zed, wrist, pose_processor, action_summary, all_objs):
        self.net = net
        self.robot = robot
        self.zed = zed
        self.wrist = wrist
        self.wrist_start_inds = action_summary['median_wrist_inds']
        self.wrist_end_inds = action_summary['wrist_end_inds']
        self.median_len = action_summary['median_traj_len']
        self.pose_processor = pose_processor
        self.curret_ind = 0
        self.all_objs = all_objs

    def step(self, current_ind = None):
        current_tcp_in_base = np.array(robot.get_tcp_pose())
        current_tcp_in_base[:3] = current_tcp_in_base[:3] * SCALE
        current_tcp_in_base_quat = rotvect_to_quat(current_tcp_in_base).flatten()
        if current_ind is None:
            current_ind = self.curret_ind

        if current_ind in self.wrist_start_inds: ### Do IBVS ###
            pass
        else:### Use transformer
            if current_ind == 0:
                # self.zed.clear_folder()
                # self.zed.capture_video()
                # self.zed.analyze_video()
                self.pose_processor.get_pose_zed()
                self.net.update_obj_sequence(self.pose_processor.obj_pose_zed, None, current_tcp_in_base_quat, self.all_objs)
                traj = self.net.predict_traj(self.median_len)
                print(traj.shape)
                raise
            else:
            #     self.wrist.clear_folder()
            #     self.wrist.take_pics()
            #     self.wrist.analyze_image()
                self.pose_processor.get_pose_wrist()

            ### Do transformer ###

if __name__ == "__main__":
    SCALE = 1000
    ### Load task config ###
    project_dir = 'C:/Users/xyao0/Desktop/project/assembly'
    with open(os.path.join(project_dir, 'data/task_config.yaml')) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    ### Create dirs to reproduce
    DATE = str(datetime.date.today())
    FOLDER = os.path.join(project_dir, 'data/reproduce')
    if not os.path.isdir(FOLDER):
        os.makedirs(FOLDER, exist_ok=True)

    ### Load transformer data
    max_len = 200
    objs = sorted(config['objects'])
    all_objs = objs + ['trajectory']
    actions = config['actions']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Cuda available: ", torch.cuda.is_available())
    model_folder = 'speed_grasp_weights-hidden-64-smooth_pos'
    model_ind = 50000
    seed = 123
    model_path = f'C:/Users/xyao0/Desktop/project/assembly/transformer/{model_folder}/{seed}/model_{model_ind}.pth'
    model_stat_path = f'C:/Users/xyao0/Desktop/project/assembly/transformer/{model_folder}/{seed}/train_stat.pickle'
    with open(model_stat_path, 'rb') as f:
        train_stat = pickle.load(f)
    my_net = Network(max_len, train_stat, device)
    my_net.load_model(model_path)
    my_net.get_obj_tags(all_objs)

    ### Connect to the robot #######
    FREQUENCY = 1
    dt = 1/ FREQUENCY
    host = "192.168.3.5"
    robot = Robot(host, FREQUENCY)
    # robot.connect()

    ### Connect to the zed camera ###
    destfolder_zed = os.path.join(FOLDER, 'zed')
    zed_cam = Zed(destfolder_zed)
    # zed_cam.connect_camera()
    # zed_cam.capture_video()
    # zed_cam.analyze_video()


    ### Connect to the writst cameras ###
    url_left = 'http://192.168.0.102:8080/shot.jpg'
    url_right = 'http://192.168.0.101:8080/shot.jpg'
    destfolder_wrist = destfolder_zed.replace('zed', 'wrist')
    wrist_cam = Wrist(destfolder_wrist, url_left, url_right)
    # wrist_cam.take_pics()
    # wrist_cam.analyze_image()

    ### Object pose processor ###
    transformation_dir = config['transformation_path']
    pose_processor = PoseProcessor(transformation_dir, destfolder_zed, destfolder_wrist)
    # pose_processor.get_pose_zed()

    action = 'action_0'
    action_summary_file = os.path.join(project_dir, 'data', 'processed', action, 'action_summary.pickle')
    with open(action_summary_file, 'rb') as f:
        action_summary = pickle.load(f)
    median_traj_len = action_summary['median_traj_len']
    vs_start_inds = action_summary['median_wrist_inds']
    action_summary['wrist_end_inds'] = [81, 104, 144]

    my_planner = Planner(my_net, robot, zed_cam, wrist_cam, pose_processor, action_summary, all_objs)
    my_planner.step()
    raise

    action = 'action_0'
    action_summary_file = os.path.join(project_dir, 'data', 'processed', action, 'action_summary.pickle')
    with open(action_summary_file, 'rb') as f:
        action_summary = pickle.load(f)
    median_traj_len = action_summary['median_traj_len']
    vs_start_inds = action_summary['median_wrist_inds']
    vs_end_inds = [81, 104, 144]
    gripper_close_ind = 81
    gripper_open_ind = 144

    current_ind = 0
    for i, ind in enumerate(vs_start_inds):
        # ### Predict from transformer
        # current_tcp_in_base = np.array(robot.get_tcp_pose())
        # print(current_tcp_in_base)
        # current_tcp_in_base[:3] = current_tcp_in_base[:3] * SCALE
        # pose_processor.get_pose_wrist(h5_3d_file_wrist, current_tcp_in_base, destfolder_wrist)
        # current_tcp_in_base_quat = rotvect_to_quat(current_tcp_in_base).flatten()
        # my_net.udpate_obj_sequence(pose_processor.obj_pose_zed, pose_processor.obj_pose_wrist, current_tcp_in_base_quat,
        #                            all_objs)
        # output_seq = my_net.predict_traj(median_traj_len, current_tcp_in_base_quat).detach().numpy()[0]
        # traj = quat_to_rotvect(output_seq[current_ind:ind])
        # traj[:, :3] = (traj[:, :3] * train_stat['std'] + train_stat['mean']) / SCALE
        # robot.servoL(traj)

        ### Take wrist images
        # time.sleep(5)
        # wrist_cam.take_pics()
        # wrist_cam.analyze_image()
        ibvs = IBVS(robot, wrist_cam)
        target_dir = os.path.join(project_dir, 'data', 'reproduce', 'ibvs', action)
        ibvs.run(destfolder_wrist, target_dir, ['nut1'])
        raise



    # colors =  {'bolt': 'green', 'nut':'yellow', 'bin':'black', 'jig':'purple', 'traj':'red', 'trajectory': 'pink'}
    # my_net.plot_traj_and_obj_pos(my_net.output_seq.detach().numpy()[0], my_net.obj_seq.detach().numpy()[0], colors, all_objs)
    # plt.show()
    # print(output_seq.shape)
    # raise
    ### Prodict trajectory using Transformer


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


