import socket
import select
import struct
import time
import os
import numpy as np
import utils
from simulation import vrep
from utils import get_heightmap
from utils import pixel_to_world
class Robot(object):
    def __init__(self, is_sim, obj_mesh_dir, num_obj, workspace_limits,
                 tcp_host_ip, tcp_port, rtc_host_ip, rtc_port,
                 is_testing, test_preset_cases, test_preset_file):

        self.is_sim = is_sim
        self.workspace_limits = workspace_limits
        # self.heightmap_resolution = heightmap_resolution
        # If in simulation...
        if self.is_sim:

            # 定义颜色空间
            self.color_space = np.asarray([[78.0, 121.0, 167.0], # blue
                                           [89.0, 161.0, 79.0], # green
                                           [156, 117, 95], # brown
                                           [242, 142, 43], # orange
                                           [237.0, 201.0, 72.0], # yellow
                                           [186, 176, 172], # gray
                                           [255.0, 87.0, 89.0], # red
                                           [176, 122, 161], # purple
                                           [118, 183, 178], # cyan
                                           [255, 157, 167]])/255.0 #pink

            # 加载对象网格目录下的文件列表
            self.obj_mesh_dir = obj_mesh_dir#存放物体网格模型的路径。
            self.num_obj = num_obj#要添加到场景中的物体数量。
            self.mesh_list = os.listdir(self.obj_mesh_dir)

            # 随机选择物体并分配颜色
            self.obj_mesh_ind = np.random.randint(0, len(self.mesh_list), size=self.num_obj)
            self.obj_mesh_color = self.color_space[np.asarray(range(self.num_obj)) % 10, :]

            # Make sure to have the server side running in V-REP:
            # in a child script of a V-REP scene, add following command
            # to be executed just once, at simulation start:
            #
            # simExtRemoteApiStart(19999)
            #
            # then start simulation, and run this program.
            #
            # IMPORTANT: for each successful call to simxStart, there
            # should be a corresponding call to simxFinish at the end!

            # MODIFY remoteApiConnections.txt

            # Connect to simulator
            vrep.simxFinish(-1) # 关闭所有已有的 V-REP 远程连接。
            #连接本地运行的 V-REP
            self.sim_client = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5) # Connect to V-REP on port 19997
            #如果连接成功，会返回一个客户端句柄（非 -1），否则返回 -1。
            if self.sim_client == -1:
                print('Failed to connect to simulation (V-REP remote API server). Exiting.')
                exit()
            else:
                print('Connected to simulation.')
                self.restart_sim()

            #初始化测试相关参数。
            self.is_testing = is_testing#是否为测试模式
            self.test_preset_cases = test_preset_cases#是否使用预设测试案例
            self.test_preset_file = test_preset_file#测试案例文件路径

            # 获取吸盘句柄
            sim_ret, self.suction_cup_handle = vrep.simxGetObjectHandle(
                self.sim_client, 'BaxterVacuumCup', vrep.simx_opmode_blocking)
            # 获取吸盘状态信号
            sim_ret, self.suction_signal_name = vrep.simxGetStringSignal(
                self.sim_client, 'suctionActive', vrep.simx_opmode_blocking)

            res, self.dummy_link1_handle = vrep.simxGetObjectHandle(self.sim_client, 'loopClosureDummy1',
                                                                    vrep.simx_opmode_blocking)
            res, self.suction_base_handle = vrep.simxGetObjectHandle(self.sim_client, 'BaxterVacuumCup',
                                                                     vrep.simx_opmode_blocking)

            #  设置仿真中的虚拟相机。
            self.setup_sim_camera()

            # 判断是否为测试模式且使用预设案例。
            if self.is_testing and self.test_preset_cases:
                #打开并读取预设测试案例文件。
                file = open(self.test_preset_file, 'r')#测试案例文件路径
                file_content = file.readlines()# 按行读取的文件内容，每一行对应一个物体的信息
                self.test_obj_mesh_files = []#物体的网格文件路径
                self.test_obj_mesh_colors = []#物体的颜色
                self.test_obj_positions = []#物体的位置坐标 (x, y, z)
                self.test_obj_orientations = []#物体的姿态角度 (rx, ry, rz)（欧拉角）
                #遍历每个物体，将每行内容按空格分割成字段。
                for object_idx in range(self.num_obj):
                    file_content_curr_object = file_content[object_idx].split()
                    #将网格文件名与基础路径拼接，形成完整路径。
                    self.test_obj_mesh_files.append(os.path.join(self.obj_mesh_dir,file_content_curr_object[0]))
                    # 存储物体颜色 [R, G, B]，从字符串转为浮点型列表。
                    self.test_obj_mesh_colors.append([float(file_content_curr_object[1]),float(file_content_curr_object[2]),float(file_content_curr_object[3])])
                    #存储物体的初始位置 (x, y, z)。
                    self.test_obj_positions.append([float(file_content_curr_object[4]),float(file_content_curr_object[5]),float(file_content_curr_object[6])])
                    #存储物体的初始姿态 (rx, ry, rz)（单位为弧度或其它形式，取决于仿真设置）
                    self.test_obj_orientations.append([float(file_content_curr_object[7]),float(file_content_curr_object[8]),float(file_content_curr_object[9])])
                file.close()
                self.obj_mesh_color = np.asarray(self.test_obj_mesh_colors)

            # Add objects to simulation environment
            self.add_objects()
            self.initialize_stacked_blocks()

        # If in real-settings...
        else:

            # 设置 TCP 客户端连接参数。
            self.tcp_host_ip = tcp_host_ip#UR 机械臂控制器的 IP 地址（如 "192.168.1.100"）
            self.tcp_port = tcp_port#UR 机械臂的 TCP 端口（通常为 30002）
            # self.tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # 设置实时客户端（RTC）连接参数，收机器人状态数据（如力传感器、TCP 位姿等）。
            self.rtc_host_ip = rtc_host_ip
            self.rtc_port = rtc_port

            # 设置默认的“Home”位置关节角度（单位为弧度）。确保机械臂回到一个已知且安全的位置。
            # self.home_joint_config = [-np.pi, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]
            self.home_joint_config = [-(180.0/360.0)*2*np.pi, -(84.2/360.0)*2*np.pi, (112.8/360.0)*2*np.pi, -(119.7/360.0)*2*np.pi, -(90.0/360.0)*2*np.pi, 0.0]

            # 设置关节运动的最大加速度和速度。
            self.joint_acc = 8 # Safe: 1.4
            self.joint_vel = 3 # Safe: 1.05

            # 关节到位容差。当前实际关节角度与目标角度差异小于该值时认为到达目标。
            self.joint_tolerance = 0.01

            # 工具坐标系下的最大加速度和速度。
            self.tool_acc = 1.2 # Safe: 0.5
            self.tool_vel = 0.25 # Safe: 0.2

            # 工具位姿到位容差，前三个元素 [x, y, z] 是位置误差容忍范围（米），后三个元素 [rx, ry, rz] 是姿态误差容忍范围（弧度）。
            self.tool_pose_tolerance = [0.002,0.002,0.002,0.01,0.01,0.01]

            # Move robot to home pose
            self.go_home()

            # 导入并初始化 RealSense 相机模块。
            from real.camera import Camera
            #获取相机内参矩阵 intrinsics，用于后续图像处理
            self.camera = Camera()
            self.cam_intrinsics = self.camera.intrinsics

            # 加载相机外参（相对于机器人基座的位姿），加载深度图缩放因子（用于将原始深度值转为米）
            self.cam_pose = np.loadtxt('real/camera_pose.txt', delimiter=' ')
            self.cam_depth_scale = np.loadtxt('real/camera_depth_scale.txt', delimiter=' ')
            #####
            self.tcp_socket = None
            ######################

        #####################
    def setup_sim_camera(self):

        # Get handle to camera
        sim_ret, self.cam_handle = vrep.simxGetObjectHandle(self.sim_client, 'Vision_sensor_persp', vrep.simx_opmode_blocking)

        # Get camera pose and intrinsics in simulation
        sim_ret, cam_position = vrep.simxGetObjectPosition(self.sim_client, self.cam_handle, -1, vrep.simx_opmode_blocking)
        sim_ret, cam_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.cam_handle, -1, vrep.simx_opmode_blocking)
        cam_trans = np.eye(4,4)
        cam_trans[0:3,3] = np.asarray(cam_position)
        cam_orientation = [-cam_orientation[0], -cam_orientation[1], -cam_orientation[2]]
        cam_rotm = np.eye(4,4)
        cam_rotm[0:3,0:3] = np.linalg.inv(utils.euler2rotm(cam_orientation))
        self.cam_pose = np.dot(cam_trans, cam_rotm) # Compute rigid transformation representating camera pose
        self.cam_intrinsics = np.asarray([[618.62, 0, 320], [0, 618.62, 240], [0, 0, 1]])
        self.cam_depth_scale = 1

        # Get background image
        self.bg_color_img, self.bg_depth_img = self.get_camera_data()
        self.bg_depth_img = self.bg_depth_img * self.cam_depth_scale

    def add_objects(self):
        # Add each object to robot workspace at x,y location and orientation (random or pre-loaded)
        self.object_handles = []
        sim_obj_handles = []
        for object_idx in range(len(self.obj_mesh_ind)):
            curr_mesh_file = os.path.join(self.obj_mesh_dir, self.mesh_list[self.obj_mesh_ind[object_idx]])
            if self.is_testing and self.test_preset_cases:
                curr_mesh_file = self.test_obj_mesh_files[object_idx]

            curr_shape_name = 'shape_%02d' % object_idx

            # 固定堆叠位置
            stack_center_x = (self.workspace_limits[0][0] + self.workspace_limits[0][1]) / 2
            stack_center_y = (self.workspace_limits[1][0] + self.workspace_limits[1][1]) / 2
            block_height = 0.05
            object_z = 0.05 + object_idx * block_height  # 初始高度 + 层数 × 高度
            object_position = [stack_center_x, stack_center_y, object_z]

            # 固定朝向
            #object_orientation = [0, 0, np.random.uniform(0, 2 * np.pi)]  # 仅绕Z轴旋转
            object_orientation = [0, np.random.uniform(0, 2 * np.pi),  0] #y

            if self.is_testing and self.test_preset_cases:
                object_position = [self.test_obj_positions[object_idx][0],
                                   self.test_obj_positions[object_idx][1],
                                   self.test_obj_positions[object_idx][2]]
                object_orientation = [self.test_obj_orientations[object_idx][0],
                                      self.test_obj_orientations[object_idx][1],
                                      self.test_obj_orientations[object_idx][2]]
            object_color = [self.obj_mesh_color[object_idx][0], self.obj_mesh_color[object_idx][1],
                            self.obj_mesh_color[object_idx][2]]
            print(
                f"Object {object_idx}: Position {object_position}, Orientation {object_orientation}")  # 添加调试信息
            ret_resp, ret_ints, ret_floats, ret_strings, ret_buffer = vrep.simxCallScriptFunction(
                self.sim_client,
                'remoteApiCommandServer',
                vrep.sim_scripttype_childscript,
                'importShape',
                [0, 0, 255, 0],
                object_position + object_orientation + object_color,
                [curr_mesh_file, curr_shape_name],
                bytearray(),
                vrep.simx_opmode_blocking
            )
            #设置物体初始角速度为0
            #vrep.simxSetObjectAngularVelocity(self.sim_client, curr_shape_handle, [0, 0, 0], vrep.simx_opmode_blocking)

            if ret_resp == 8:
                print('Failed to add new objects to simulation. Please restart.')
                exit()
            curr_shape_handle = ret_ints[0]
            self.object_handles.append(curr_shape_handle)
            if not (self.is_testing and self.test_preset_cases):
                time.sleep(2)
        self.prev_obj_positions = []
        self.obj_positions = []



    def initialize_stacked_blocks(self):
        """强制在初始化时堆叠所有物体"""
        for object_handle in self.object_handles:
            object_idx = self.object_handles.index(object_handle)
            stack_center_x = (self.workspace_limits[0][0] + self.workspace_limits[0][1]) / 2
            stack_center_y = (self.workspace_limits[1][0] + self.workspace_limits[1][1]) / 2
            block_height = 0.05
            object_z = 0.05 + object_idx * block_height
            object_position = [stack_center_x, stack_center_y, object_z]
            vrep.simxSetObjectPosition(self.sim_client, object_handle, -1, object_position, vrep.simx_opmode_blocking)
            time.sleep(0.5)

    def restart_sim(self):
        sim_ret, self.UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_target',
                                                                   vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (-0.5, 0, 0.3),
                                   vrep.simx_opmode_blocking)

        vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
        vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
        time.sleep(1)

        sim_ret, self.suction_tip_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_tip',
                                                                    vrep.simx_opmode_blocking)
        sim_ret, tip_position = vrep.simxGetObjectPosition(self.sim_client, self.suction_tip_handle, -1,
                                                           vrep.simx_opmode_blocking)
        print("Tip position: ", tip_position)
        while tip_position[2] > 0.4:  # 防止 V-REP 异常未重置
            vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
            vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
            time.sleep(1)
            sim_ret, tip_position = vrep.simxGetObjectPosition(self.sim_client, self.suction_tip_handle, -1,
                                                               vrep.simx_opmode_blocking)

    def check_sim(self):
        # 检查仿真是否稳定：吸盘末端是否仍在工作空间内
        sim_ret, tip_position = vrep.simxGetObjectPosition(self.sim_client, self.suction_tip_handle, -1,
                                                           vrep.simx_opmode_blocking)

        sim_ok = (
                self.workspace_limits[0][0] - 0.1 < tip_position[0] < self.workspace_limits[0][1] + 0.1 and
                self.workspace_limits[1][0] - 0.1 < tip_position[1] < self.workspace_limits[1][1] + 0.1 and
                self.workspace_limits[2][0] < tip_position[2] < self.workspace_limits[2][1]
        )

        if not sim_ok:
            print('Simulation unstable. Restarting environment.')
            self.restart_sim()
            self.add_objects()

    def get_task_score(self):

        key_positions = np.asarray([[-0.625, 0.125, 0.0], # red
                                    [-0.625, -0.125, 0.0], # blue
                                    [-0.375, 0.125, 0.0], # green
                                    [-0.375, -0.125, 0.0]]) #yellow

        obj_positions = np.asarray(self.get_obj_positions())
        obj_positions.shape = (1, obj_positions.shape[0], obj_positions.shape[1])
        obj_positions = np.tile(obj_positions, (key_positions.shape[0], 1, 1))

        key_positions.shape = (key_positions.shape[0], 1, key_positions.shape[1])
        key_positions = np.tile(key_positions, (1 ,obj_positions.shape[1] ,1))

        key_dist = np.sqrt(np.sum(np.power(obj_positions - key_positions, 2), axis=2))
        key_nn_idx = np.argmin(key_dist, axis=0)

        return np.sum(key_nn_idx == np.asarray(range(self.num_obj)) % 4)


    def check_goal_reached(self):

        goal_reached = self.get_task_score() == self.num_obj
        return goal_reached


    def get_obj_positions(self):

        obj_positions = []
        for object_handle in self.object_handles:
            sim_ret, object_position = vrep.simxGetObjectPosition(self.sim_client, object_handle, -1, vrep.simx_opmode_blocking)
            obj_positions.append(object_position)

        return obj_positions

    def get_obj_positions_and_orientations(self):

        obj_positions = []
        obj_orientations = []
        for object_handle in self.object_handles:
            sim_ret, object_position = vrep.simxGetObjectPosition(self.sim_client, object_handle, -1, vrep.simx_opmode_blocking)
            sim_ret, object_orientation = vrep.simxGetObjectOrientation(self.sim_client, object_handle, -1, vrep.simx_opmode_blocking)
            obj_positions.append(object_position)
            obj_orientations.append(object_orientation)

        return obj_positions, obj_orientations

#物体随机放置在机器人工作区域
    def reposition_objects(self, workspace_limits):#将物体随机放置在工作区中

        # Move gripper out of the way
        self.move_to([-0.1, 0, 0.3], None)
        # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
        # vrep.simxSetObjectPosition(self.sim_client, UR5_target_handle, -1, (-0.5,0,0.3), vrep.simx_opmode_blocking)
        # time.sleep(1)

        for object_handle in self.object_handles:

            # Drop object at random x,y location and random orientation in robot workspace
            drop_x = (workspace_limits[0][1] - workspace_limits[0][0] - 0.2) * np.random.random_sample() + workspace_limits[0][0] + 0.1
            drop_y = (workspace_limits[1][1] - workspace_limits[1][0] - 0.2) * np.random.random_sample() + workspace_limits[1][0] + 0.1
            object_position = [drop_x, drop_y, 0.15]
            object_orientation = [2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample()]
            vrep.simxSetObjectPosition(self.sim_client, object_handle, -1, object_position, vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, object_handle, -1, object_orientation, vrep.simx_opmode_blocking)
            time.sleep(2)


    def get_camera_data(self):

        if self.is_sim:

            # Get color image from simulation
            sim_ret, resolution, raw_image = vrep.simxGetVisionSensorImage(self.sim_client, self.cam_handle, 0, vrep.simx_opmode_blocking)
            color_img = np.asarray(raw_image)
            color_img.shape = (resolution[1], resolution[0], 3)
            color_img = color_img.astype(float)/255
            color_img[color_img < 0] += 1
            color_img *= 255
            color_img = np.fliplr(color_img)
            color_img = color_img.astype(np.uint8)

            # Get depth image from simulation
            sim_ret, resolution, depth_buffer = vrep.simxGetVisionSensorDepthBuffer(self.sim_client, self.cam_handle, vrep.simx_opmode_blocking)
            depth_img = np.asarray(depth_buffer)
            depth_img.shape = (resolution[1], resolution[0])
            depth_img = np.fliplr(depth_img)
            zNear = 0.01
            zFar = 10
            depth_img = depth_img * (zFar - zNear) + zNear

        else:
            # Get color and depth image from ROS service
            color_img, depth_img = self.camera.get_data()
            # color_img = self.camera.color_data.copy()
            # depth_img = self.camera.depth_data.copy()

        return color_img, depth_img


    def parse_tcp_state_data(self, state_data, subpackage):

        # Read package header
        data_bytes = bytearray()
        data_bytes.extend(state_data)
        data_length = struct.unpack("!i", data_bytes[0:4])[0];
        robot_message_type = data_bytes[4]
        assert(robot_message_type == 16)
        byte_idx = 5

        # Parse sub-packages
        subpackage_types = {'joint_data' : 1, 'cartesian_info' : 4, 'force_mode_data' : 7, 'tool_data' : 2}
        while byte_idx < data_length:
            # package_length = int.from_bytes(data_bytes[byte_idx:(byte_idx+4)], byteorder='big', signed=False)
            package_length = struct.unpack("!i", data_bytes[byte_idx:(byte_idx+4)])[0]
            byte_idx += 4
            package_idx = data_bytes[byte_idx]
            if package_idx == subpackage_types[subpackage]:
                byte_idx += 1
                break
            byte_idx += package_length - 4

        def parse_joint_data(data_bytes, byte_idx):
            actual_joint_positions = [0,0,0,0,0,0]
            target_joint_positions = [0,0,0,0,0,0]
            for joint_idx in range(6):
                actual_joint_positions[joint_idx] = struct.unpack('!d', data_bytes[(byte_idx+0):(byte_idx+8)])[0]
                target_joint_positions[joint_idx] = struct.unpack('!d', data_bytes[(byte_idx+8):(byte_idx+16)])[0]
                byte_idx += 41
            return actual_joint_positions

        def parse_cartesian_info(data_bytes, byte_idx):
            actual_tool_pose = [0,0,0,0,0,0]
            for pose_value_idx in range(6):
                actual_tool_pose[pose_value_idx] = struct.unpack('!d', data_bytes[(byte_idx+0):(byte_idx+8)])[0]
                byte_idx += 8
            return actual_tool_pose

        def parse_tool_data(data_bytes, byte_idx):
            byte_idx += 2
            tool_analog_input2 = struct.unpack('!d', data_bytes[(byte_idx+0):(byte_idx+8)])[0]
            return tool_analog_input2

        parse_functions = {'joint_data' : parse_joint_data, 'cartesian_info' : parse_cartesian_info, 'tool_data' : parse_tool_data}
        return parse_functions[subpackage](data_bytes, byte_idx)

    def parse_rtc_state_data(self, state_data):

        # Read package header
        data_bytes = bytearray()
        data_bytes.extend(state_data)
        data_length = struct.unpack("!i", data_bytes[0:4])[0];
        assert(data_length == 812)
        byte_idx = 4 + 8 + 8*48 + 24 + 120
        TCP_forces = [0,0,0,0,0,0]
        for joint_idx in range(6):
            TCP_forces[joint_idx] = struct.unpack('!d', data_bytes[(byte_idx+0):(byte_idx+8)])[0]
            byte_idx += 8

        return TCP_forces

    def get_state(self):

        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        state_data = self.tcp_socket.recv(2048)
        self.tcp_socket.close()
        return state_data


    def move_to(self, tool_position, tool_orientation):#机械臂末端移动到指定位置和姿态
    #给定目标点 方向
        if self.is_sim:

            # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
            sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle,-1,vrep.simx_opmode_blocking)

            move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1], tool_position[2] - UR5_target_position[2]])
            move_magnitude = np.linalg.norm(move_direction)
            move_step = 0.02*move_direction/move_magnitude
            num_move_steps = int(np.floor(move_magnitude/0.02))

            for step_iter in range(num_move_steps):
                vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(UR5_target_position[0] + move_step[0], UR5_target_position[1] + move_step[1], UR5_target_position[2] + move_step[2]),vrep.simx_opmode_blocking)
                sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client,self.UR5_target_handle,-1,vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(tool_position[0],tool_position[1],tool_position[2]),vrep.simx_opmode_blocking)

        else:

            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
            tcp_command = "movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0)\n" % (tool_position[0],tool_position[1],tool_position[2],tool_orientation[0],tool_orientation[1],tool_orientation[2],self.tool_acc,self.tool_vel)
            self.tcp_socket.send(str.encode(tcp_command))

            # Block until robot reaches target tool position
            tcp_state_data = self.tcp_socket.recv(2048)
            actual_tool_pose = self.parse_tcp_state_data(tcp_state_data, 'cartesian_info')
            while not all([np.abs(actual_tool_pose[j] - tool_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
                # [min(np.abs(actual_tool_pose[j] - tool_orientation[j-3]), np.abs(np.abs(actual_tool_pose[j] - tool_orientation[j-3]) - np.pi*2)) < self.tool_pose_tolerance[j] for j in range(3,6)]
                # print([np.abs(actual_tool_pose[j] - tool_position[j]) for j in range(3)] + [min(np.abs(actual_tool_pose[j] - tool_orientation[j-3]), np.abs(np.abs(actual_tool_pose[j] - tool_orientation[j-3]) - np.pi*2)) for j in range(3,6)])
                tcp_state_data = self.tcp_socket.recv(2048)
                prev_actual_tool_pose = np.asarray(actual_tool_pose).copy()
                actual_tool_pose = self.parse_tcp_state_data(tcp_state_data, 'cartesian_info')
                time.sleep(0.01)
            self.tcp_socket.close()
#制机器人工具以 1cm 为步长逐步移动到目标位置
    def guarded_move_to(self, tool_position, tool_orientation):

        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.rtc_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        self.rtc_socket.connect((self.rtc_host_ip, self.rtc_port))

        # Read actual tool position
        tcp_state_data = self.tcp_socket.recv(2048)
        actual_tool_pose = self.parse_tcp_state_data(tcp_state_data, 'cartesian_info')
        execute_success = True

        # Increment every cm, check force
        self.tool_acc = 0.1 # 1.2 # 0.5

        while not all([np.abs(actual_tool_pose[j] - tool_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
            # [min(np.abs(actual_tool_pose[j] - tool_orientation[j-3]), np.abs(np.abs(actual_tool_pose[j] - tool_orientation[j-3]) - np.pi*2)) < self.tool_pose_tolerance[j] for j in range(3,6)]

            # Compute motion trajectory in 1cm increments
            increment = np.asarray([(tool_position[j] - actual_tool_pose[j]) for j in range(3)])
            if np.linalg.norm(increment) < 0.01:
                increment_position = tool_position
            else:
                increment = 0.01*increment/np.linalg.norm(increment)
                increment_position = np.asarray(actual_tool_pose[0:3]) + increment

            # Move to next increment position (blocking call)
            tcp_command = "movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0)\n" % (increment_position[0],increment_position[1],increment_position[2],tool_orientation[0],tool_orientation[1],tool_orientation[2],self.tool_acc,self.tool_vel)
            self.tcp_socket.send(str.encode(tcp_command))

            time_start = time.time()
            tcp_state_data = self.tcp_socket.recv(2048)
            actual_tool_pose = self.parse_tcp_state_data(tcp_state_data, 'cartesian_info')
            while not all([np.abs(actual_tool_pose[j] - increment_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
                # print([np.abs(actual_tool_pose[j] - increment_position[j]) for j in range(3)])
                tcp_state_data = self.tcp_socket.recv(2048)
                actual_tool_pose = self.parse_tcp_state_data(tcp_state_data, 'cartesian_info')
                time_snapshot = time.time()
                if time_snapshot - time_start > 1:
                    break
                time.sleep(0.01)

            # Reading TCP forces from real-time client connection
            rtc_state_data = self.rtc_socket.recv(6496)
            TCP_forces = self.parse_rtc_state_data(rtc_state_data)

            # If TCP forces in x/y exceed 20 Newtons, stop moving
            # print(TCP_forces[0:3])
            if np.linalg.norm(np.asarray(TCP_forces[0:2])) > 20 or (time_snapshot - time_start) > 1:
                print('Warning: contact detected! Movement halted. TCP forces: [%f, %f, %f]' % (TCP_forces[0], TCP_forces[1], TCP_forces[2]))
                execute_success = False
                break

            time.sleep(0.01)

        self.tool_acc = 1.2 # 1.2 # 0.5

        self.tcp_socket.close()
        self.rtc_socket.close()

        return execute_success


    def move_joints(self, joint_configuration):

        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        tcp_command = "movej([%f" % joint_configuration[0]
        for joint_idx in range(1,6):
            tcp_command = tcp_command + (",%f" % joint_configuration[joint_idx])
        tcp_command = tcp_command + "],a=%f,v=%f)\n" % (self.joint_acc, self.joint_vel)
        self.tcp_socket.send(str.encode(tcp_command))

        # Block until robot reaches home state
        state_data = self.tcp_socket.recv(2048)
        actual_joint_positions = self.parse_tcp_state_data(state_data, 'joint_data')
        while not all([np.abs(actual_joint_positions[j] - joint_configuration[j]) < self.joint_tolerance for j in range(6)]):
            state_data = self.tcp_socket.recv(2048)
            actual_joint_positions = self.parse_tcp_state_data(state_data, 'joint_data')
            time.sleep(0.01)

        self.tcp_socket.close()


    def go_home(self):

        self.move_joints(self.home_joint_config)

    #######################
    def check_suction(self):

        state_data = self.get_state()
        tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
        return tool_analog_input2 > 0.26
    # Primitives ----------------------------------------------------------


    def suction(self, position, heightmap_rotation_angle, workspace_limits):
        print('Executing: suction at (%f, %f, %f)' % (position[0], position[1], position[2]))

        if self.is_sim:
            #计算吸盘工具旋转角度（绕Y轴）
            tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi / 2
            # 修正目标位置的Z高度，确保不超出最低限
            position = np.asarray(position).copy()
            position[2] = max(position[2] - 0.04, workspace_limits[2][0] + 0.01)
            # 生成吸取前的靠近位置
            approach_position = (position[0], position[1], position[2] + 0.1)
            tool_position=approach_position
            # 获取当前末端位置
            sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
            # 生成插值移动路径（位置）
            move_direction = np.asarray(
                [tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1],
                 tool_position[2] - UR5_target_position[2]])
            move_magnitude = np.linalg.norm(move_direction)
            move_step = 0.05 * move_direction / move_magnitude
            num_move_steps = int(np.floor(move_direction[0] / move_step[0]))

            # 获取当前旋转角度，并生成旋转插值（绕Y轴）
            sim_ret, suction_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
            rotation_step = 0.3 if (tool_rotation_angle - suction_orientation[1] > 0) else -0.3
            num_rotation_steps = int(np.floor((tool_rotation_angle - suction_orientation[1]) / rotation_step))
            # 插值执行移动和旋转
            for step_iter in range(max(num_move_steps, num_rotation_steps)):
                vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (
                UR5_target_position[0] + move_step[0] * min(step_iter, num_move_steps),
                UR5_target_position[1] + move_step[1] * min(step_iter, num_move_steps),
                UR5_target_position[2] + move_step[2] * min(step_iter, num_move_steps)), vrep.simx_opmode_blocking)

                vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (
                np.pi / 2, suction_orientation[1] + rotation_step * min(step_iter, num_rotation_steps), np.pi / 2),
                                              vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1,
                                       (tool_position[0], tool_position[1], tool_position[2]),
                                       vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1,
                                          (np.pi / 2, tool_rotation_angle, np.pi / 2), vrep.simx_opmode_blocking)


            # 激活吸盘
            self.activate_suction()
            vrep.simxSetIntegerSignal(self.sim_client, 'suctionActive', 1, vrep.simx_opmode_blocking)
            # 移动
            self.move_to(position, None)

            # 抬起吸盘
            self.move_to(approach_position, None)

            print("suction target :",position)
            # 检查是否吸附成功（通过 dummy link 判断）
            suction_success = self.is_object_attached()

            if suction_success:
                object_positions = np.asarray(self.get_obj_positions())
                object_positions = object_positions[:, 2]
                suctioned_object_ind = np.argmax(object_positions)
                suctioned_object_handle = self.object_handles[suctioned_object_ind]
                vrep.simxSetObjectPosition(self.sim_client, suctioned_object_handle, -1,
                                           (-0.5, 0.5 + 0.05 * float(suctioned_object_ind), 0.1),
                                           vrep.simx_opmode_blocking)

            # 停吸
            vrep.simxSetIntegerSignal(self.sim_client, 'suctionActive', 0, vrep.simx_opmode_blocking)

            return suction_success

        else:
            raise NotImplementedError("Suction on real robot is not implemented yet.")

#获取机器人工作空间的彩色高度图和深度高度图
    # def get_heightmaps(self):
    #     color_img, depth_img = self.get_camera_data()
    #     color_heightmap, depth_heightmap = get_heightmap(
    #         color_img, depth_img,
    #         self.cam_intrinsics,
    #         self.cam_pose,
    #         self.workspace_limits,
    #         self.heightmap_resolution
    #     )
    #     return color_heightmap, depth_heightmap

    def execute_learned_suction(self, trainer, cam_intrinsics, cam_pose, workspace_limits):
        #获取当前相机视角下的 RGB-D 图像，并将其投影为高度图
        color_heightmap, depth_heightmap = self.get_heightmaps()
        suction_predictions, _ = trainer.forward(color_heightmap, depth_heightmap)
        #找到吸取预测中置信度最大的位置，即当前最有希望成功的位置
        best_pix_ind = np.unravel_index(np.argmax(suction_predictions), suction_predictions.shape)
        rotate_idx, pixel_y, pixel_x = best_pix_ind
        #将角度索引转换为实际旋转角度
        rotation_angle = (rotate_idx / 16.0) * 2 * np.pi
        #将像素坐标和深度图转换为 3D 空间中实际的吸取位置（X, Y, Z）
        suction_position = pixel_to_world(
            pixel_x, pixel_y, depth_heightmap,
            cam_intrinsics, cam_pose,
            self.heightmap_resolution, self.workspace_limits
        )
        self.verify_suction_position(suction_position)
        #控制机械臂移动到指定位置，调整末端方向，然后执行吸盘吸附。
        suction_success = self.suction(suction_position, rotation_angle, workspace_limits)
        return suction_success

    def verify_suction_position(self, target_position):
        """调试函数：打印目标吸盘位置与吸盘末端实际位置的偏差"""
        _, tip_position = vrep.simxGetObjectPosition(self.sim_client, self.suction_tip_handle, -1,
                                                     vrep.simx_opmode_blocking)
        print("[DEBUG] Suction target position:", target_position)
        print("[DEBUG] Suction tip actual position:", tip_position)
        offset = np.linalg.norm(np.array(tip_position) - np.array(target_position))
        print("[DEBUG] Tip vs Target offset (m):", offset)

    def activate_suction(self):
        """激活吸盘"""
        if self.is_sim:
            # 调用V-REP脚本激活吸盘
            vrep.simxCallScriptFunction(
                self.sim_client, 'remoteApiCommandServer',
                vrep.sim_scripttype_childscript,
                'activateSuction', [self.suction_cup_handle], [], [], bytearray(),
                vrep.simx_opmode_blocking)
        else:
            # 真实机器人的吸盘控制（根据实际硬件修改）
            self.send_command("SUCTION_ON")

        self.suction_active = True
        time.sleep(0.5)  # 等待吸盘激活

    def deactivate_suction(self):
        """释放吸盘"""
        if self.is_sim:
            # 调用V-REP脚本释放吸盘
            vrep.simxCallScriptFunction(
                self.sim_client, 'remoteApiCommandServer',
                vrep.sim_scripttype_childscript,
                'deactivateSuction', [self.suction_cup_handle], [], [], bytearray(),
                vrep.simx_opmode_blocking)
        else:
            # 真实机器人的吸盘控制
            self.send_command("SUCTION_OFF")

        self.suction_active = False
        time.sleep(0.5)  # 等待吸盘释放

    def is_object_attached(self):
        # 检查 dummy_link1 的 parent 是不是 suction_base，如果不是说明吸住了物体
        code, parent = vrep.simxGetObjectParent(self.sim_client, self.dummy_link1_handle, vrep.simx_opmode_blocking)
        print("[DEBUG] dummy parent handle:", parent)
        if code != 0:
            print("Failed to get parent of dummy_link1")
            return False
        return parent != self.suction_base_handle

    def send_command(self, command):
        if not hasattr(self, 'tcp_socket'):
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        self.tcp_socket.send(str.encode(f"{command}\n"))

    ########################
    def restart_real(self):

        # Compute tool orientation from heightmap rotation angle
        grasp_orientation = [1.0,0.0]
        tool_rotation_angle = -np.pi/4
        tool_orientation = np.asarray([grasp_orientation[0]*np.cos(tool_rotation_angle) - grasp_orientation[1]*np.sin(tool_rotation_angle), grasp_orientation[0]*np.sin(tool_rotation_angle) + grasp_orientation[1]*np.cos(tool_rotation_angle), 0.0])*np.pi
        tool_orientation_angle = np.linalg.norm(tool_orientation)
        tool_orientation_axis = tool_orientation/tool_orientation_angle
        tool_orientation_rotm = utils.angle2rotm(tool_orientation_angle, tool_orientation_axis, point=None)[:3,:3]

        tilt_rotm = utils.euler2rotm(np.asarray([-np.pi/4,0,0]))
        tilted_tool_orientation_rotm = np.dot(tilt_rotm, tool_orientation_rotm)
        tilted_tool_orientation_axis_angle = utils.rotm2angle(tilted_tool_orientation_rotm)
        tilted_tool_orientation = tilted_tool_orientation_axis_angle[0]*np.asarray(tilted_tool_orientation_axis_angle[1:4])

        # Move to box grabbing position
        box_grab_position = [0.5,-0.35,-0.12]
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        tcp_command = "def process():\n"
        tcp_command += " set_digital_out(8,False)\n"
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.09)\n" % (box_grab_position[0],box_grab_position[1],box_grab_position[2]+0.1,tilted_tool_orientation[0],tilted_tool_orientation[1],tilted_tool_orientation[2],self.joint_acc,self.joint_vel)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_grab_position[0],box_grab_position[1],box_grab_position[2],tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc,self.joint_vel)
        tcp_command += " set_digital_out(8,True)\n"
        tcp_command += "end\n"
        self.tcp_socket.send(str.encode(tcp_command))
        self.tcp_socket.close()

        # Block until robot reaches box grabbing position and gripper fingers have stopped moving
        state_data = self.get_state()
        tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
        while True:
            state_data = self.get_state()
            new_tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
            actual_tool_pose = self.parse_tcp_state_data(state_data, 'cartesian_info')
            if tool_analog_input2 < 3.7 and (abs(new_tool_analog_input2 - tool_analog_input2) < 0.01) and all([np.abs(actual_tool_pose[j] - box_grab_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
                break
            tool_analog_input2 = new_tool_analog_input2

        # Move to box release position
        box_release_position = [0.5,0.08,-0.12]
        home_position = [0.49,0.11,0.03]
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        tcp_command = "def process():\n"
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_release_position[0],box_release_position[1],box_release_position[2],tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc*0.1,self.joint_vel*0.1)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_release_position[0],box_release_position[1],box_release_position[2]+0.3,tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc*0.02,self.joint_vel*0.02)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.29)\n" % (box_grab_position[0]-0.05,box_grab_position[1]+0.1,box_grab_position[2]+0.3,tilted_tool_orientation[0],tilted_tool_orientation[1],tilted_tool_orientation[2],self.joint_acc*0.5,self.joint_vel*0.5)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_grab_position[0]-0.05,box_grab_position[1]+0.1,box_grab_position[2],tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc*0.5,self.joint_vel*0.5)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_grab_position[0],box_grab_position[1],box_grab_position[2],tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc*0.1,self.joint_vel*0.1)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_grab_position[0]+0.05,box_grab_position[1],box_grab_position[2],tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc*0.1,self.joint_vel*0.1)
        tcp_command += " set_digital_out(8,False)\n"
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.09)\n" % (box_grab_position[0],box_grab_position[1],box_grab_position[2]+0.1,tilted_tool_orientation[0],tilted_tool_orientation[1],tilted_tool_orientation[2],self.joint_acc,self.joint_vel)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (home_position[0],home_position[1],home_position[2],tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc,self.joint_vel)
        tcp_command += "end\n"
        self.tcp_socket.send(str.encode(tcp_command))
        self.tcp_socket.close()

        # Block until robot reaches home position
        state_data = self.get_state()
        tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
        while True:
            state_data = self.get_state()
            new_tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
            actual_tool_pose = self.parse_tcp_state_data(state_data, 'cartesian_info')
            if tool_analog_input2 > 3.0 and (abs(new_tool_analog_input2 - tool_analog_input2) < 0.01) and all([np.abs(actual_tool_pose[j] - home_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
                break
            tool_analog_input2 = new_tool_analog_input2


# #将被抓取的物体放置到指定位置。
#     def place(self, position, orientation, workspace_limits):
#         print('Executing: place at (%f, %f, %f)' % (position[0], position[1], position[2]))
#
#         # 确保 Z 值不低于工作空间下限
#         position[2] = max(position[2], workspace_limits[2][0])
#         #移动到目标点上方 20cm
#         self.move_to([position[0], position[1], position[2] + 0.2], orientation)
#         # 继续下降到目标点上方 5cm
#         self.move_to([position[0], position[1], position[2] + 0.05], orientation)
#         #设置低速缓慢下降
#         self.tool_acc = 1 # 0.05
#         self.tool_vel = 0.02 # 0.02
#         #移动到最终放置位置
#         self.move_to([position[0], position[1], position[2]], orientation)
#         #打开夹爪释放物体
#         self.open_gripper()
#         self.tool_acc = 1 # 0.5
#         self.tool_vel = 0.2 # 0.2
#         #抬高机械臂
#         self.move_to([position[0], position[1], position[2] + 0.2], orientation)
#         self.close_gripper()
#         self.go_home()

    # def place(self, position, heightmap_rotation_angle, workspace_limits):
    #     print('Executing: place at (%f, %f, %f)' % (position[0], position[1], position[2]))
    #
    #     if self.is_sim:
    #
    #         # Compute tool orientation from heightmap rotation angle
    #         tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi/2
    #
    #         # Avoid collision with floor
    #         position[2] = max(position[2] + 0.04 + 0.02, workspace_limits[2][0] + 0.02)
    #
    #         # Move gripper to location above place target
    #         place_location_margin = 0.1
    #         sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
    #         location_above_place_target = (position[0], position[1], position[2] + place_location_margin)
    #         self.move_to(location_above_place_target, None)
    #
    #         sim_ret,gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, UR5_target_handle, -1, vrep.simx_opmode_blocking)
    #         if tool_rotation_angle - gripper_orientation[1] > 0:
    #             increment = 0.2
    #         else:
    #             increment = -0.2
    #         while abs(tool_rotation_angle - gripper_orientation[1]) >= 0.2:
    #             vrep.simxSetObjectOrientation(self.sim_client, UR5_target_handle, -1, (np.pi/2, gripper_orientation[1] + increment, np.pi/2), vrep.simx_opmode_blocking)
    #             time.sleep(0.01)
    #             sim_ret,gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, UR5_target_handle, -1, vrep.simx_opmode_blocking)
    #         vrep.simxSetObjectOrientation(self.sim_client, UR5_target_handle, -1, (np.pi/2, tool_rotation_angle, np.pi/2), vrep.simx_opmode_blocking)
    #
    #         # Approach place target
    #         self.move_to(position, None)
    #
    #         # Ensure gripper is open
    #         self.open_gripper()
    #
    #         # Move gripper to location above place target
    #         self.move_to(location_above_place_target, None)
    #
    #         place_success = True
    #         return place_success
    #
    #


























# JUNK

# command = "movel(p[%f,%f,%f,%f,%f,%f],0.5,0.2,0,0,a=1.2,v=0.25)\n" % (-0.5,-0.2,0.1,2.0171,2.4084,0)

# import socket

# HOST = "192.168.1.100"
# PORT = 30002
# s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
# s.connect((HOST,PORT))

# j0 = 0
# j1 = -3.1415/2
# j2 = 3.1415/2
# j3 = -3.1415/2
# j4 = -3.1415/2
# j5 = 0;

# joint_acc = 1.2
# joint_vel = 0.25

# # command = "movej([%f,%f,%f,%f,%f,%f],a=%f,v=%f)\n" % (j0,j1,j2,j3,j4,j5,joint_acc,joint_vel)



# #


# # True closes
# command = "set_digital_out(8,True)\n"

# s.send(str.encode(command))
# data = s.recv(1024)



# s.close()
# print("Received",repr(data))





# print()

# String.Format ("movej([%f,%f,%f,%f,%f, %f], a={6}, v={7})\n", j0, j1, j2, j3, j4, j5, a, v);