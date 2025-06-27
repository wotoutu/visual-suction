#!/usr/bin/env python

import time
import os
import random
import threading
import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import cv2
from collections import namedtuple

import torch
from torch.autograd import Variable
from robot import Robot
from trainer import Trainer
from logger import Logger
import utils
from simulation import vrep

def main(args):


    # --------------- 环境参数初始化 ---------------
    is_sim = args.is_sim #是否模拟还是真实机器人
    obj_mesh_dir = os.path.abspath(args.obj_mesh_dir) if is_sim else None # 如果在模拟模式下，设置物体网格模型目录路径。
    num_obj = args.num_obj if is_sim else None # 在模拟模式下指定要添加到场景中的物体数量。
    #如果不是模拟模式（即使用真实 UR5 机械臂），设置 TCP 和 RTC 客户端的 IP 地址和端口号。
    tcp_host_ip = args.tcp_host_ip if not is_sim else None
    tcp_port = args.tcp_port if not is_sim else None
    rtc_host_ip = args.rtc_host_ip if not is_sim else None
    rtc_port = args.rtc_port if not is_sim else None

    if is_sim:
        # 设置机器人工作空间的边界范围（单位：米）
        workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
    else:
        workspace_limits = np.asarray([[0.3, 0.748], [-0.224, 0.224], [-0.255, -0.1]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)

    #设置高度图的分辨率（单位：米/像素）
    heightmap_resolution = args.heightmap_resolution
    #设置随机种子，确保实验可重复
    random_seed = args.random_seed
    force_cpu = args.force_cpu

    # ------------- 算法选择 -------------
    method = args.method# 当前只使用吸取策略
    suction_rewards=args.suction_rewards if method == 'suction_net' else None
    #未来奖励的折扣因子（γ），用于强化学习中的 Q-learning 更新公式。
    future_reward_discount = args.future_reward_discount
    #是否启用经验回放
    experience_replay = args.experience_replay
    #是否启用启发式引导
    heuristic_bootstrap = args.heuristic_bootstrap
    #是否启用探索率衰减
    explore_rate_decay = args.explore_rate_decay
    # # 是否只执行抓取动作，跳过推的动作。
    # grasp_only = args.grasp_only

    # -------------- Testing options --------------
    is_testing = args.is_testing#是否运行在测试模式。
    max_test_trials = args.max_test_trials # 个测试案例的最大尝试次数。
    test_preset_cases = args.test_preset_cases#是否使用预设的测试场景（从文件加载物体位置等信息）。
    #指定预设测试场景的文件路径。
    test_preset_file = os.path.abspath(args.test_preset_file) if test_preset_cases else None

    # ------ Pre-loading and logging options ------
    load_snapshot = args.load_snapshot #是否加载一个预先训练好的模型快照，如果为 True，则会从指定路径加载模型权重继续训练或测试。
    snapshot_file = os.path.abspath(args.snapshot_file)  if load_snapshot else None
    #是否继续之前记录的日志（例如训练数据、图像、高度图等）。
    continue_logging = args.continue_logging
    #日志保存的目录。
    logging_directory = os.path.abspath(args.logging_directory) if continue_logging else os.path.abspath('logs')
    #是否保存网络预测的可视化结果（如推/抓取热力图）。
    save_visualizations = args.save_visualizations
    # Set random seed
    np.random.seed(random_seed)

    # 机器人初始化
    robot = Robot(is_sim, obj_mesh_dir, num_obj=args.num_obj, workspace_limits=workspace_limits,
                  tcp_host_ip=tcp_host_ip, tcp_port=tcp_port, rtc_host_ip=rtc_host_ip, rtc_port=rtc_port,
                  is_testing=is_testing, test_preset_cases=test_preset_cases, test_preset_file=test_preset_file)


    # 负责神经网络的前向推理、反向传播、经验回放等训练逻辑
    trainer = Trainer(method,suction_rewards,future_reward_discount,is_testing, load_snapshot, snapshot_file, force_cpu)
    # 记录训练过程中的图像、高度图、动作、模型快照等信息
    logger = Logger(continue_logging, logging_directory)
    logger.save_camera_info(robot.cam_intrinsics, robot.cam_pose, robot.cam_depth_scale) #保存相机参数
    logger.save_heightmap_info(workspace_limits, heightmap_resolution) # Save heightmap parameters

    #如果启用了 continue_logging，则从已有日志目录中加载之前保存的训练记录
    if continue_logging:
        trainer.preload(logger.transitions_directory)

    # 判断是否连续多次动作未引起场景变化。
    no_change_count = [2, 2] if not is_testing else [0, 0]
    #探索概率控制智能体是随机尝试新动作（探索）还是选择已知最优动作（利用）。
    explore_prob = 0.5 if not is_testing else 0.0

    # Quick hack for nonlocal memory between threads in Python 2
    nonlocal_variables = {'executing_action' : False,#是否正在执行动作（防止并发冲突）
                          'primitive_action' : None,#当前动作类型（'push' 或 'grasp'）
                          'best_pix_ind' : None,# 网络预测的最佳动作坐标（旋转角度 + 像素坐标）
                          'suction_success' : False}
    def process_actions():
        while True:
            if nonlocal_variables['executing_action']:

                # === STEP 1: 推理 suction predictions ===
                # suction_predictions, _ = trainer.forward(color_heightmap, depth_heightmap)
                # assert suction_predictions is not None, "trainer.forward() returned None"

                best_suction_conf = np.max(suction_predictions)
                print('Primitive confidence score: %.6f (suction)' % best_suction_conf)

                nonlocal_variables['primitive_action'] = 'suction'
                # explore_actions = np.random.uniform() < explore_prob
                explore_actions = False
                # === STEP 2: 记录策略 ===
                strategy = 'explore' if explore_actions else 'exploit'
                print('[INFO] Strategy: %s (exploration prob: %.6f)' % (strategy, explore_prob))
                trainer.is_exploit_log.append([0 if explore_actions else 1])
                logger.write_to_log('is-exploit', trainer.is_exploit_log)

                # === STEP 3: 决策点 ===
                #use_heuristic = heuristic_bootstrap and no_change_count[0] >= 2
                if heuristic_bootstrap and nonlocal_variables['primitive_action'] == 'suction' and no_change_count[1] >= 2:
                    print('Heuristic triggered due to 2+ unchanged steps')
                    best_pix_ind = trainer.suction_heuristic(valid_depth_heightmap)
                    no_change_count[0] = 0
                    predicted_value=suction_predictions[nonlocal_variables['best_pix_ind']]
                    use_heuristic=True
                else:
                    # best_pix_ind = np.unravel_index(np.argmax(suction_predictions), suction_predictions.shape)
                    use_heuristic = False

                    if nonlocal_variables['primitive_action'] == 'suction':
                        nonlocal_variables['best_pix_ind'] = np.unravel_index(np.argmax(suction_predictions),
                                                                              suction_predictions.shape)
                        predicted_value = np.max(suction_predictions)
                #predicted_value = suction_predictions[best_pix_ind]
                trainer.use_heuristic_log.append([1 if use_heuristic else 0])
                logger.write_to_log('use-heuristic', trainer.use_heuristic_log)

                trainer.predicted_value_log.append([predicted_value])
                logger.write_to_log('predicted-value', trainer.predicted_value_log)

                print('Action: %s at (rot: %d, y: %d, x: %d)' % (nonlocal_variables['primitive_action'], nonlocal_variables['best_pix_ind'][0], nonlocal_variables['best_pix_ind'][1], nonlocal_variables['best_pix_ind'][2]))
                #nonlocal_variables['best_pix_ind'] = best_pix_ind

                # === STEP 4: 像素转世界坐标 ===
                best_rotation_angle = np.deg2rad(nonlocal_variables['best_pix_ind'][0]*(360.0/trainer.model.num_rotations))
                best_pix_x = nonlocal_variables['best_pix_ind'][2]
                best_pix_y = nonlocal_variables['best_pix_ind'][1]

                primitive_position = [
                    best_pix_x * heightmap_resolution + workspace_limits[0][0],
                    best_pix_y * heightmap_resolution + workspace_limits[1][0],
                    valid_depth_heightmap[best_pix_y][best_pix_x] + workspace_limits[2][0]
                ]

                # === STEP 5: 日志记录 ===
                # trainer.executed_action_log.append([2, best_pix_ind[0], best_pix_y, best_pix_x])
                # logger.write_to_log('executed-action', trainer.executed_action_log)

                if save_visualizations:
                    suction_pred_vis = trainer.get_prediction_vis(suction_predictions, color_heightmap, nonlocal_variables['best_pix_ind'])
                    logger.save_visualizations(trainer.iteration, suction_pred_vis, 'suction')
                    cv2.imwrite('visualization.suction.png', suction_pred_vis)

                # === STEP 6: 执行吸取 ===
                success = robot.suction(primitive_position, best_rotation_angle, workspace_limits)
                nonlocal_variables['suction_success'] = success
                print('[INFO] Suction successful: %r' % success)

                # === STEP 7: 标记完成 ===
                nonlocal_variables['suction_success'] = False
                change_detected=False

                if nonlocal_variables['primitive_action'] == 'suction':
                    nonlocal_variables['suction_success'] = robot.suction(primitive_position, best_rotation_angle,
                                                                    workspace_limits)
                    print('Suction successful: %r' % (nonlocal_variables['suction_success']))

                nonlocal_variables['executing_action'] = False

            time.sleep(0.01)

    # 启动动作线程
    action_thread = threading.Thread(target=process_actions)
    action_thread.daemon = True
    action_thread.start()
    exit_called=False
#-------------------------------------------------------------
    # def calculate_reward(trainer, nonlocal_variables,
    #                      next_color_heightmap, next_depth_heightmap,
    #                      suction_predictions):
    #     """根据吸取结果与未来预测计算奖励（含 Q 折扣）"""
    #     reward = 0
    #
    #     # 当前动作即时奖励
    #     if nonlocal_variables['last_action'] == 'suction':
    #         if nonlocal_variables['last_action_success']:
    #             reward += 10.0
    #         else:
    #             reward -= 2.0
    #
    #     # 未来折扣奖励
    #     if not nonlocal_variables['is_testing'] and nonlocal_variables['future_reward_discount'] < 1.0:
    #         with torch.no_grad():
    #             future_predictions, _ = trainer.forward(
    #                 next_color_heightmap, next_depth_heightmap
    #             )
    #             max_future_q = np.max(future_predictions)
    #             reward += nonlocal_variables['future_reward_discount'] * max_future_q
    #
    #     return reward

    # def check_object_grasped():
    #     return robot.is_object_attached()
    # -------------------------------------------------------------
    # -------------------------------------------------------------

    while True:
        print('\n%s iteration: %d' % ('Testing' if is_testing else 'Training', trainer.iteration))
        iteration_time_0 = time.time()

        if is_sim:
            robot.check_sim()

        color_img, depth_img = robot.get_camera_data()
        depth_img = depth_img * robot.cam_depth_scale

        color_heightmap, depth_heightmap = utils.get_heightmap(
            color_img, depth_img,
            robot.cam_intrinsics, robot.cam_pose,
            workspace_limits, heightmap_resolution
        )
        valid_depth_heightmap = depth_heightmap.copy()
        valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0

        logger.save_images(trainer.iteration, color_img, depth_img, '0')
        logger.save_heightmaps(trainer.iteration, color_heightmap, valid_depth_heightmap, '0')

        # 检查是否需要重新放置物体
        stuff_count = np.zeros(valid_depth_heightmap.shape)
        stuff_count[valid_depth_heightmap > 0.02] = 1
        empty_threshold = 300
        if is_sim and is_testing:
            empty_threshold = 10
        if np.sum(stuff_count) < empty_threshold or (is_sim and no_change_count[0] + no_change_count[1] > 10):
            no_change_count = [0, 0]
            if is_sim:
                print('Not enough objects in view (value: %d)! Repositioning objects.' % (np.sum(stuff_count)))
                robot.restart_sim()
                robot.add_objects()
                if is_testing:  # If at end of test run, re-load original weights (before test run)
                    trainer.model.load_state_dict(torch.load(snapshot_file))
            else:
                # print('Not enough stuff on the table (value: %d)! Pausing for 30 seconds.' % (np.sum(stuff_count)))
                # time.sleep(30)
                print('Not enough stuff on the table (value: %d)! Flipping over bin of objects...' % (
                    np.sum(stuff_count)))
                robot.restart_real()

            trainer.clearance_log.append([trainer.iteration])
            logger.write_to_log('clearance', trainer.clearance_log)
            if is_testing and len(trainer.clearance_log) >= max_test_trials:
                exit_called = True
            continue
        if not exit_called:
            # Run forward pass with network to get affordances
            suction_predictions, state_feat = trainer.forward(color_heightmap, valid_depth_heightmap,
                                                                              is_volatile=True)

            # Execute best primitive action on robot in another thread
            nonlocal_variables['executing_action'] = True

        # # 执行吸取预测
        # if not exit_called:
        #     nonlocal_variables['best_pix_ind'] = trainer.suction_heuristic(valid_depth_heightmap)
        #
        #     # 构造虚拟 suction_predictions
        #     suction_predictions = np.zeros((16, 224, 224))  # 假设模型支持16种rotation
        #     suction_predictions[nonlocal_variables['best_pix_ind'][0],
        #                         nonlocal_variables['best_pix_ind'][1],
        #                         nonlocal_variables['best_pix_ind'][2]] = 1.0
        #
        #     nonlocal_variables['executing_action'] = True
        # # 每次都预测新的吸取动作
        # nonlocal_variables['best_pix_ind'] = trainer.suction_heuristic(valid_depth_heightmap)
        #
        # suction_predictions = np.zeros((16, 224, 224))
        # suction_predictions[nonlocal_variables['best_pix_ind'][0],
        #                     nonlocal_variables['best_pix_ind'][1],
        #                     nonlocal_variables['best_pix_ind'][2]] = 1.0
        #
        # nonlocal_variables['executing_action'] = True
        #
        # # 等待当前吸取动作完成
        # while nonlocal_variables['executing_action']:
        #     time.sleep(0.01)

        # 检测场景是否发生变化
        if 'prev_color_img' in locals():
            # Detect changes
            depth_diff = abs(depth_heightmap - prev_depth_heightmap)
            depth_diff[np.isnan(depth_diff)] = 0
            depth_diff[depth_diff > 0.3] = 0
            depth_diff[depth_diff < 0.01] = 0
            depth_diff[depth_diff > 0] = 1
            change_threshold = 300
            change_value = np.sum(depth_diff)
            change_detected = change_value > change_threshold or prev_suction_success
            print('Change detected: %r (value: %d)' % (change_detected, change_value))

            if change_detected:
                if prev_primitive_action == 'suction':
                    no_change_count[0] = 0
                else:
                    no_change_count[1] = 0
            else:
                if prev_primitive_action == 'suction':
                    no_change_count[0] += 1
                else:
                    no_change_count[1] += 1

            #no_change_count[0] = 0 if change_detected else no_change_count[0] + 1
            #根据当前行为是否成功、是否场景变化，计算强化学习中的目标值 label
            label_value, prev_reward_value = trainer.get_label_value(
                prev_primitive_action,
                prev_suction_success,
                change_detected,
                prev_suction_predictions,
                color_heightmap,
                valid_depth_heightmap)

            trainer.backprop(prev_color_heightmap, prev_valid_depth_heightmap,prev_primitive_action,
                             prev_best_pix_ind, label_value)

            # trainer.label_value_log.append([label_value])
            # logger.write_to_log('label-value', trainer.label_value_log)
            # trainer.reward_value_log.append([prev_reward_value])
            # logger.write_to_log('reward-value', trainer.reward_value_log)

            if not is_testing:
                explore_prob = max(0.5 * np.power(0.9998, trainer.iteration), 0.1) if explore_rate_decay else 0.5

            if experience_replay and not is_testing:
                sample_primitive_action = prev_primitive_action
                if sample_primitive_action == 'suction':
                    sample_primitive_action_id = 0
                    if method == 'reactive_net':
                        sample_reward_value = 0 if prev_reward_value == 1 else 1  # random.randint(1, 2) # 2
                    elif method == 'suction_net':
                        sample_reward_value = 0 if prev_reward_value == 0.5 else 0.5


                # Get samples of the same primitive but with different results
                sample_ind = np.argwhere(
                    np.logical_and(np.asarray(trainer.reward_value_log)[1:trainer.iteration, 0] == sample_reward_value,
                                   np.asarray(trainer.executed_action_log)[1:trainer.iteration,
                                   0] == sample_primitive_action_id))

                if sample_ind.size > 0:

                    # Find sample with highest surprise value
                    if method == 'suction_net':
                        sample_surprise_values = np.abs(np.asarray(trainer.predicted_value_log)[sample_ind[:, 0]] -
                                                        np.asarray(trainer.label_value_log)[sample_ind[:, 0]])

                    sorted_surprise_ind = np.argsort(sample_surprise_values[:, 0])
                    sorted_sample_ind = sample_ind[sorted_surprise_ind, 0]
                    pow_law_exp = 2
                    rand_sample_ind = int(np.round(np.random.power(pow_law_exp, 1) * (sample_ind.size - 1)))
                    sample_iteration = sorted_sample_ind[rand_sample_ind]
                    print('Experience replay: iteration %d (surprise value: %f)' % (
                    sample_iteration, sample_surprise_values[sorted_surprise_ind[rand_sample_ind]]))

                    # Load sample RGB-D heightmap
                    sample_color_heightmap = cv2.imread(
                        os.path.join(logger.color_heightmaps_directory, '%06d.0.color.png' % (sample_iteration)))
                    sample_color_heightmap = cv2.cvtColor(sample_color_heightmap, cv2.COLOR_BGR2RGB)
                    sample_depth_heightmap = cv2.imread(
                        os.path.join(logger.depth_heightmaps_directory, '%06d.0.depth.png' % (sample_iteration)), -1)
                    sample_depth_heightmap = sample_depth_heightmap.astype(np.float32) / 100000

                    # Compute forward pass with sample
                    with torch.no_grad():
                        sample_suction_predictions, sample_state_feat = trainer.forward(
                            sample_color_heightmap, sample_depth_heightmap, is_volatile=True)

                    # Load next sample RGB-D heightmap
                    next_sample_color_heightmap = cv2.imread(
                        os.path.join(logger.color_heightmaps_directory, '%06d.0.color.png' % (sample_iteration + 1)))
                    next_sample_color_heightmap = cv2.cvtColor(next_sample_color_heightmap, cv2.COLOR_BGR2RGB)
                    next_sample_depth_heightmap = cv2.imread(
                        os.path.join(logger.depth_heightmaps_directory, '%06d.0.depth.png' % (sample_iteration + 1)),
                        -1)
                    next_sample_depth_heightmap = next_sample_depth_heightmap.astype(np.float32) / 100000

                    sample_suction_success = sample_reward_value == 0.5
                    sample_change_detected = sample_suction_success
                    # new_sample_label_value, _ = trainer.get_label_value(sample_primitive_action, sample_push_success, sample_grasp_success, sample_change_detected, sample_push_predictions, sample_grasp_predictions, next_sample_color_heightmap, next_sample_depth_heightmap)

                    # Get labels for sample and backpropagate
                    sample_best_pix_ind = (np.asarray(trainer.executed_action_log)[sample_iteration, 1:4]).astype(int)
                    trainer.backprop(sample_color_heightmap, sample_depth_heightmap, sample_primitive_action,
                                     sample_best_pix_ind, trainer.label_value_log[sample_iteration])

                    if sample_primitive_action == 'suction':
                        trainer.predicted_value_log[sample_iteration] = [np.max(sample_suction_predictions)]
                else:
                    print('Not enough prior training samples. Skipping experience replay.')

                    # Save model snapshot
            if not is_testing:
                logger.save_backup_model(trainer.model, method)
                if trainer.iteration % 50 == 0:
                    logger.save_model(trainer.iteration, trainer.model, method)
                    if trainer.use_cuda:
                        trainer.model = trainer.model.cuda()

        while nonlocal_variables['executing_action']:
            time.sleep(0.01)

        if exit_called:
            break

        # 更新前一帧状态
        prev_color_img = color_img.copy()
        prev_depth_img = depth_img.copy()
        prev_color_heightmap = color_heightmap.copy()
        prev_depth_heightmap = depth_heightmap.copy()
        prev_valid_depth_heightmap = valid_depth_heightmap.copy()
        prev_suction_success = nonlocal_variables['suction_success']
        prev_primitive_action = nonlocal_variables['primitive_action']
        prev_suction_predictions = suction_predictions.copy()
        prev_best_pix_ind = nonlocal_variables['best_pix_ind']

        iteration_time_1 = time.time()
        print('Time elapsed: %f' % (iteration_time_1 - iteration_time_0))
        trainer.iteration += 1

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='Train robotic agents to learn how to plan complementary pushing and grasping actions for manipulation with deep reinforcement learning in PyTorch.')

    # --------------- Setup options ---------------
    parser.add_argument('--is_sim', dest='is_sim', action='store_true', default=False,                                    help='run in simulation?')
    parser.add_argument('--obj_mesh_dir', dest='obj_mesh_dir', action='store', default='objects/blocks',                  help='directory containing 3D mesh files (.obj) of objects to be added to simulation')
    parser.add_argument('--num_obj', dest='num_obj', type=int, action='store', default=4,                                help='number of objects to add to simulation')
    parser.add_argument('--tcp_host_ip', dest='tcp_host_ip', action='store', default='100.127.7.223',                     help='IP address to robot arm as TCP client (UR5)')
    parser.add_argument('--tcp_port', dest='tcp_port', type=int, action='store', default=30002,                           help='port to robot arm as TCP client (UR5)')
    parser.add_argument('--rtc_host_ip', dest='rtc_host_ip', action='store', default='100.127.7.223',                     help='IP address to robot arm as real-time client (UR5)')
    parser.add_argument('--rtc_port', dest='rtc_port', type=int, action='store', default=30003,                           help='port to robot arm as real-time client (UR5)')
    parser.add_argument('--heightmap_resolution', dest='heightmap_resolution', type=float, action='store', default=0.002, help='meters per pixel of heightmap')
    parser.add_argument('--random_seed', dest='random_seed', type=int, action='store', default=1234,                      help='random seed for simulation and neural net initialization')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,                                    help='force code to run in CPU mode')

    # ------------- Algorithm options -------------
    parser.add_argument('--method', dest='method', action='store', default='suction_net',                               help='set to \'reactive\' (supervised learning) or \'reinforcement\' (reinforcement learning ie Q-learning)')
    parser.add_argument('--suction_rewards', dest='suction_rewards', action='store_true', default=False,                        help='use immediate rewards (from change detection) for pushing?')
    parser.add_argument('--future_reward_discount', dest='future_reward_discount', type=float, action='store', default=0.5)
    parser.add_argument('--experience_replay', dest='experience_replay', action='store_true', default=False,              help='use prioritized experience replay?')
    parser.add_argument('--heuristic_bootstrap', dest='heuristic_bootstrap', action='store_true', default=False,          help='use handcrafted grasping algorithm when grasping fails too many times in a row during training?')
    parser.add_argument('--explore_rate_decay', dest='explore_rate_decay', action='store_true', default=False)
    #parser.add_argument('--grasp_only', dest='grasp_only', action='store_true', default=False)

    # -------------- Testing options --------------
    parser.add_argument('--is_testing', dest='is_testing', action='store_true', default=False)
    parser.add_argument('--max_test_trials', dest='max_test_trials', type=int, action='store', default=30,                help='maximum number of test runs per case/scenario')
    parser.add_argument('--test_preset_cases', dest='test_preset_cases', action='store_true', default=False)
    parser.add_argument('--test_preset_file', dest='test_preset_file', action='store', default='test-10-obj-01.txt')

    # ------ Pre-loading and logging options ------
    parser.add_argument('--load_snapshot', dest='load_snapshot', action='store_true', default=False,                      help='load pre-trained snapshot of model?')
    parser.add_argument('--snapshot_file', dest='snapshot_file', action='store')
    parser.add_argument('--continue_logging', dest='continue_logging', action='store_true', default=False,                help='continue logging from previous session?')
    parser.add_argument('--logging_directory', dest='logging_directory', action='store')
    parser.add_argument('--save_visualizations', dest='save_visualizations', action='store_true', default=False,          help='save visualizations of FCN predictions?')

    # Run main program with specified arguments
    args = parser.parse_args()
    main(args)
