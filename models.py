#!/usr/bin/env python

from collections import OrderedDict
import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt
import time

class reactive_net(nn.Module):

    def __init__(self, use_cuda): # , snapshot=None
        super(reactive_net, self).__init__()
        self.use_cuda = use_cuda

        # Initialize network trunks with DenseNet pre-trained on ImageNet
        self.suction_color_trunk = torchvision.models.densenet.densenet121(pretrained=True)
        self.suction_depth_trunk = torchvision.models.densenet.densenet121(pretrained=True)

        self.num_rotations = 16

        # Construct network branches for suctioning and grasping
        self.suctionnet = nn.Sequential(OrderedDict([
            ('suction-norm0', nn.BatchNorm2d(2048)),
            ('suction-relu0', nn.ReLU(inplace=True)),
            ('suction-conv0', nn.Conv2d(2048, 64, kernel_size=1, stride=1, bias=False)),
            ('suction-norm1', nn.BatchNorm2d(64)),
            ('suction-relu1', nn.ReLU(inplace=True)),
            ('suction-conv1', nn.Conv2d(64, 3, kernel_size=1, stride=1, bias=False))
            # ('suction-upsample2', nn.Upsample(scale_factor=4, mode='bilinear'))
        ]))

        # Initialize network weights
        for m in self.named_modules():
            if 'suction-' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    nn.init.kaiming_normal(m[1].weight.data)
                elif isinstance(m[1], nn.BatchNorm2d):
                    m[1].weight.data.fill_(1)
                    m[1].bias.data.zero_()

        # Initialize output variable (for backprop)
        self.interm_feat = []
        self.output_prob = []


    def forward(self, input_color_data, input_depth_data, is_volatile=False, specific_rotation=-1):

        if is_volatile:
            output_prob = []
            interm_feat = []

            # Apply rotations to images
            for rotate_idx in range(self.num_rotations):
                rotate_theta = np.radians(rotate_idx*(360/self.num_rotations))

                # Compute sample grid for rotation BEFORE neural network
                affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                affine_mat_before.shape = (2,3,1)
                affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
                if self.use_cuda:
                    flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_color_data.size())
                else:
                    flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False), input_color_data.size())

                # Rotate images clockwise
                if self.use_cuda:
                    rotate_color = F.grid_sample(Variable(input_color_data, volatile=True).cuda(), flow_grid_before, mode='nearest')
                    rotate_depth = F.grid_sample(Variable(input_depth_data, volatile=True).cuda(), flow_grid_before, mode='nearest')
                else:
                    rotate_color = F.grid_sample(Variable(input_color_data, volatile=True), flow_grid_before, mode='nearest')
                    rotate_depth = F.grid_sample(Variable(input_depth_data, volatile=True), flow_grid_before, mode='nearest')

                # Compute intermediate features
                interm_suction_color_feat = self.suction_color_trunk.features(rotate_color)
                interm_suction_depth_feat = self.suction_depth_trunk.features(rotate_depth)
                interm_suction_feat = torch.cat((interm_suction_color_feat, interm_suction_depth_feat), dim=1)
                interm_feat.append(interm_suction_feat)

                # Compute sample grid for rotation AFTER branches
                affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],[-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
                affine_mat_after.shape = (2,3,1)
                affine_mat_after = torch.from_numpy(affine_mat_after).permute(2,0,1).float()
                if self.use_cuda:
                    flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(), interm_suction_feat.data.size())
                else:
                    flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False), interm_suction_feat.data.size())

                # Forward pass through branches, undo rotation on output predictions, upsample results
                output_prob.append(nn.Upsample(scale_factor=16, mode='bilinear').forward(F.grid_sample(self.suctionnet(interm_suction_feat),
                                                                                                       flow_grid_after, mode='nearest')))

            return output_prob, interm_feat

        else:
            self.output_prob = []
            self.interm_feat = []

            # Apply rotations to intermediate features
            # for rotate_idx in range(self.num_rotations):
            rotate_idx = specific_rotation
            rotate_theta = np.radians(rotate_idx*(360/self.num_rotations))

            # Compute sample grid for rotation BEFORE branches
            affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
            affine_mat_before.shape = (2,3,1)
            affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
            if self.use_cuda:
                flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_color_data.size())
            else:
                flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False), input_color_data.size())

            # Rotate images clockwise
            if self.use_cuda:
                rotate_color = F.grid_sample(Variable(input_color_data, requires_grad=False).cuda(), flow_grid_before, mode='nearest')
                rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False).cuda(), flow_grid_before, mode='nearest')
            else:
                rotate_color = F.grid_sample(Variable(input_color_data, requires_grad=False), flow_grid_before, mode='nearest')
                rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False), flow_grid_before, mode='nearest')

            # Compute intermediate features
            interm_suction_color_feat = self.suction_color_trunk.features(rotate_color)
            interm_suction_depth_feat = self.suction_depth_trunk.features(rotate_depth)
            interm_suction_feat = torch.cat((interm_suction_color_feat, interm_suction_depth_feat), dim=1)
            self.interm_feat.append(interm_suction_feat)

            # Compute sample grid for rotation AFTER branches
            affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],[-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
            affine_mat_after.shape = (2,3,1)
            affine_mat_after = torch.from_numpy(affine_mat_after).permute(2,0,1).float()
            if self.use_cuda:
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(), interm_suction_feat.data.size())
            else:
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False), interm_suction_feat.data.size())

            # Forward pass through branches, undo rotation on output predictions, upsample results
            self.output_prob.append(nn.Upsample(scale_factor=16, mode='bilinear').forward(F.grid_sample(self.suctionnet(interm_suction_feat),
                                                                                                        flow_grid_after, mode='nearest')))

            return self.output_prob, self.interm_feat

class suction_net(nn.Module):
    def __init__(self, use_cuda):
        super(suction_net, self).__init__()
        self.use_cuda = use_cuda

        # 主干使用 DenseNet 特征提取器（预训练）
        self.suction_color_trunk = torchvision.models.densenet.densenet121(pretrained=True)
        self.suction_depth_trunk = torchvision.models.densenet.densenet121(pretrained=True)

        self.num_rotations = 16

        # 吸取分支（融合 RGB + depth）
        self.suctionnet = nn.Sequential(OrderedDict([
            ('suction-norm0', nn.BatchNorm2d(2048)),
            ('suction-relu0', nn.ReLU(inplace=True)),
            ('suction-conv0', nn.Conv2d(2048, 64, kernel_size=1, stride=1, bias=False)),
            ('suction-norm1', nn.BatchNorm2d(64)),
            ('suction-relu1', nn.ReLU(inplace=True)),
            ('suction-conv1', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False))
        ]))

        # 初始化参数
        for m in self.named_modules():
            if 'suction-' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    nn.init.kaiming_normal_(m[1].weight.data)
                elif isinstance(m[1], nn.BatchNorm2d):
                    m[1].weight.data.fill_(1)
                    m[1].bias.data.zero_()
        self.interm_feat = []
        self.output_prob = []

    def forward(self, input_color_data, input_depth_data, is_volatile=False, specific_rotation=-1):
        output_prob = []

        if is_volatile:
            with torch.no_grad():
                output_prob = []
                interm_feat = []

                # Apply rotations to images
                for rotate_idx in range(self.num_rotations):
                    rotate_theta = np.radians(rotate_idx * (360 / self.num_rotations))

                    # Compute sample grid for rotation BEFORE neural network
                    affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],
                                                    [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                    affine_mat_before.shape = (2, 3, 1)
                    affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()
                    if self.use_cuda:
                        flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(),
                                                         input_color_data.size())
                    else:
                        flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False),
                                                         input_color_data.size())

                    # Rotate images clockwise
                    if self.use_cuda:
                        rotate_color = F.grid_sample(Variable(input_color_data, volatile=True).cuda(), flow_grid_before,
                                                     mode='nearest')
                        rotate_depth = F.grid_sample(Variable(input_depth_data, volatile=True).cuda(), flow_grid_before,
                                                     mode='nearest')
                    else:
                        rotate_color = F.grid_sample(Variable(input_color_data, volatile=True), flow_grid_before,
                                                     mode='nearest')
                        rotate_depth = F.grid_sample(Variable(input_depth_data, volatile=True), flow_grid_before,
                                                     mode='nearest')

                    # Compute intermediate features
                    interm_suction_color_feat = self.suction_color_trunk.features(rotate_color)
                    interm_suction_depth_feat = self.suction_depth_trunk.features(rotate_depth)
                    interm_suction_feat = torch.cat((interm_suction_color_feat, interm_suction_depth_feat), dim=1)

                    interm_feat.append(interm_suction_feat)

                    # Compute sample grid for rotation AFTER branches
                    affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],
                                                   [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
                    affine_mat_after.shape = (2, 3, 1)
                    affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
                    if self.use_cuda:
                        flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(),
                                                        interm_suction_feat.data.size())
                    else:
                        flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False),
                                                        interm_suction_feat.data.size())

                    # Forward pass through branches, undo rotation on output predictions, upsample results
                    output_prob.append(nn.Upsample(scale_factor=16, mode='bilinear').forward(F.grid_sample(self.suctionnet(interm_suction_feat), flow_grid_after, mode='nearest')))

            return output_prob, interm_feat

        else:
            self.output_prob = []
            self.interm_feat = []

            # Apply rotations to intermediate features
            # for rotate_idx in range(self.num_rotations):
            rotate_idx = specific_rotation
            rotate_theta = np.radians(rotate_idx * (360 / self.num_rotations))

            # Compute sample grid for rotation BEFORE branches
            affine_mat_before = np.asarray(
                [[np.cos(-rotate_theta), np.sin(-rotate_theta), 0], [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
            affine_mat_before.shape = (2, 3, 1)
            affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()
            if self.use_cuda:
                flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(),
                                                 input_color_data.size())
            else:
                flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False),
                                                 input_color_data.size())

            # Rotate images clockwise
            if self.use_cuda:
                rotate_color = F.grid_sample(Variable(input_color_data, requires_grad=False).cuda(), flow_grid_before,
                                             mode='nearest')
                rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False).cuda(), flow_grid_before,
                                             mode='nearest')
            else:
                rotate_color = F.grid_sample(Variable(input_color_data, requires_grad=False), flow_grid_before,
                                             mode='nearest')
                rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False), flow_grid_before,
                                             mode='nearest')

            # Compute intermediate features
            interm_suction_color_feat = self.suction_color_trunk.features(rotate_color)
            interm_suction_depth_feat = self.suction_depth_trunk.features(rotate_depth)
            interm_suction_feat = torch.cat((interm_suction_color_feat, interm_suction_depth_feat), dim=1)

            self.interm_feat.append(interm_suction_feat)

            # Compute sample grid for rotation AFTER branches
            affine_mat_after = np.asarray(
                [[np.cos(rotate_theta), np.sin(rotate_theta), 0], [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
            affine_mat_after.shape = (2, 3, 1)
            affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
            if self.use_cuda:
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(),
                                                interm_suction_feat.data.size())
            else:
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False),
                                                interm_suction_feat.data.size())

            # Forward pass through branches, undo rotation on output predictions, upsample results
            self.output_prob.append(nn.Upsample(scale_factor=16, mode='bilinear').forward(
                F.grid_sample(self.suctionnet(interm_suction_feat), flow_grid_after, mode='nearest')))

            return self.output_prob, self.interm_feat