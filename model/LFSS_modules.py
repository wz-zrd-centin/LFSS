#!/usr/bin/python3
#coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import os
from model.batchnorm import SynchronizedBatchNorm2d

eps = 1e-12

def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear): 
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.Sigmoid, nn.Softmax, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool1d, nn.Sigmoid, nn.Identity)):
            pass
        else:
            m.initialize()


class CropLayer(nn.Module):
    #   E.g., (-1, 0) means this layer should crop the first and last rows of the feature map. And (0, -1) crops the first and last columns
    def __init__(self, crop_set):
        super(CropLayer, self).__init__()
        self.rows_to_crop = - crop_set[0]
        self.cols_to_crop = - crop_set[1]
        assert self.rows_to_crop >= 0
        assert self.cols_to_crop >= 0

    def forward(self, input):
        return input[:, :, self.rows_to_crop:-self.rows_to_crop, self.cols_to_crop:-self.cols_to_crop]


class asyConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False):
        super(asyConv, self).__init__()
        self.deploy = deploy
        if deploy:
            self.fused_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size,kernel_size), stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
            self.initialize()
        else:
            self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(kernel_size, kernel_size), stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=False,
                                         padding_mode=padding_mode)
            self.square_bn = nn.BatchNorm2d(num_features=out_channels)

            center_offset_from_origin_border = padding - kernel_size // 2
            ver_pad_or_crop = (center_offset_from_origin_border + 1, center_offset_from_origin_border)
            hor_pad_or_crop = (center_offset_from_origin_border, center_offset_from_origin_border + 1)
            if center_offset_from_origin_border >= 0:
                self.ver_conv_crop_layer = nn.Identity()
                ver_conv_padding = ver_pad_or_crop
                self.hor_conv_crop_layer = nn.Identity()
                hor_conv_padding = hor_pad_or_crop
            else:
                self.ver_conv_crop_layer = CropLayer(crop_set=ver_pad_or_crop)
                ver_conv_padding = (0, 0)
                self.hor_conv_crop_layer = CropLayer(crop_set=hor_pad_or_crop)
                hor_conv_padding = (0, 0)
            self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1),
                                      stride=stride,
                                      padding=ver_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)

            self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3),
                                      stride=stride,
                                      padding=hor_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)
            self.ver_bn = nn.BatchNorm2d(num_features=out_channels)
            self.hor_bn = nn.BatchNorm2d(num_features=out_channels)
            self.initialize()


    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)
        else:
            square_outputs = self.square_conv(input)
            square_outputs = self.square_bn(square_outputs)
            vertical_outputs = self.ver_conv_crop_layer(input)
            vertical_outputs = self.ver_conv(vertical_outputs)
            vertical_outputs = self.ver_bn(vertical_outputs)
            horizontal_outputs = self.hor_conv_crop_layer(input)
            horizontal_outputs = self.hor_conv(horizontal_outputs)
            horizontal_outputs = self.hor_bn(horizontal_outputs)
            return square_outputs + vertical_outputs + horizontal_outputs

    def initialize(self):
        weight_init(self)


class AttentionConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels // 2, 1)
        self.kernel = nn.Parameter(torch.normal(0, 0.01, size=(out_channels, out_channels, kernel_size, kernel_size)))

        self.qconv = nn.Conv2d(out_channels // 2, out_channels, 1)
        self.kconv = nn.Conv2d(out_channels // 2, out_channels, 1)
        self.vconv = nn.Conv2d(out_channels // 2, out_channels, 1)

        self.conv3 = nn.Conv2d(out_channels * 2, out_channels, kernel_size, stride, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.softmax_dim = 0  # Change softmax dimension according to your needs

        self.initialize()

    def forward(self, x):
        batch_size, channels = x.size()[0], x.size()[1]
        shortcut = self.shortcut(x)

        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))

        q = self.qconv(x1)
        k = self.kconv(x2)
        v = self.vconv(x2)

        batch_size, channels, height, width = v.shape
        q = q.reshape(batch_size, channels, -1)
        k = k.reshape(batch_size, channels, -1)
        v = v.reshape(batch_size, channels, -1)
        scale = k.shape[-1] ** -0.5

        # print(q.size(),k.size())
        attn = torch.bmm(q, k.permute(0, 2, 1)) * (scale + 1e-8)
        attn = F.softmax(attn, dim=self.softmax_dim)
        # print(attn.size(),v.size())
        weighted_v = torch.bmm(attn, v)

        weighted_v = weighted_v.reshape(batch_size, channels, height, width)

        out = F.conv2d(weighted_v, self.kernel, padding=self.kernel_size // 2)
        out = self.bn(out)
        out = self.gamma * out + shortcut
        out = self.conv3(torch.cat([out, shortcut], dim=1))

        return out

    def initialize(self):
        # pass
        nn.init.xavier_uniform_(self.kernel)

class MFE(nn.Module):
    """ Enhance the feature diversity.
    """
    def __init__(self, x, y):
        super(MFE, self).__init__()
        self.asyConv = asyConv(in_channels=x, out_channels=y, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False)
        self.oriConv = nn.Conv2d(x, y, kernel_size=3, stride=1, padding=1)
        self.atrConv = nn.Sequential(
            nn.Conv2d(x, y, kernel_size=3, dilation=2, padding=2, stride=1), nn.BatchNorm2d(y), nn.PReLU()
        )
        self.dynamic_conv = AttentionConv2d(x,y,3,1,1)
        self.conv2d = nn.Conv2d(y*3, y, kernel_size=3, stride=1, padding=1)
        self.bn2d = nn.BatchNorm2d(y)
        self.initialize()

    def forward(self, f):
        p1 = self.oriConv(f)
        p2 = self.asyConv(f)
        p3 = self.dynamic_conv(f)
        p = torch.cat((p1, p2, p3), 1)
        p = F.relu(self.bn2d(self.conv2d(p)), inplace=True)

        return p

    def initialize(self):
        #pass
        weight_init(self)

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):        
        x = self.atrous_conv(x)

        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class SSE(nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super(SSE, self).__init__()

        self.aspp_3 = _ASPPModule(in_channels, out_channels, 3, padding=3, dilation=3, BatchNorm=nn.BatchNorm2d)
        self.aspp_6 = _ASPPModule(in_channels, out_channels, 3, padding=6, dilation=6, BatchNorm=nn.BatchNorm2d)
        self.aspp_12 = _ASPPModule(in_channels, out_channels, 3, padding=12, dilation=12, BatchNorm=nn.BatchNorm2d)
        self.conv_cross = nn.Conv2d(3*in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_cross = nn.BatchNorm2d(in_channels)
        self.aspp_conv = nn.Conv2d(out_channels, out_channels, 1)
        self.aspp_bn = nn.BatchNorm2d(out_channels)

        self.conv_1 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, in1, in2=None, in3=None):
        if in2!=None and in1.size()[2:] != in2.size()[2:]:
            in2 = F.interpolate(in2, size=in1.size()[2:], mode='bilinear')
        else: in2 = in1
        if in3!=None and in1.size()[2:] != in3.size()[2:]:
            in3 = F.interpolate(in3, size=in1.size()[2:], mode='bilinear')
        else: in3 = in1

        x = torch.cat((in1,in2,in3), 1)

        # print('conv前',x.size())
        x = self.conv_cross(x)
        # print('conv后',x.size())
        x = self.bn_cross(x)
        # print('bn_cross后',x.size())
        feat =  F.relu(x)
        # print('relu后',x.size())
        # feat = F.relu(self.bn_cross(self.conv_cross(x))) #[B, C, H, W]

        # feat = torch.cat([in1, in2], dim=1) if in3 is None else torch.cat([in1, in2, in3], dim=1)
        x3 = self.aspp_3(feat)
        # print('x3',x.size())
        x6 = self.aspp_6(feat)
        # print('x6',x.size())
        if x3 is None:
            x_aspp = x3 + x6
        else:
            x12 = self.aspp_12(feat)


            x_aspp = x3 + x6 + x12

        # print('x12', x.size())
        x_aspp = self.aspp_conv(x_aspp)
        # print('x_aspp', x_aspp.size())

        x_aspp = self.aspp_bn(x_aspp)
        # print('x_aspp_bn', x_aspp.size())

        x = self.conv_1(x_aspp)
        # print('x_aspp_bn_conv1', x_aspp.size())
        return x

class EmRouting2d(nn.Module):
    def __init__(self, A, B, caps_size, kernel_size=3, stride=1, padding=1, iters=3, final_lambda=1e-2):
        super(EmRouting2d, self).__init__()
        self.A = A
        self.B = B
        self.psize = caps_size
        self.mat_dim = int(caps_size ** 0.5)

        self.k = kernel_size
        self.kk = kernel_size ** 2
        self.kkA = self.kk * A

        self.stride = stride
        self.pad = padding

        self.iters = iters

        self.W = nn.Parameter(torch.FloatTensor(self.kkA, B, self.mat_dim, self.mat_dim))
        nn.init.kaiming_uniform_(self.W.data)

        self.beta_u = nn.Parameter(torch.FloatTensor(1, 1, B, 1))
        self.beta_a = nn.Parameter(torch.FloatTensor(1, 1, B))
        nn.init.constant_(self.beta_u, 0)
        nn.init.constant_(self.beta_a, 0)

        self.final_lambda = final_lambda
        self.ln_2pi = math.log(2*math.pi)
        self.initialize()


    def m_step(self, v, a_in, r):
        # v: [b, l, kkA, B, psize]
        # a_in: [b, l, kkA]
        # r: [b, l, kkA, B, 1]
        b, l, _, _, _ = v.shape

        # r: [b, l, kkA, B, 1]
        r = r * a_in.view(b, l, -1, 1, 1)
        # r_sum: [b, l, 1, B, 1]
        r_sum = r.sum(dim=2, keepdim=True)
        # coeff: [b, l, kkA, B, 1]
        coeff = r / (r_sum + eps)

        # mu: [b, l, 1, B, psize]
        mu = torch.sum(coeff * v, dim=2, keepdim=True)
        # sigma_sq: [b, l, 1, B, psize]
        sigma_sq = torch.sum(coeff * (v - mu)**2, dim=2, keepdim=True) + eps

        # [b, l, B, 1]
        r_sum = r_sum.squeeze(2)
        # [b, l, B, psize]
        sigma_sq = sigma_sq.squeeze(2)
        # [1, 1, B, 1] + [b, l, B, psize] * [b, l, B, 1]
        cost_h = (self.beta_u + torch.log(sigma_sq.sqrt())) * r_sum
        # cost_h = (torch.log(sigma_sq.sqrt())) * r_sum

        # [b, l, B]
        a_out = torch.sigmoid(self.lambda_*(self.beta_a - cost_h.sum(dim=3)))
        # a_out = torch.sigmoid(self.lambda_*(-cost_h.sum(dim=3)))

        return a_out, mu, sigma_sq

    def e_step(self, v, a_out, mu, sigma_sq):
        b, l, _ = a_out.shape
        # v: [b, l, kkA, B, psize]
        # a_out: [b, l, B]
        # mu: [b, l, 1, B, psize]
        # sigma_sq: [b, l, B, psize]

        # [b, l, 1, B, psize]
        sigma_sq = sigma_sq.unsqueeze(2)

        ln_p_j = -0.5 * torch.sum(torch.log(sigma_sq*self.ln_2pi), dim=-1) \
                    - torch.sum((v - mu)**2 / (2 * sigma_sq), dim=-1)

        # [b, l, kkA, B]
        ln_ap = ln_p_j + torch.log(a_out.view(b, l, 1, self.B))
        # [b, l, kkA, B]
        r = torch.softmax(ln_ap, dim=-1)
        # [b, l, kkA, B, 1]
        return r.unsqueeze(-1)

    def forward(self, a_in, pose):
        # pose: [batch_size, A, psize]
        # a: [batch_size, A]
        batch_size = a_in.shape[0]

        # a: [b, A, h, w]
        # pose: [b, A*psize, h, w]
        b, _, h, w = a_in.shape

        # [b, A*psize*kk, l]
        pose = F.unfold(pose, self.k, stride=self.stride, padding=self.pad)
        l = pose.shape[-1]
        # [b, A, psize, kk, l]
        pose = pose.view(b, self.A, self.psize, self.kk, l)
        # [b, l, kk, A, psize]
        pose = pose.permute(0, 4, 3, 1, 2).contiguous()
        # [b, l, kkA, psize]
        pose = pose.view(b, l, self.kkA, self.psize)
        # [b, l, kkA, 1, mat_dim, mat_dim]
        pose = pose.view(batch_size, l, self.kkA, self.mat_dim, self.mat_dim).unsqueeze(3)

        # [b, l, kkA, B, mat_dim, mat_dim]
        pose_out = torch.matmul(pose, self.W)

        # [b, l, kkA, B, psize]
        v = pose_out.view(batch_size, l, self.kkA, self.B, -1)

        # [b, kkA, l]
        a_in = F.unfold(a_in, self.k, stride=self.stride, padding=self.pad)
        # [b, A, kk, l]
        a_in = a_in.view(b, self.A, self.kk, l)
        # [b, l, kk, A]
        a_in = a_in.permute(0, 3, 2, 1).contiguous()
        # [b, l, kkA]
        a_in = a_in.view(b, l, self.kkA)

        r = a_in.new_ones(batch_size, l, self.kkA, self.B, 1)
        for i in range(self.iters):
            # this is from open review
            self.lambda_ = self.final_lambda * (1 - 0.95 ** (i+1)) 
            a_out, pose_out, sigma_sq = self.m_step(v, a_in, r)
            if i < self.iters - 1:
                r = self.e_step(v, a_out, pose_out, sigma_sq)

        # [b, l, B*psize]
        pose_out = pose_out.squeeze(2).view(b, l, -1)
        # [b, B*psize, l]
        pose_out = pose_out.transpose(1, 2)
        # [b, B, l]
        a_out = a_out.transpose(1, 2).contiguous()

        oh = ow = math.floor(l**(1/2))

        a_out = a_out.view(b, -1, oh, ow)
        pose_out = pose_out.view(b, -1, oh, ow)

        return a_out, pose_out


    def initialize(self):
        weight_init(self)

class LGE(nn.Module):
    def __init__(self, channels):
        super(LGE, self).__init__()
        self.conv_trans = nn.Conv2d(channels*3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_trans = nn.BatchNorm2d(64)

        self.num_caps = 8
        planes = 16
        last_size = 6

        self.conv_m = nn.Conv2d(64, self.num_caps, kernel_size=5, stride=1, padding=1, bias=False)
        self.conv_pose = nn.Conv2d(64, self.num_caps*16, kernel_size=5, stride=1, padding=1, bias=False)

        self.bn_m = nn.BatchNorm2d(self.num_caps)
        self.bn_pose = nn.BatchNorm2d(self.num_caps*16)

        self.emrouting = EmRouting2d(self.num_caps, self.num_caps, 16, kernel_size=3, stride=2, padding=1)
        self.bn_caps = nn.BatchNorm2d(self.num_caps*planes)


        self.conv_rec = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_rec = nn.BatchNorm2d(64)

        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.fuse1  = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.fuse2  = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.fuse3  = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.initialize()

    def forward(self, input1, input2, input3):

        if input1.size()[2:] != input2.size()[2:]:
            input1 = F.interpolate(input1, size=input2.size()[2:], mode='bilinear')
        if input3.size()[2:] != input2.size()[2:]:
            input3 = F.interpolate(input3, size=input2.size()[2:], mode='bilinear')

        input_fuse = torch.cat((input1, input2, input3), 1)

        # conv
        input_fuse = F.relu(self.bn_trans(self.conv_trans(input_fuse)), inplace=True)

        # primary caps
        m, pose = self.conv_m(input_fuse), self.conv_pose(input_fuse)

        m, pose = torch.sigmoid(self.bn_m(m)), self.bn_pose(pose)

        # caps
        m, pose = self.emrouting(m, pose)
        pose = self.bn_caps(pose)

        # reconstruction
        pose = F.relu(self.bn_rec(self.conv_rec(pose)), inplace=True)


        if pose.size()[2:] != input1.size()[2:]:
            pose1 = F.interpolate(pose, size=input1.size()[2:], mode='bilinear')
        if pose.size()[2:] != input2.size()[2:]:
            pose2 = F.interpolate(pose, size=input2.size()[2:], mode='bilinear')
        if pose.size()[2:] != input3.size()[2:]:
            pose3 = F.interpolate(pose, size=input3.size()[2:], mode='bilinear')

        out1 = torch.cat((input1, pose1),1)
        out2 = torch.cat((input2, pose2),1)
        out3 = torch.cat((input3, pose3),1)

        out1 = F.relu(self.bn1(self.conv1(out1)), inplace=True)
        out2 = F.relu(self.bn2(self.conv2(out2)), inplace=True)
        out3 = F.relu(self.bn3(self.conv3(out3)), inplace=True)

        out1 = F.interpolate(out1,   size=out2.size()[2:], mode='bilinear')
        out2  = self.fuse2(out2*out1) + out2
        out2 = F.interpolate(out2,   size=out3.size()[2:], mode='bilinear')
        out3  = self.fuse3(out3*out2) + out3

        return out1, out2, out3, pose

    def initialize(self):
        weight_init(self)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.initialize()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)

    def initialize(self):
        weight_init(self)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.initialize()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

    def initialize(self):
        weight_init(self)
