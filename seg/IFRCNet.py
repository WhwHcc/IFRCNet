
import torch.nn as nn
import torch
from torch import autograd
from functools import partial
import torch.nn.functional as F
from torchvision import models

#from seg import deform_conv
from torch.autograd import Variable, Function
import torch
from torch import nn
import numpy as np


class DeformConv2D(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, bias=None):
        super(DeformConv2D, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv_kernel = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

    def forward(self, x, offset):
        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        # Change offset's order from [x1, x2, ..., y1, y2, ...] to [x1, y1, x2, y2, ...]
        # Codes below are written to make sure same results of MXNet implementation.
        # You can remove them, and it won't influence the module's performance.
        offsets_index = Variable(torch.cat([torch.arange(0, 2*N, 2), torch.arange(1, 2*N+1, 2)]), requires_grad=False).type_as(x).long()
        offsets_index = offsets_index.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1).expand(*offset.size())
        offset = torch.gather(offset, dim=1, index=offsets_index)
        # ------------------------------------------------------------------------

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = Variable(p.data, requires_grad=False).floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], -1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], -1)

        # (b, h, w, N)
        mask = torch.cat([p[..., :N].lt(self.padding)+p[..., :N].gt(x.size(2)-1-self.padding),
                          p[..., N:].lt(self.padding)+p[..., N:].gt(x.size(3)-1-self.padding)], dim=-1).type_as(p)
        mask = mask.detach()
        floor_p = p - (p - torch.floor(p))
        p = p*(1-mask) + floor_p*mask
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv_kernel(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = np.meshgrid(range(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
                          range(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1), indexing='ij')
        # (2N, 1)
        p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten()))
        p_n = np.reshape(p_n, (1, 2*N, 1, 1))
        p_n = Variable(torch.from_numpy(p_n).type(dtype), requires_grad=False)

        return p_n

    @staticmethod
    def _get_p_0(h, w, N, dtype):
        p_0_x, p_0_y = np.meshgrid(range(1, h+1), range(1, w+1), indexing='ij')
        p_0_x = p_0_x.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0_y = p_0_y.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0 = np.concatenate((p_0_x, p_0_y), axis=1)
        p_0 = Variable(torch.from_numpy(p_0).type(dtype), requires_grad=False)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class DoubleConv2_1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv2_1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, dilation=3, stride=2, padding=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)

        )

    def forward(self, input):
        return self.conv(input)


class DoubleConv2_2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv2_2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, dilation=2, stride=2, padding=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class RCBAM(nn.Module):
    def __init__(self, channel):
        super(RCBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        out = x+out
        return out

class MSAG(nn.Module):
    """
    Multi-scale attention gate
    """
    def __init__(self, channel):
        super(MSAG, self).__init__()
        self.channel = channel
        self.pointwiseConv = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(self.channel),
        )
        self.ordinaryConv = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(self.channel),
        )
        self.dilationConv = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=3, padding=2, stride=1, dilation=2, bias=True),
            nn.BatchNorm2d(self.channel),
        )
        self.voteConv = nn.Sequential(
            nn.Conv2d(self.channel * 3, self.channel, kernel_size=(1, 1)),
            nn.BatchNorm2d(self.channel),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.pointwiseConv(x)
        x2 = self.ordinaryConv(x)
        x3 = self.dilationConv(x)
        _x = self.relu(torch.cat((x1, x2, x3), dim=1))
        _x = self.voteConv(_x)
        x = x + x * _x
        return x
class RCANet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(RCANet, self).__init__()

        self.conv1 = DoubleConv2_1(in_ch, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv2_2(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = Conv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = Conv(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = Conv(256, 512)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.rcbam1 = RCBAM(32)
        self.rcbam2 = RCBAM(64)
        self.rcbam3 = RCBAM(128)
        self.rcbam4 = RCBAM(256)
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 3)
        )

    def forward(self, x):
        # print(x.shape)
        c1 = self.conv1(x)
        m1 = self.rcbam1(c1)
        p1 = self.pool1(m1)
        # print(p1.shape)
        c2 = self.conv2(p1)
        m2 = self.rcbam2(c2)
        p2 = self.pool2(m2)
        # print(p2.shape)
        c3 = self.conv3(p2)
        m3 = self.rcbam3(c3)
        p3 = self.pool3(m3)
        # print(p3.shape)
        c4 = self.conv4(p3)
        m4 = self.rcbam4(c4)
        p4 = self.pool4(m4)
        c5 = self.conv5(p4)
        # print(p4.shape)
        x = self.avgpool(c5)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class RCAUNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(RCAUNet, self).__init__()

        self.conv1 = DoubleConv(in_ch, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(256, 512)
        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6 = DoubleConv(512, 256)
        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7 = DoubleConv(256, 128)
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8 = DoubleConv(128, 64)
        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9 = DoubleConv(64, 32)
        self.conv10 = nn.Conv2d(32, out_ch, 1)

        self.rcbam1 = RCBAM(32)
        self.rcbam2 = RCBAM(64)
        self.rcbam3 = RCBAM(128)
        self.rcbam4 = RCBAM(256)
        self.msag4 = MSAG(256)
        self.msag3 = MSAG(128)
        self.msag2 = MSAG(64)
        self.msag1 = MSAG(32)

    def forward(self, x):
        # print(x.shape)
        c1 = self.conv1(x)
        m1 = self.rcbam1(c1)
        p1 = self.pool1(m1)
        # print(p1.shape)
        c2 = self.conv2(p1)
        m2 = self.rcbam2(c2)
        p2 = self.pool2(m2)
        # print(p2.shape)
        c3 = self.conv3(p2)
        m3 = self.rcbam3(c3)
        p3 = self.pool3(m3)
        # print(p3.shape)
        c4 = self.conv4(p3)
        m4 = self.rcbam4(c4)
        p4 = self.pool4(m4)
        # print(p4.shape)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        t4 = self.msag4(c4)
        merge6 = torch.cat((up_6, t4), dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        t3 = self.msag3(c3)
        merge7 = torch.cat((up_7, t3), dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        t2 = self.msag2(c2)
        merge8 = torch.cat((up_8, t2), dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        t1 = self.msag1(c1)
        merge9 = torch.cat((up_9, t1), dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        out = nn.Sigmoid()(c10)
        return out

class IFRCNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(IFRCNet, self).__init__()

        self.conv1 = DoubleConv(in_ch, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(256, 512)
        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6 = DoubleConv(512, 256)
        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7 = DoubleConv(256, 128)
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8 = DoubleConv(128, 64)
        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9 = DoubleConv(64, 32)
        self.conv10 = nn.Conv2d(32, out_ch, 1)
        #
        # for p in self.parameters():
        #     p.requires_grad = False
        self.rcbam1 = RCBAM(32)
        self.rcbam2 = RCBAM(64)
        self.rcbam3 = RCBAM(128)
        self.rcbam4 = RCBAM(256)
        self.msag4 = MSAG(256)
        self.msag3 = MSAG(128)
        self.msag2 = MSAG(64)
        self.msag1 = MSAG(32)
        self.conv1_1 = DoubleConv2_1(4, 32)
        self.pool1_1 = nn.MaxPool2d(2)
        self.Dconv1 = DoubleConv2_2(64, 32)
        self.conv2_2 = Conv(32, 64)
        self.pool2_2 = nn.MaxPool2d(2)
        self.Dconv2 = DoubleConv(128, 64)
        self.conv3_3 = Conv(64, 128)
        self.pool3_3 = nn.MaxPool2d(2)
        self.Dconv3 = DoubleConv(256, 128)
        self.conv4_4 = Conv(128, 256)
        self.pool4_4 = nn.MaxPool2d(2)
        self.Dconv4 = DoubleConv(512, 256)
        self.conv5_5 = Conv(256, 512)
        self.pool5_5 = nn.MaxPool2d(2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 3)
        )

    def forward(self, x):
        original=x
        c1 = self.conv1(x)
        m1 = self.rcbam1(c1)
        p1 = self.pool1(m1)
        # print(p1.shape)
        c2 = self.conv2(p1)
        m2 = self.rcbam2(c2)
        p2 = self.pool2(m2)
        # print(p2.shape)
        c3 = self.conv3(p2)
        m3 = self.rcbam3(c3)
        p3 = self.pool3(m3)
        # print(p3.shape)
        c4 = self.conv4(p3)
        m4 = self.rcbam4(c4)
        p4 = self.pool4(m4)
        # print(p4.shape)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        t4 = self.msag4(m4)
        merge6 = torch.cat((up_6, t4), dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        t3 = self.msag3(m3)
        merge7 = torch.cat((up_7, t3), dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        t2 = self.msag2(m2)
        merge8 = torch.cat((up_8, t2), dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        t1 = self.msag1(m1)
        merge9 = torch.cat((up_9, t1), dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        out = nn.Sigmoid()(c10)


        merge1 = torch.cat([original, out], dim=1)
        c1_1 = self.conv1_1(merge1)
        m1_1 = self.rcbam1(c1_1)
        merge2 = torch.cat([m1_1, p1], dim=1)
        d1_1 = self.Dconv1(merge2)
        c2_2 = self.conv2_2(d1_1)
        m2_2 = self.rcbam2(c2_2)
        print(m2_2.shape)
        print(p2.shape)
        merge3 = torch.cat([m2_2, p2], dim=1)
        d2_2 = self.Dconv2(merge3)
        c3_3 = self.conv3_3(d2_2)
        m3_3 = self.rcbam3(c3_3)
        merge4 = torch.cat([m3_3, m3], dim=1)
        d3_3 = self.Dconv3(merge4)
        c4_4 = self.conv4_4(d3_3)
        m4_4 = self.rcbam4(c4_4)
        p4_4 = self.pool4_4(m4_4)
        merge5 = torch.cat([p4_4, m4], dim=1)
        d4_4 = self.Dconv4(merge5)
        c5_5 = self.conv5_5(d4_4)
        # p5_5= self.pool5_5(c5_5)
        x = self.avgpool(c5_5)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
class FRCNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(FRCNet, self).__init__()

        self.conv1 = DoubleConv(in_ch, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(256, 512)
        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6 = DoubleConv(512, 256)
        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7 = DoubleConv(256, 128)
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8 = DoubleConv(128, 64)
        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9 = DoubleConv(64, 32)
        self.conv10 = nn.Conv2d(32, out_ch, 1)
        #
        # for p in self.parameters():
        #     p.requires_grad = False
        self.rcbam1 = RCBAM(32)
        self.rcbam2 = RCBAM(64)
        self.rcbam3 = RCBAM(128)
        self.rcbam4 = RCBAM(256)
        self.conv1_1 = DoubleConv2_1(4, 32)
        self.pool1_1 = nn.MaxPool2d(2)
        # self.Dconv1 = DoubleConv(64, 32)
        self.conv2_2 = DoubleConv2_2(32, 64)
        self.pool2_2 = nn.MaxPool2d(2)
        # self.Dconv2 = DoubleConv(128, 64)
        self.conv3_3 = Conv(64, 128)
        self.pool3_3 = nn.MaxPool2d(2)
        # self.Dconv3 = DoubleConv(256, 128)
        self.conv4_4 = Conv(128, 256)
        self.pool4_4 = nn.MaxPool2d(2)
        # self.Dconv4 = DoubleConv(512, 256)
        self.conv5_5 = Conv(256, 512)
        self.pool5_5 = nn.MaxPool2d(2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 3)
        )

    def forward(self, x):
        original=x
        c1 = self.conv1(x)
        m1 = self.rcbam1(c1)
        p1 = self.pool1(m1)
        # print(p1.shape)
        c2 = self.conv2(p1)
        m2 = self.rcbam2(c2)
        p2 = self.pool2(m2)
        # print(p2.shape)
        c3 = self.conv3(p2)
        m3 = self.rcbam3(c3)
        p3 = self.pool3(m3)
        # print(p3.shape)
        c4 = self.conv4(p3)
        m4 = self.rcbam4(c4)
        p4 = self.pool4(m4)
        # print(p4.shape)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat((up_6, c4), dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat((up_7, c3), dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat((up_8, c2), dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat((up_9, c1), dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        out = nn.Sigmoid()(c10)


        merge1 = torch.cat([original, out], dim=1)
        c1_1 = self.conv1_1(merge1)
        m1_1 = self.rcbam1(c1_1)
        # print(m1_1.shape)
        # merge2 = torch.cat([m1_1, p1], dim=1)
        # d1_1 = self.Dconv1(merge2)
        c2_2 = self.conv2_2(m1_1)
        m2_2 = self.rcbam2(c2_2)
        # print(m2_2.shape)
        # merge3 = torch.cat([m2_2, p2], dim=1)
        # d2_2 = self.Dconv2(merge3)
        c3_3 = self.conv3_3(m2_2)
        m3_3 = self.rcbam3(c3_3)
        # print(m3_3.shape)
        # p3_3 = self.pool3_3(m3_3)
        # merge4 = torch.cat([p3_3, p3], dim=1)
        # d3_3 = self.Dconv3(c3_3)
        c4_4 = self.conv4_4(m3_3)
        m4_4 = self.rcbam4(c4_4)
        # print(m4_4.shape)
        p4_4 = self.pool4_4(m4_4)
        # merge5 = torch.cat([p4_4, p4], dim=1)
        # d4_4 = self.Dconv4(merge5)
        c5_5 = self.conv5_5(p4_4)
        # p5_5= self.pool5_5(c5_5)
        x = self.avgpool(c5_5)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class IRCNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(IRCNet, self).__init__()

        self.conv1 = DoubleConv(in_ch, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(256, 512)
        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6 = DoubleConv(512, 256)
        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7 = DoubleConv(256, 128)
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8 = DoubleConv(128, 64)
        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9 = DoubleConv(64, 32)
        self.conv10 = nn.Conv2d(32, out_ch, 1)
        #
        # for p in self.parameters():
        #     p.requires_grad = False
        self.rcbam1 = RCBAM(32)
        self.rcbam2 = RCBAM(64)
        self.rcbam3 = RCBAM(128)
        self.rcbam4 = RCBAM(256)
        self.conv1_1 = DoubleConv2_1(3, 32)
        self.pool1_1 = nn.MaxPool2d(2)
        self.Dconv1 = DoubleConv(64, 32)
        self.conv2_2 = DoubleConv2_2(32, 64)
        self.pool2_2 = nn.MaxPool2d(2)
        self.Dconv2 = DoubleConv(128, 64)
        self.conv3_3 = Conv(64, 128)
        self.pool3_3 = nn.MaxPool2d(2)
        self.Dconv3 = DoubleConv(256, 128)
        self.conv4_4 = Conv(128, 256)
        self.pool4_4 = nn.MaxPool2d(2)
        self.Dconv4 = DoubleConv(512, 256)
        self.conv5_5 = Conv(256, 512)
        self.pool5_5 = nn.MaxPool2d(2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 3)
        )

    def forward(self, x):
        original=x
        c1 = self.conv1(x)
        m1 = self.rcbam1(c1)
        p1 = self.pool1(m1)
        # print(p1.shape)
        c2 = self.conv2(p1)
        m2 = self.rcbam2(c2)
        p2 = self.pool2(m2)
        # print(p2.shape)
        c3 = self.conv3(p2)
        m3 = self.rcbam3(c3)
        p3 = self.pool3(m3)
        # print(p3.shape)
        c4 = self.conv4(p3)
        m4 = self.rcbam4(c4)
        p4 = self.pool4(m4)
        # print(p4.shape)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat((up_6, c4), dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat((up_7, c3), dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat((up_8, c2), dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat((up_9, c1), dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        out = nn.Sigmoid()(c10)


        # merge1 = torch.cat([original, out], dim=1)
        c1_1 = self.conv1_1(original)
        m1_1 = self.rcbam1(c1_1)
        merge2 = torch.cat([m1_1, p1], dim=1)
        d1_1 = self.Dconv1(merge2)
        c2_2 = self.conv2_2(d1_1)
        m2_2 = self.rcbam2(c2_2)
        merge3 = torch.cat([m2_2, p2], dim=1)
        d2_2 = self.Dconv2(merge3)
        c3_3 = self.conv3_3(d2_2)
        m3_3 = self.rcbam3(c3_3)
        p3_3 = self.pool3_3(m3_3)
        merge4 = torch.cat([p3_3, p3], dim=1)
        d3_3 = self.Dconv3(merge4)
        c4_4 = self.conv4_4(d3_3)
        m4_4 = self.rcbam4(c4_4)
        p4_4 = self.pool4_4(m4_4)
        merge5 = torch.cat([p4_4, p4], dim=1)
        d4_4 = self.Dconv4(merge5)
        c5_5 = self.conv5_5(d4_4)
        # p5_5= self.pool5_5(c5_5)
        x = self.avgpool(c5_5)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
