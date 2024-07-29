import torch
from model.utils.AF.Fsmish import smish as Fsmish
from model.utils.AF.Xsmish import Smish
from .pvtv2_encoder import pvt_v2_b4
from .LFSS_modules import MFE, LGE, SSE
import torch.nn as nn
import torch.nn.functional as F
from .LFSS_modules import ChannelAttention, SpatialAttention
from model.pretained.convnext import ConvNeXt

def weight_init_backbone(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (
                nn.ReLU, nn.Sigmoid, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool1d, nn.Sigmoid, nn.Identity)):
            pass
        else:
            m.initialize()


def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            # nn.init.xavier_normal_(m.weight, gain=1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            # nn.init.xavier_normal_(m.weight, gain=1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (
                nn.ReLU, nn.Sigmoid, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool1d, nn.Sigmoid, nn.Identity)):
            pass


class DoubleFusion(nn.Module):
    def __init__(self):
        super(DoubleFusion, self).__init__()
        self.DWconv1 = nn.Conv2d(192, 64, kernel_size=3, stride=1, padding=1)
        self.PSconv1 = nn.PixelShuffle(1)

        self.DWconv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.AF = Smish()  # Replace this with your activation function

    def forward(self, x1, x2, x3):
        if x1.size()[2:] != x2.size()[2:]:
            x1 = F.interpolate(x1, size=x2.size()[2:], mode='bilinear')
        if x3.size()[2:] != x2.size()[2:]:
            x3 = F.interpolate(x3, size=x2.size()[2:], mode='bilinear')

        x_cat = torch.cat((x1, x2, x3), dim=1)

        attn = self.PSconv1(self.DWconv1(self.AF(x_cat)))

        attn2 = self.PSconv1(self.DWconv2(self.AF(attn)))

        return Fsmish(((attn2 + attn)).squeeze(1))


class Classifier_Module(nn.Module):
    def __init__(self, dilation_series, padding_series, NoLabels, input_channel):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(input_channel, NoLabels, kernel_size=3, stride=1, padding=padding, dilation=dilation,
                          bias=True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes)
        )

    def forward(self, x):
        x = self.conv_bn(x)
        return x

class CSA(nn.Module):
    def __init__(self, x):
        super(CSA, self).__init__()

        self.atten_depth_channel = ChannelAttention(x)
        self.atten_depth_spatial = SpatialAttention()

    def forward(self, x_depth):
        x_depth = x_depth.mul(self.atten_depth_channel(x_depth))
        x_depth = x_depth.mul(self.atten_depth_spatial(x_depth))

        return x_depth

class LFSS(torch.nn.Module):
    def __init__(self, cfg):
        super(LFSS, self).__init__()
        self.cfg = cfg

        self.encoder = pvt_v2_b4()
        pretrained_dict = torch.load('../pretained/pvt_v2_b4.pth')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(pretrained_dict)

        self.depth = pvt_v2_b4()
        pretrained_dict = torch.load('../pretained/pvt_v2_b4.pth')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.depth.state_dict()}
        self.depth.load_state_dict(pretrained_dict)

        self.conv11 = nn.Conv2d(128, 64, 1, 1)
        self.conv1 = nn.Conv2d(128, 64, 1, 1)
        self.conv2 = nn.Conv2d(256, 128, 1, 1)
        self.conv3 = nn.Conv2d(640, 320, 1, 1)
        self.conv4 = nn.Conv2d(1024, 512, 1, 1)

        self.conv11 = nn.Conv2d(128, 64, 1, 1)
        self.conv22 = nn.Conv2d(256, 128, 1, 1)
        self.conv33 = nn.Conv2d(640, 320, 1, 1)
        self.conv44 = nn.Conv2d(1024, 512, 1, 1)

        self.conv111 = nn.Conv2d(128, 64, 1, 1)

        self.mfe1 = MFE(512, 64)
        self.mfe2 = MFE(320, 64)
        self.mfe3 = MFE(128, 64)
        self.mfe4 = MFE(64, 64)

        self.edg_weight = nn.Parameter(torch.tensor(0.2))
        self.double_weight = nn.Parameter(torch.tensor(0.5))

        self.convnext = ConvNeXt(3, depths=[3, 3, 9, 3], dims=[64, 128, 320, 512])
        self.convnext_depth = ConvNeXt(3, depths=[3, 3, 9, 3], dims=[64, 128, 320, 512])

        self.fuse = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.doulefusion = DoubleFusion()
        self.csa1 = CSA(64)
        self.csa2 = CSA(128)
        self.csa3 = CSA(320)
        self.csa4 = CSA(512)

        self.sse1 = SSE()
        self.sse2 = SSE()
        self.sse3 = SSE()
        self.sse4 = SSE()
        self.lge = LGE(64)

        self.fuse = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.predtrans1 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.predtrans2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.predtrans3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.predtrans4 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.predtrans5 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        self.initialize()

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, x, x_depth, shape=None, name=None, is_depth=False):
        if (shape == None):
            shape = x.size()[2:]
        out = []
        cnexts = self.convnext(x)
        v1 = cnexts[0]
        v2 = cnexts[1]
        v3 = cnexts[2]
        v4 = cnexts[3]

        cnexts_depth = self.convnext_depth(x_depth)

        v1_depth = cnexts_depth[0]
        v2_depth = cnexts_depth[1]
        v3_depth = cnexts_depth[2]
        v4_depth = cnexts_depth[3]

        x = self.encoder(x, 1)

        x_depth = self.depth(x_depth, 1)
        x_depth = self.conv1(torch.cat([x_depth, v1_depth], dim=1))

        temp = self.csa1(x_depth)

        x += temp
        out.append(x)

        x = self.encoder(x, 2)
        x_depth = self.depth(x_depth, 2)

        x_depth = self.conv2(torch.cat([x_depth, v2_depth], dim=1))

        temp = self.csa2(x_depth)
        x += temp

        out.append(x)

        x = self.encoder(x, 3)
        x_depth = self.depth(x_depth, 3)

        x_depth = self.conv3(torch.cat([x_depth, v3_depth], dim=1))

        temp = self.csa3(x_depth)
        x += temp
        out.append(x)

        x = self.encoder(x, 4)
        x_depth = self.depth(x_depth, 4)

        x_depth = self.conv4(torch.cat([x_depth, v4_depth], dim=1))

        temp = self.csa4(x_depth)
        x += temp

        out.append(x)

        x1 = out[3]
        x2 = out[2]
        x3 = out[1]
        x4 = out[0]

        x1 = self.conv44(torch.cat([x1, v4], dim=1))
        x2 = self.conv33(torch.cat([x2, v3], dim=1))
        x3 = self.conv22(torch.cat([x3, v2], dim=1))
        x4 = self.conv11(torch.cat([x4, v1], dim=1))

        x1 = self.mfe1(x1)
        x2 = self.mfe2(x2)
        x3 = self.mfe3(x3)
        x4 = self.mfe4(x4)

        x1 = self.sse1(in1=x1, in2=x2)
        x2 = self.sse2(in1=x2, in2=x1, in3=x3)
        x3 = self.sse3(in1=x3, in2=x2, in3=x4)
        x4 = self.sse4(in1=x4, in2=x3)

        x1, x2, x3, pose = self.lge(x1, x2, x3)

        x3 = F.interpolate(x3, size=x4.size()[2:], mode='bilinear')
        x4 = self.fuse(x4 * x3) + x4

        fuse_fea = self.doulefusion(x1, x2, x3)
        fuse_fea = F.interpolate(fuse_fea, scale_factor=0.5, mode='bilinear', align_corners=False)
        fuse_fea = F.interpolate(fuse_fea, (11, 11), mode='bilinear')
        final_fea = torch.cat([pose, fuse_fea * self.double_weight], dim=1)
        final_fea = self.conv111(final_fea)

        pred1 = F.interpolate(self.predtrans1(x1), size=shape, mode='bilinear')
        pred2 = F.interpolate(self.predtrans2(x2), size=shape, mode='bilinear')
        pred3 = F.interpolate(self.predtrans3(x3), size=shape, mode='bilinear')
        pred4 = F.interpolate(self.predtrans4(x4), size=shape, mode='bilinear')
        pose = F.interpolate(self.predtrans5(final_fea), size=shape, mode='bilinear')

        return pred1, pred2, pred3, pred4, pose

    def initialize(self):
        if self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot))
        else:
            weight_init(self)