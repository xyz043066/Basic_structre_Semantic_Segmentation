# from model.build_contextpath import build_contextpath
import warnings

import torch
from torch import nn
import torch.nn.functional as F

warnings.filterwarnings(action='ignore')

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))

class Spatial_path(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convblock1 = ConvBlock(in_channels=3, out_channels=64)
        self.convblock2 = ConvBlock(in_channels=64, out_channels=128)
        self.convblock3 = ConvBlock(in_channels=128, out_channels=256)

    def forward(self, input):
        x = self.convblock1(input)
        x = self.convblock2(x)
        x = self.convblock3(x)
        return x

class AttentionRefinementModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, input):
        # global average pooling
        x = self.avgpool(input)
        assert self.in_channels == x.size(1), 'in_channels and out_channels should all be {}'.format(x.size(1))
        x = self.conv(x)
        # x = self.sigmoid(self.bn(x))
        x = self.sigmoid(x)
        # channels of input and x should be same
        x = torch.mul(input, x)
        return x


class AttentionFusionModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.convblock_1 = ConvBlock(in_channels=self.in_channels, out_channels=256, kernel_size=1, stride=1, padding=0)
        # self.sigmoid = nn.Sigmoid()
        # self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # self.softmax = nn.Softmax()
        # self.avgpool = nn.AdaptiveAvgPool2d(output=(1, 1))
        self.convblock_2 = ConvBlock(in_channels=512, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, input, low_feature_map):
        low_feature_map = self.convblock_1(low_feature_map)
        # feasure = self.avgpool(low_feature_map)
        # feasure = self.softmax(feasure)
        # feasure = torch.mul(feasure, low_feature_map)
        x = F.interpolate(input, size=low_feature_map.size()[2:], mode='bilinear', align_corners=True)
        assert x.size()[1] == 256, "The channels of x must be 256"
        x = torch.cat((x, low_feature_map), 1) # the channels between x and feature must be same
        x = self.convblock_2(x)

        return x

class FeatureFusionModule(torch.nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()
        # self.in_channels = input_1.channels + input_2.channels
        # resnet101 3328 = 256(from context path) + 1024(from spatial path) + 2048(from spatial path)
        # resnet18  1024 = 256(from context path) + 256(from spatial path) + 512(from spatial path)
        self.in_channels = in_channels

        self.convblock = ConvBlock(in_channels=self.in_channels, out_channels=num_classes, stride=1)
        self.conv1 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))


    def forward(self, input_1, input_2):
        # print("input_1 size = ", input_1.size(), "input_2 = ", input_2.size())
        # exit()
        x = torch.cat((input_1, input_2), dim=1)
        # print("input_1 size = ", input_1.size(), "input_2 = ", input_2.size(), "x = ", x.size())
        # exit()
        assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.convblock(x)
        x = self.avgpool(feature)

        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = torch.mul(feature, x)
        x = torch.add(x, feature)
        return x

class SFAM(torch.nn.Module): # Scale-Feature Attention Module
    def __init__(self, in_channels_1, in_channels_2):  # in_channels_1 => high  in_channels_2 => low
        super().__init__()
        self.conv1 = ConvBlock(in_channels=in_channels_1, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBlock(in_channels=in_channels_2, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.conv3 = ConvBlock(in_channels=in_channels_2, out_channels=256, kernel_size=1, stride=1, padding=0)
        # self.conv1 = nn.Conv2d(in_channels=in_channels_1, out_channels=256, kernel_size=1, stride=1, padding=0)
        # self.conv2 = nn.Conv2d(in_channels=in_channels_2, out_channels=256, kernel_size=1, stride=1, padding=0)
        # self.conv3 = nn.Conv2d(in_channels=in_channels_2, out_channels=256, kernel_size=1, stride=1, padding=0)
        # self.conv = ConvBlock(in_channels=in_channels_1+256, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax()

    def forward(self, input_low, input_high):
        input_high = F.interpolate(input_high, size=input_low.size()[2:], mode='bilinear', align_corners=True)
        x_h = self.conv1(input_high)
        x_l = self.conv2(input_low)
        assert x_l.size() == x_h.size(), "The size between low and high feature maps must be same!"
        # x = self.softmax(x_h*x_l)
        x = self.softmax(torch.mul(x_h, x_l))
        x_l = self.conv3(input_low)
        # x = x*x_l
        x = torch.mul(x, x_l)
        x = torch.add(x, x_l)
        # x = torch.cat((x, x_l), dim=1)
        # x = self.conv(x)
        return x

class SLAM(torch.nn.Module): # Scale-Layer Attention Module
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(channels)
        self.sigmoid = nn.Sigmoid()
    def forward(self, input):
        x  = self.sigmoid(self.bn(self.conv(input)))
        assert x.size() == input.size(), "The size between input and x must be same!"
        # x = x*input
        x = torch.mul(x, input)
        # x = x + input
        x = torch.add(x, input)
        return x

class CAA(nn.Module): # Class Augmented Attention
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.C = out_channels
        self.N = 6
        self.C_1 = 256
        self.convblock_1 = ConvBlock(in_channels=in_channels, out_channels=self.C, kernel_size=3, stride=1, padding=1) # C = out_channels
        self.convblock_2 = ConvBlock(in_channels=out_channels, out_channels=self.N, kernel_size=1, stride=1, padding=0)   # N = 6
        self.convblock_3 = ConvBlock(in_channels=out_channels, out_channels=self.C_1, kernel_size=1, stride=1, padding=0) # C' = 256
        self.softmax_1 = nn.Softmax(dim=1)
        self.softmax_2 = nn.Softmax(dim=2)
        self.convblock_4 = ConvBlock(in_channels=self.C_1, out_channels=self.C, kernel_size=1, stride=1, padding=0)
        self.convblock_5 = ConvBlock(in_channels=2*out_channels, out_channels=self.C, kernel_size=1, stride=1, padding=0)
        self.CCA = CCA(in_channels=self.N, alpha=150)

    def forward(self, input):
        input = self.convblock_1(input) # [B, C, H, W]
        b, c, h, w = input.size()
        p = self.softmax_1(self.convblock_2(input))  # [B, N, H, W]
        A = p.view(b, self.N, -1).permute(0, 2, 1).contiguous() # [B, H*W, N]
        x = self.convblock_3(input) # [B, C', H, W]
        x = x.view(b, self.C_1, -1) # [B, C', H*W]
        A = self.softmax_2(torch.matmul(x, A)) # [B, C', N]
        # Add the CCA Module
        A = self.CCA(input=p, affinity=A)
        x = torch.matmul(A, p.view(b, self.N, -1)).reshape(b, self.C_1, h, w) # [B, C', H, W]
        x = self.convblock_4(x) # [B, C, H, W]
        x = torch.cat([x, input], 1)  # [B, 2C, H, W]
        x = self.convblock_5(x) # [B, C, H, W]

        return x

class CCA(nn.Module):
    def __init__(self, in_channels, alpha):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.layers_1 = nn.Linear(in_features=in_channels, out_features=alpha*in_channels)
        self.layers_2 = nn.Linear(in_features=alpha*in_channels, out_features=in_channels)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, input, affinity): # size: input=>[B, N, H, W] affinity=>[B, C', N]
        x = self.avgpool(input) # [B, N, 1, 1]
        x = x.squeeze(3).permute(0, 2, 1).contiguous() # [B, 1, N]
        x = self.layers_1(x) # [B, 1, alpha*N]
        x = self.layers_2(x) # [B, 1, N]
        x = torch.mul(affinity, x) # [B, C', N]
        x = x + affinity
        x = self.softmax(x)
        return x

class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, sample,  output_channels=512):
        super().__init__()
        self.sample = sample
        self.convblock_1 = ConvBlock(in_channels=2048, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.convblock_2 = ConvBlock(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.convblock_3 = ConvBlock(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.convblock_4 = ConvBlock(in_channels=512*3, out_channels=output_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, high_level_feat, low_level_feat_1, low_level_feat_2):
        high_level_feat = self.convblock_1(high_level_feat)   # output stride : 32
        low_level_feat_1 = self.convblock_2(low_level_feat_1) # output stride : 16
        low_level_feat_2 = self.convblock_3(low_level_feat_2)  # output stride : 8
        if self.sample == "down":
            low_level_feat_1 = F.interpolate(low_level_feat_1, size=(high_level_feat.size()[2], high_level_feat.size()[3]), mode='bilinear', align_corners=True)
            low_level_feat_2 = F.interpolate(low_level_feat_2, size=(high_level_feat.size()[2], high_level_feat.size()[3]), mode='bilinear', align_corners=True)
        elif self.sample == "up":
            low_level_feat_1 = F.interpolate(low_level_feat_1, size=(low_level_feat_2.size()[2], low_level_feat_2.size()[3]), mode='bilinear', align_corners=True)
            high_level_feat = F.interpolate(high_level_feat, size=(low_level_feat_2.size()[2], low_level_feat_2.size()[3]), mode='bilinear', align_corners=True)

        output = torch.cat([high_level_feat, low_level_feat_1, low_level_feat_2], 1)
        output = self.convblock_4(output)
        return output
# class BiSeNet(torch.nn.Module):
#     def __init__(self, num_classes, context_path):
#         super().__init__()
#         # build spatial path
#         self.saptial_path = Spatial_path()

#         # build context path
#         self.context_path = build_contextpath(name=context_path)

#         # build attention refinement module  for resnet 101
#         if context_path == 'resnet101':
#             self.attention_refinement_module1 = AttentionRefinementModule(1024, 1024)
#             self.attention_refinement_module2 = AttentionRefinementModule(2048, 2048)
#             # supervision block
#             self.supervision1 = nn.Conv2d(in_channels=1024, out_channels=num_classes, kernel_size=1)
#             self.supervision2 = nn.Conv2d(in_channels=2048, out_channels=num_classes, kernel_size=1)
#             # build feature fusion module
#             self.feature_fusion_module = FeatureFusionModule(num_classes, 3328)

#         elif context_path == 'resnet18':
#             # build attention refinement module  for resnet 18
#             self.attention_refinement_module1 = AttentionRefinementModule(256, 256)
#             self.attention_refinement_module2 = AttentionRefinementModule(512, 512)
#             # supervision block
#             self.supervision1 = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1)
#             self.supervision2 = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1)
#             # build feature fusion module
#             self.feature_fusion_module = FeatureFusionModule(num_classes, 1024)
#         else:
#             print('Error: unspport context_path network \n')

#         # build final convolution
#         self.conv = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1)

#         self.init_weight()

#         self.mul_lr = []
#         self.mul_lr.append(self.saptial_path)
#         self.mul_lr.append(self.attention_refinement_module1)
#         self.mul_lr.append(self.attention_refinement_module2)
#         self.mul_lr.append(self.supervision1)
#         self.mul_lr.append(self.supervision2)
#         self.mul_lr.append(self.feature_fusion_module)
#         self.mul_lr.append(self.conv)

#     def init_weight(self):
#         for name, m in self.named_modules():
#             if 'context_path' not in name:
#                 if isinstance(m, nn.Conv2d):
#                     nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
#                 elif isinstance(m, nn.BatchNorm2d):
#                     m.eps = 1e-5
#                     m.momentum = 0.1
#                     nn.init.constant_(m.weight, 1)
#                     nn.init.constant_(m.bias, 0)

#     def forward(self, input):
#         # output of spatial path
#         sx = self.saptial_path(input)

#         # output of context path
#         cx1, cx2, tail = self.context_path(input)
#         cx1 = self.attention_refinement_module1(cx1)
#         cx2 = self.attention_refinement_module2(cx2)
#         cx2 = torch.mul(cx2, tail)
#         # upsampling
#         cx1 = torch.nn.functional.interpolate(cx1, size=sx.size()[-2:], mode='bilinear')
#         cx2 = torch.nn.functional.interpolate(cx2, size=sx.size()[-2:], mode='bilinear')
#         cx = torch.cat((cx1, cx2), dim=1)

#         if self.training == True:
#             cx1_sup = self.supervision1(cx1)
#             cx2_sup = self.supervision2(cx2)
#             cx1_sup = torch.nn.functional.interpolate(cx1_sup, size=input.size()[-2:], mode='bilinear')
#             cx2_sup = torch.nn.functional.interpolate(cx2_sup, size=input.size()[-2:], mode='bilinear')

#         # output of feature fusion module
#         result = self.feature_fusion_module(sx, cx)

#         # upsampling
#         result = torch.nn.functional.interpolate(result, scale_factor=8, mode='bilinear')
#         result = self.conv(result)

#         if self.training == True:
#             return result, cx1_sup, cx2_sup

#         return result


# if __name__ == '__main__':
#     import os
#     os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#     model = BiSeNet(32, 'resnet18')
#     # model = nn.DataParallel(model)
#
#     model = model.cuda()
#     x = torch.rand(2, 3, 256, 256)
#     record = model.parameters()
#     # for key, params in model.named_parameters():
#     #     if 'bn' in key:
#     #         params.requires_grad = False
#     # params_list = []
#     # for module in model.mul_lr:
#     #     params_list = group_weight(params_list, module, nn.BatchNorm2d, 10)
#     # params_list = group_weight(params_list, model.context_path, torch.nn.BatchNorm2d, 1)
#
#     print(model.parameters())