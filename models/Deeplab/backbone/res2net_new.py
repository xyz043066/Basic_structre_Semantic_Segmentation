import torch.nn as nn
import math
import os
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
from models.Deeplab.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.Deeplab.spatial_path import *
__all__ = ['Res2Net', 'res2net50']

model_urls = {
    'res2net50_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_4s-06e79181.pth',
    'res2net50_48w_2s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_48w_2s-afed724a.pth',
    'res2net50_14w_8s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_14w_8s-6527dddc.pth',
    'res2net50_26w_6s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_6s-19041792.pth',
    'res2net50_26w_8s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_8s-2c7c9f12.pth',
    'res2net101_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_26w_4s-02a759a1.pth',
}

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.se = SELayer(channel=planes * self.expansion, reduction=16)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        # SE Block
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)


        out += residual
        out = self.relu(out)

        return out


class Res2Net(nn.Module):

    def __init__(self, block, layers, baseWidth=26, scale=4, pretrained=True, model_type='res2net101_26w_4s'):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.model_type = model_type
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(64)
        # self.relu2 = nn.ReLU(inplace=True)
        # self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(128)
        # self.relu3 = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.convblock_1 = ConvBlock(in_channels=256 + 512, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.convblock_2 = ConvBlock(in_channels=512 + 1024, out_channels=1024, kernel_size=1, stride=1, padding=0)
        self.convblock_3 = ConvBlock(in_channels=1024 + 2048, out_channels=2048, kernel_size=1, stride=1, padding=0)
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weight()

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        #########################
        ####### First conv ######
        #########################
        x = self.relu1(self.bn1(self.conv1(input)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        # x = self.conv1(input)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.relu11(self.bn11(self.conv11(input)))
        # x = self.relu22(self.bn22(self.conv22(x)))

        low_level_feat_0 = x

        #########################
        #######  stage 1 ########
        #########################
        x = self.layer1(x)
        low_level_feat_1 = x

        #########################
        #######  stage 2 ########
        #########################
        x = self.layer2(x)
        # low_level_feat_2 = x
        low_level_feat_2 = F.interpolate(low_level_feat_0, size=x.size()[2:], align_corners=True)
        low_level_feat_2 = torch.cat([low_level_feat_2, x], dim=1)
        low_level_feat_2 = self.convblock_1(low_level_feat_2)
        x = low_level_feat_2

        #########################
        #######  stage 3 ########
        #########################

        x = self.layer3(x)
        low_level_feat_3 = F.interpolate(low_level_feat_1, size=x.size()[2:], align_corners=True)
        low_level_feat_3 = torch.cat([low_level_feat_3, x], dim=1)
        low_level_feat_3 = self.convblock_2(low_level_feat_3)
        x = low_level_feat_3

        #########################
        #######  stage 4 ########
        #########################

        x = self.layer4(x)
        low_level_feat_4 = F.interpolate(low_level_feat_2, size=x.size()[2:], align_corners=True)
        low_level_feat_4 = torch.cat([low_level_feat_4, x], dim=1)
        low_level_feat_4 = self.convblock_3(low_level_feat_4)
        x = low_level_feat_4

        return x, low_level_feat_1, low_level_feat_2, low_level_feat_3

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        # pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        if self.model_type == 'res2net101_26w_4s':
            cached_file = './model_data/res2net101_26w_4s-02a759a1.pth'
            if os.path.exists(cached_file):
                pretrain_dict = torch.load(cached_file)
            else:
                pretrain_dict = model_zoo.load_url('https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_26w_4s-02a759a1.pth')
        elif self.model_type == 'res2net50_26w_4s':
            cached_file = './model_data/res2net50_26w_4s-06e79181.pth'
            if os.path.exists(cached_file):
                pretrain_dict = torch.load(cached_file)
            else:
                pretrain_dict = model_zoo.load_url('https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_4s-06e79181.pth')
        elif self.model_type == 'res2net50_48w_2s':
            pretrain_dict = model_zoo.load_url('https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_48w_2s-afed724a.pth')
        elif self.model_type == 'res2net50_14w_8s':
            pretrain_dict = model_zoo.load_url('https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_14w_8s-6527dddc.pth')
        elif self.model_type == 'res2net50_26w_6s':
            pretrain_dict = model_zoo.load_url('https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_6s-19041792.pth')
        elif self.model_type == 'res2net50_26w_8s':
            pretrain_dict = model_zoo.load_url('https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_8s-2c7c9f12.pth')
        else:
            print('no such model')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

def res2net50(pretrained=False, **kwargs):
    """Constructs a Res2Net-50 model.
    Res2Net-50 refers to the Res2Net-50_26w_4s.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, pretrained=pretrained, model_type='res2net50_26w_4s')
    return model


def res2net50_26w_4s(pretrained=True, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, pretrained=pretrained, model_type='res2net50_26w_4s')

    return model


def res2net101_26w_4s(pretrained=True, **kwargs):
    """Constructs a Res2Net-101_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4, pretrained=pretrained, model_type='res2net101_26w_4s')
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['res2net101_26w_4s']))
    return model


def res2net50_26w_6s(pretrained=True, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=6, pretrained=pretrained, model_type='res2net50_26w_6s')
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['res2net50_26w_6s']))
    return model


def res2net50_26w_8s(pretrained=True, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=8,  pretrained=pretrained, model_type='res2net50_26w_8s')
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['res2net50_26w_8s']))
    return model


def res2net50_48w_2s(pretrained=True, **kwargs):
    """Constructs a Res2Net-50_48w_2s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=48, scale=2,  pretrained=pretrained, model_type='res2net50_48w_2s')
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['res2net50_48w_2s']))
    return model


def res2net50_14w_8s(pretrained=True, **kwargs):
    """Constructs a Res2Net-50_14w_8s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=14, scale=8,  pretrained=pretrained, model_type='res2net50_14w_8s')
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['res2net50_14w_8s']))
    return model


if __name__ == '__main__':
    images = torch.rand(1, 3, 224, 224).cuda(0)
    model = res2net101_26w_4s(pretrained=True)
    model = model.cuda(0)
    print(model(images).size())
