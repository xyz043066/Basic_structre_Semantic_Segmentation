import torch.nn.functional as F
from models.Deeplab.spatial_path import *
from models.Deeplab.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()

        if backbone == 'resnet-101' or backbone == 'resnet-50' or backbone == 'drn' or 'res2net101_26w_4s' \
                or 'res2net50_26w_4s' or 'res2net50_26w_6s' or 'res2net50_26w_8s' or 'res2net50_18w_8s' or 'res2net50_48w_2s':
            low_level_inplanes_1 = 256
            low_level_inplanes_2 = 512
            low_level_inplanes_3 = 1024
        elif backbone == 'xception':
            low_level_inplanes_1 = 128
        elif backbone == 'resnet-18' or backbone == 'resnet-34':
            low_level_inplanes_1 = 64
            low_level_inplanes_2 = 128
        elif backbone == 'mobilenet':
            low_level_inplanes_1 = 24
        else:
            raise NotImplementedError

        # self.convblock = ConvBlock(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0)
        # self.AttentionFusionModule1 = AttentionFusionModule(in_channels=low_level_inplanes_3, out_channels=256)
        # self.AttentionFusionModule2 = AttentionFusionModule(in_channels=low_level_inplanes_2, out_channels=256)
        # self.AttentionFusionModule3 = AttentionFusionModule(in_channels=low_level_inplanes_1, out_channels=num_classes)
        self.SFAM_1 = SFAM(in_channels_2=low_level_inplanes_3, in_channels_1=256)
        self.SFAM_2  = SFAM(in_channels_2=low_level_inplanes_2, in_channels_1=256)
        self.SFAM_3  = SFAM(in_channels_2=low_level_inplanes_1, in_channels_1=256)
         # self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        # self.bn1 = BatchNorm(48)
        # self.relu = nn.ReLU()
        # self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
        #                                BatchNorm(256),
        #                                nn.ReLU(),
        #                                nn.Dropout(0.5),
        #                                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
        #                                BatchNorm(256),
        #                                nn.ReLU(),
        #                                nn.Dropout(0.1),
        #                                nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        #
        # self.feature_fusion_module = FeatureFusionModule(num_classes, 304)
        self.final_conv = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1)
        self._init_weight()

    def forward(self, x, low_level_feat_1, low_level_feat_2, low_level_feat_3):
        # x = F.interpolate(x, size=low_level_feat_2.size()[2:], mode='bilinear', align_corners=True)
        # x = self.convblock(x)
        x = self.SFAM_1(low_level_feat_3, x)
        x = self.SFAM_2(low_level_feat_2, x)
        x = self.SFAM_3(low_level_feat_1, x)
        # low_level_feat1 = self.conv1(low_level_feat_1)
        # low_level_feat = self.bn1(low_level_feat_1)
        # low_level_feat = self.relu(low_level_feat_1)

        # x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        # x = F.interpolate(x, size=(low_level_feat.size()[2], low_level_feat.size()[3]), mode='bilinear',
        #                   align_corners=True)
        #
        # # Upsample_module = nn.Upsample(size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        # # x = Upsample_module(x)
        #
        # # x = torch.cat((x, low_level_feat), dim=1)
        # # x = self.last_conv(x)
        # # print("x size = ", x.size())
        # # print("low_level_feat size =", low_level_feat.size(), "x size = ", x.size())
        # # exit()
        #
        # x = self.feature_fusion_module(low_level_feat, x)
        x = self.final_conv(x)
        # x = torch.cat((x, low_level_feat), dim=1)
        # x = self.last_conv(x)

        return x

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


def build_decoder(num_classes, backbone, BatchNorm):
    return Decoder(num_classes, backbone, BatchNorm)
