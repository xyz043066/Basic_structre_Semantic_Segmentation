import torch.nn.functional as F
from models.Deeplab.spatial_path import *
from models.Deeplab.aspp import build_aspp
from models.Deeplab.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm, output_stride):
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
        self.out_channels = 1024
        # BatchNorm = nn.BatchNorm2d
        self.MSFF_1 = MultiScaleFeatureFusion(sample="down", output_channels=self.out_channels)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm, inplanes=self.out_channels)
        self.MSFF_2 = MultiScaleFeatureFusion(sample="up",  output_channels=self.out_channels)
        self.AttentionFusionModule = AttentionFusionModule(in_channels=self.out_channels, out_channels=256)
        self.final_conv = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1)


        # self.reduction = 16
        # self.MSFF_1 = MultiScaleFeatureFusion(sample="down", output_channels=self.out_channels)
        # self.MSFF_2 = MultiScaleFeatureFusion(sample="up",  output_channels=self.out_channels)
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Sequential(
        #     nn.Linear(self.out_channels, self.out_channels // self.reduction, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.out_channels // self.reduction, self.out_channels, bias=False),
        #     nn.Sigmoid()
        # )
        # self.convblock = ConvBlock(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0)
        # self.AttentionFusionModule1 = AttentionFusionModule(in_channels=low_level_inplanes_3, out_channels=256)
        # self.AttentionFusionModule2 = AttentionFusionModule(in_channels=low_level_inplanes_2, out_channels=256)
        # self.AttentionFusionModule3 = AttentionFusionModule(in_channels=low_level_inplanes_1, out_channels=num_classes)
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
        # self.final_conv = nn.Conv2d(in_channels=self.out_channels, out_channels=num_classes, kernel_size=1)
        self._init_weight()

    def forward(self, x, low_level_feat_1, low_level_feat_2):
        x_high = self.MSFF_1(x, low_level_feat_1, low_level_feat_2)
        x_high = self.aspp(x_high)
        x_low = self.MSFF_2(x, low_level_feat_1, low_level_feat_2)  # [b c h/8 w/8]
        # output = torch.cat((x_low, x_high), dim=1)
        output = self.AttentionFusionModule(x_high, x_low)
        output = self.final_conv(output)
        # b, _, _, _ = x.size()
        # x_high = self.MSFF_1(x, low_level_feat_1, low_level_feat_2)
        # x_high = self.avg_pool(x_high).view(b, self.out_channels)
        # x_high = self.fc(x_high).view(b, self.out_channels, 1, 1) # [b c 1 1]
        # x_low = self.MSFF_2(x, low_level_feat_1, low_level_feat_2) # [b c h/8 w/8]
        # output = x_low * x_high.expand_as(x_low)
        #
        # output = self.final_conv(output)
        # x = torch.cat((x, low_level_feat), dim=1)
        # x = self.last_conv(x)

        return output

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


def build_decoder(num_classes, backbone, BatchNorm, output_stride):
    return Decoder(num_classes, backbone, BatchNorm, output_stride)
