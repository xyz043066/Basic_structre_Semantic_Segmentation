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
        self.out_channels = 1024
        self.reduction = 16
        self.N = 256
        self.MSFF_1 = MultiScaleFeatureFusion(sample="down", output_channels=self.out_channels)
        self.MSFF_2 = MultiScaleFeatureFusion(sample="up",  output_channels=self.out_channels)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.SoftMax(dim=2)
        self.convblock_1 = ConvBlock(in_channels=self.out_channels, out_channels=self.N, kernel_size=1, stride=1, padding=0)
        self.convblock_2 = ConvBlock(in_channels=self.out_channels, out_channels=self.N, kernel_size=1, stride=1, padding=0)
        # self.convblock_2 = ConvBlock(in_channels=self.out_channels*2, out_channels=self.N, kernel_size=1, stride=1, padding=0)
        # self.fc = nn.Sequential(
        #     nn.Linear(self.out_channels, self.out_channels // self.reduction, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.out_channels // self.reduction, self.out_channels, bias=False),
        #     nn.Sigmoid()
        # )

        self.final_conv = nn.Conv2d(in_channels=self.out_channels*2, out_channels=num_classes, kernel_size=1)
        self._init_weight()

    def forward(self, input, low_level_feat_1, low_level_feat_2):
        b, _, w, h = low_level_feat_2.size() # [b c h/8 w/8]
        # x_high = self.MSFF_1(x, low_level_feat_1, low_level_feat_2)
        # x_high = self.avg_pool(x_high).view(b, self.out_channels)
        # x_high = self.fc(x_high).view(b, self.out_channels, 1, 1) # [b c 1 1]
        # x_low = self.MSFF_2(x, low_level_feat_1, low_level_feat_2) # [b c h/8 w/8]
        # output = x_low * x_high.expand_as(x_low)
        x_high = self.MSFF_1(input, low_level_feat_1, low_level_feat_2) # [b 1024 h/32 w/32]
        x_low = self.MSFF_2(input, low_level_feat_1, low_level_feat_2)  # [b 1024 h/8 w/8]
        codebook = self.convblock_1(x_high).view(b, self.N, -1)
        codebook = self.softmax(codebook).permute(0, 2, 1).contiguous()  # [b 1024 N]
        codebook = torch.matmul(x_high.view(b, self.out_channels, -1), codebook) # [b 1024 N]

        # avg_pool
        avg = self.avg_pool(x_high) # [b 1024 1 1]
        avg = torch.add(avg, x_low) # [b 1024 h/8 w/8]
        avg = self.convblock_2(avg) # [b N h/8 w/8]
        avg = torch.matmul(codebook, avg.view(b, self.N, -1)).view(b, self.out_channels, w, h) # [b 1024 h/8 w/8]
        output = torch.cat((x_low, avg), dim=1)


        output = self.final_conv(output)
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


def build_decoder(num_classes, backbone, BatchNorm):
    return Decoder(num_classes, backbone, BatchNorm)
