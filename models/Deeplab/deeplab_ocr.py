import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Deeplab.aspp import build_aspp
from models.Deeplab.backbone import build_backbone
from models.Deeplab.decoder import build_decoder
from models.Deeplab.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.HRNet.seg_hrnet_ocr import *

# from models.spatial_path import *

class DeepLab_OCR(nn.Module):
    def __init__(self, backbone='resnet-18', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab_OCR, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        last_inp_channels = 1024
        ocr_mid_channels = 512
        ocr_key_channels = 256

        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(ocr_key_channels, ocr_mid_channels,
                      kernel_size=3, stride=1, padding=1),
            BatchNorm2d(ocr_mid_channels),
            nn.ReLU(inplace=relu_inplace),
        )
        self.ocr_gather_head = SpatialGather_Module(num_classes)

        self.ocr_distri_head = SpatialOCR_Module(in_channels=ocr_mid_channels,
                                                 key_channels=ocr_key_channels,
                                                 out_channels=ocr_mid_channels,
                                                 scale=1,
                                                 dropout=0.05,
                                                 )
        self.cls_head = nn.Conv2d(
            ocr_mid_channels, ocr_key_channels, kernel_size=1, stride=1, padding=0, bias=True)

        self.aux_head = nn.Sequential(
            nn.Conv2d(last_inp_channels, last_inp_channels,
                      kernel_size=1, stride=1, padding=0),
            BatchNorm2d(last_inp_channels),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(last_inp_channels, num_classes,
                      kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feat, feats = self.backbone(input)
        x = self.aspp(x)
        # ocr
        out_aux = self.aux_head(feats)
        # compute contrast feature
        x = self.conv3x3_ocr(x)

        context = self.ocr_gather_head(x, out_aux)
        x = self.ocr_distri_head(x, context)

        x = self.cls_head(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


if __name__ == "__main__":
    model = DeepLab_OCR(backbone='resnet-18', output_stride=16)
    print(model)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())



# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
# from models.aspp import build_aspp
# from models.decoder import build_decoder
# from models.backbone import build_backbone

# class DeepLab(nn.Module):
#     def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
#                  sync_bn=True, freeze_bn=False):
#         super(DeepLab, self).__init__()
#         if backbone == 'drn':
#             output_stride = 8

#         if sync_bn == True:
#             BatchNorm = SynchronizedBatchNorm2d
#         else:
#             BatchNorm = nn.BatchNorm2d

#         self.backbone = build_backbone(backbone, output_stride, BatchNorm)
#         self.aspp = build_aspp(backbone, output_stride, BatchNorm)
#         self.decoder = build_decoder(num_classes, backbone, BatchNorm)

#         if freeze_bn:
#             self.freeze_bn()

#     def forward(self, input):
#         x, low_level_feat = self.backbone(input)
#         x, x1 = self.aspp(x)
#         x = self.decoder(x, low_level_feat, x1)
#         x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

#         return x

#     def freeze_bn(self):
#         for m in self.modules():
#             if isinstance(m, SynchronizedBatchNorm2d):
#                 m.eval()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.eval()

#     def get_1x_lr_params(self):
#         modules = [self.backbone]
#         for i in range(len(modules)):
#             for m in modules[i].named_modules():
#                 if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
#                         or isinstance(m[1], nn.BatchNorm2d):
#                     for p in m[1].parameters():
#                         if p.requires_grad:
#                             yield p

#     def get_10x_lr_params(self):
#         modules = [self.aspp, self.decoder]
#         for i in range(len(modules)):
#             for m in modules[i].named_modules():
#                 if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
#                         or isinstance(m[1], nn.BatchNorm2d):
#                     for p in m[1].parameters():
#                         if p.requires_grad:
#                             yield p


# if __name__ == "__main__":
#     model = DeepLab(backbone='mobilenet', output_stride=16)
#     model.eval()
#     input = torch.rand(1, 3, 513, 513)
#     output = model(input)
#     print(output.size())

