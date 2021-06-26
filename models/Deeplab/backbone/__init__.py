from models.Deeplab.backbone import resnet, xception, mobilenet
from models.Deeplab.backbone import drn


def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet-18':
        return resnet.ResNet18(output_stride, BatchNorm)
    elif backbone == 'resnet-34':
        return resnet.ResNet34(output_stride, BatchNorm)
    elif backbone == 'resnet-50':
        return resnet.ResNet50(output_stride, BatchNorm)
    elif backbone == 'resnet-101':
        return resnet.ResNet101(output_stride, BatchNorm)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    else:
        raise NotImplementedError