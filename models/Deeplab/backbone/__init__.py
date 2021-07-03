from models.Deeplab.backbone import resnet, res2net, xception, mobilenet
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
    elif backbone == 'res2net50_26w_4s':
        return res2net.res2net50_26w_4s()
    elif backbone == 'res2net50_26w_6s':
        return res2net.res2net50_26w_6s()
    elif backbone == 'res2net50_26w_8s':
        return res2net.res2net50_26w_8s()
    elif backbone == 'res2net50_48w_2s':
        return res2net.res2net50_48w_2s()
    elif backbone == 'res2net50_14w_8s':
        return res2net.res2net50_14w_8s()
    elif backbone == 'res2net101_26w_4s':
        return res2net.res2net101_26w_4s()
    elif backbone == 'xception':
        return xception.AlignedXception()
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    else:
        raise NotImplementedError
