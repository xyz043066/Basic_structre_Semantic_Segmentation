import os
import cv2 as cv
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from utils import utils
from models.Deeplab.sync_batchnorm.replicate import patch_replication_callback
from models.Deeplab.deeplab import *
from data.base_dataloader import *
from options.val_options import *
from options.train_val_options import *
from models.base_model import *

def view_annotated(tensor, nc=class_len, plot=True):
    image = tensor.numpy()
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(0, nc):
        idx = image == l
        r[idx] = class_color[l, 0]
        g[idx] = class_color[l, 1]
        b[idx] = class_color[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    if plot:
        plt.imshow(rgb)
    else:
        return rgb

def predict(val_loader, model, weights_path=None):
    # 预测单张图片
    dataset = val_loader.dataset
    img_list = dataset.images_list
    if weights_path is not None:
        utils.load_weights(weights_path)
    model.eval()
    with torch.no_grad():
        for input_img_path in img_list:
            input_img = Image.open(input_img_path)
            normalize = transforms.Normalize(mean=mean, std=std)
            transform = transforms.Compose([transforms.ToTensor(), normalize])
            input_img = transform(input_img).unsqueeze(0).cuda()
            output_img = model(input_img).cpu().argmax(1).squeeze(0)
            output_img = view_annotated(output_img.detach(), class_len, False)
            # for i in range(class_len):
            #     output_img[output_img == i] = classes[i]
            output_img = output_img.astype(np.uint8)
            output_img = Image.fromarray(output_img)
            output_img_path = input_img_path.replace('images', 'preds_resnet101_2')
            # cv.imwrite(output_img_path, output_img)
            output_img.save(output_img_path)
            print(f"save the {output_img_path}!")


def predict_all(input_path, model, weights_path=None):
    # 预测一个文件夹内的所有图片
    # input_path：传入图像文件夹
    # model：传入模型
    # weights_path：权重保存路径
    if weights_path is not None:
        model.load_weights(weights_path)
    for image in os.listdir(input_path):
        print(image)
        predict(os.path.join(input_path, image), model)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    opt_val = TrainOptions(is_train=False).parse()
    val_loader = My_dataloader(opt_val)
    device = torch.device('cuda')
    model = DeepLab(num_classes=6, backbone='resnet-101', output_stride=16).to(device)
    # model = ResNetUNet(8).to(device)
    # kwargs = {'map_location': lambda storage, loc: storage.cuda(gpu_id)}
    pthfile = './checkpoints/DeeplabV3+/resnet_101/weights-199--loss_0.0659--OA_0.9647--MIOU_0.9320--F1_score_0.9644.pth'
    model_dict = torch.load(pthfile,  map_location='cuda:0')['Net']
    # new_pre = {}
    # for k, v in model_dict.items():
    #     name = k[7:]
    #     new_pre[name] = v
    # model.load_state_dict(new_pre)
    model.load_state_dict(model_dict)
    input_path = 'F:\数据集\ISPRS2D_Potsdam_Vahingen\Potsdam\Small_size\Size_256\images\\122.png'
    predict(val_loader, model=model)
