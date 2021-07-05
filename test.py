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
from utils.metrics import *
import visdom

def view_annotated(tensor, nc=class_len, plot=True):
    image = tensor.cpu().numpy()
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(0, nc):
        idx = image == l
        r[idx] = class_color[l, 0]
        g[idx] = class_color[l, 1]
        b[idx] = class_color[l, 2]

    rgb = np.stack([r, g, b], axis=1)
    if plot:
        plt.imshow(rgb)
    else:
        return rgb

def predict(val_loader, model, Metric, weights_path=None, mode='train'):
    # 预测单张图片
    i = 0
    string = ''
    Metric.reset()
    viz = visdom.Visdom(env='test1_show_predictions')
    dataset = val_loader.dataset
    img_list = dataset.images_list
    if weights_path is not None:
        utils.load_weights(weights_path)
    model.eval()
    with torch.no_grad():
        for data, target in val_loader:
            # i = i + 1
            data = data.cuda()
            target = target.cuda()
            output = model(data)
            output = output.cpu().argmax(1)
            Metric.add_batch(target.cpu().numpy().astype(int), output.cpu().numpy().astype(int))
            error_img = (output.numpy() == target.cpu().numpy()).astype('uint8')  # 计算误差曲线
            error_img[error_img > 0] = 255  # 将分类错误的像素给置为0
            error_img = np.stack([error_img, error_img, error_img], 1)
            output_img = view_annotated(output.detach(), class_len, False)
            target_img = view_annotated(target.detach(), class_len, False)
            images_col_1 = np.stack([target_img[0], output_img[0], error_img[0]], 0)
            images_col_2 = np.stack([target_img[1], output_img[1], error_img[1]], 0)
            images = np.concatenate([images_col_1, images_col_2])
            if i % 5 == 0:
                MIOU = Metric.Mean_Intersection_over_Union()
                OA = Metric.Pixel_Accuracy()
                IOU, FWIOU = Metric.Frequency_Weighted_Intersection_over_Union()
                F1_score, Avg_F1_score = Metric.F1_score()
                string = string + '{'+f"'iters': {i}, 'OA': {OA}, 'F1_score': {Avg_F1_score}, 'MIOU': {MIOU}, 'FWIOU': {FWIOU}"+'}<br>'
                viz.text(string, win='_Metrics')
                viz.images(images, win='images', nrow=3, opts=dict(title='Visualization', custom='GT/Pred/Error', width=900, height=600))
                # viz.images(output_img, win='pred_img', opts=dict(title='pred_images', ))
                # viz.images(target_img, win='target_img', opts=dict(title='target_images'))
                # time.sleep(1)
            i = i + 1
    MIOU = Metric.Mean_Intersection_over_Union()
    OA = Metric.Pixel_Accuracy()
    IOU, FWIOU = Metric.Frequency_Weighted_Intersection_over_Union()
    F1_score, Avg_F1_score = Metric.F1_score()
    print(f"The OA:{OA}  MIOU:{MIOU}  F1_score:{Avg_F1_score}")
    print("The proportion of feature types:", np.sum(Metric.confusion_matrix, 0) / Metric.confusion_matrix.sum())

        # for input_img_path in img_list:
        #     input_img = Image.open(input_img_path)
        #     normalize = transforms.Normalize(mean=mean, std=std)
        #     transform = transforms.Compose([transforms.ToTensor(), normalize])
        #     # input_img = transform(input_img).unsqueeze(0).cuda()
        #     input_img = transform(input_img).unsqueeze(0).cuda()
        #     output_img = model(input_img).cpu().argmax(1).squeeze(0)
        #     output_img = view_annotated(output_img.detach(), class_len, False)
        #     # viz.image(output_img, win='pred_img', opts=dict(title='images'))
        #     # for i in range(class_len):
        #     #     output_img[output_img == i] = classes[i]
        #     output_img = output_img.astype(np.uint8)
        #     output_img = Image.fromarray(output_img)
        #     output_img_path = input_img_path.replace('images', 'preds_resnet101_3')
        #     # cv.imwrite(output_img_path, output_img)
        #     output_img.save(output_img_path)
        #     print(f"save the {output_img_path}!")


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
    print("Validation: %d" % len(val_loader.dataset))
    print("Classes: %d" % class_len)
    inputs, targets = next(iter(val_loader))
    print("Inputs size: ", inputs.size())
    print("Targets size: ", targets.size())
    print("Inputs type:", inputs.type())
    print("Targets type:", targets.type())
    Metric_val = Evaluator(opt_val.class_num)
    device = torch.device('cuda')
    opt_val.backbone = 'resnet-101'
    model = DeepLab(num_classes=6, backbone=opt_val.backbone, output_stride=16).to(device)
    # model = ResNetUNet(8).to(device)
    # kwargs = {'map_location': lambda storage, loc: storage.cuda(gpu_id)}
    pthfile = './checkpoints/DeeplabV3+/resnet_101/Vahingen/weights-294--loss_0.1121--OA_0.9672--MIOU_0.9185--F1_score_0.9570.pth'
    # pthfile = './checkpoints/DeeplabV3+/resnet_50/Vahingen/weights-268--loss_0.1193--OA_0.9641--MIOU_0.9088--F1_score_0.9516.pth'
    # pthfile = './checkpoints/DeeplabV3+/resnet_34/weights-199--loss_0.1104--OA_0.9440--MIOU_0.8947--F1_score_0.9438.pth'
    # pthfile = './checkpoints/DeeplabV3+/resnet_18/weights-195--loss_0.1682--OA_0.9190--MIOU_0.8540--F1_score_0.9203.pth'
    model_dict = torch.load(pthfile,  map_location='cuda:0')['Net']
    # new_pre = {}
    # for k, v in model_dict.items():
    #     name = k[7:]
    #     new_pre[name] = v
    # model.load_state_dict(new_pre)
    model.load_state_dict(model_dict)
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print("[Network %s] The total number of parameters : %.3f M" % (opt_val.model_name+'_'+opt_val.backbone, num_params / 1e6))
    input_path = 'F:\数据集\ISPRS2D_Potsdam_Vahingen\Potsdam\Small_size\Size_256\images\\122.png'
    predict(val_loader, model=model, Metric=Metric_val)


