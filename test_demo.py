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
from data import custom_transforms as tr
from options.val_options import *
from options.train_val_options import *
from models.base_model import *
from utils.metrics import *
import torchvision.transforms.functional as Func
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

    rgb = np.stack([r, g, b], axis=2)
    if plot:
        plt.imshow(rgb)
    else:
        return rgb

def get_rot_mat(theta):
    theta = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])


def rot_img(x, theta, dtype):
    rot_mat = get_rot_mat(theta)[None, ...].type(dtype).repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid)
    return x

def predict(input_img_path, model, weights_path=None):
    # 预测单张图片

    if weights_path is not None:
        utils.load_weights(weights_path)
    model.eval()
    with torch.no_grad():
        # input_img_path = os.path.join(input_img_path, image)
        # predict(os.path.join(input_path, image), model)
        label_img_path = input_img_path.replace('images', 'GTs_no_boundary')
        input = Image.open(input_img_path)
        input_img = np.asarray(input)
        # input_img = input_img.reshape(input_img.shape[2], input_img.shape[0], input_img.shape[1])
        # input_img = np.swapaxes(input_img, 0, 2)
        # input_img = np.swapaxes(input_img, 1, 2)
        normalize = transforms.Normalize(mean=mean, std=std)
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        w, h = input.size
        input_narrow_1 = input.resize((int(w * 0.5), int(h * 0.5)), Image.BILINEAR)
        input_narrow_2 = input.resize((int(w * 0.75), int(h * 0.75)), Image.BILINEAR)
        input_expand_1 = input.resize((int(w * 1.25), int(h * 1.25)), Image.BILINEAR)
        input_expand_2 = input.resize((int(w * 1.5), int(h * 1.5)), Image.BILINEAR)
        input_expand_3 = input.resize((int(w * 1.75), int(h * 1.75)), Image.BILINEAR)
        input_Hflip = input.transpose(Image.FLIP_LEFT_RIGHT)
        input_Vflip = input.transpose(Image.FLIP_TOP_BOTTOM)
        input_rotate_180 = input.rotate(180, Image.BILINEAR)
        input_all = [input_narrow_1, input_narrow_2, input, input_expand_2, input_Hflip, input_Vflip, input_rotate_180]
        # input_all = [input_narrow_2, input, input_expand_1, input_Vflip]
        # input_all = [input_narrow_2, input, input_expand_1]
        # input_all = [input, input_Hflip, input_Vflip, input_rotate_180]
        output_all = torch.zeros([len(input_all), class_len, w, h])
        # output_select = torch.zeros([1, class_len, w, h])
        # normalize = transforms.Normalize(mean=mean, std=std)
        # transform = transforms.Compose([transforms.ToTensor(), normalize])
        transform2 = transforms.Compose([
            tr.ToTensor(),
            tr.Normalize(mean=mean, std=mean)])
        softmax = nn.Softmax(dim=1)
        for i in range(len(output_all)):
            img = input_all[i]
            # input to Tensor 类型
            target = Image.open(label_img_path)
            sample = {'image': img, 'label': target}
            sample = transform2(sample)
            img = sample['image'].unsqueeze(0).cuda()
            target = sample['label'].cuda()
            # img = transform(img).unsqueeze(0).cuda()
            output = model(img).cpu()  # [1,C,H,W]
            output = softmax(output)

            print(output.size())
            # output = output.argmax(1).squeeze(0)  # [H,W]
            if (output.size()[2] != w) or (output.size()[3] != h):
                output = F.interpolate(output, size=(w, h), mode="bilinear", align_corners=True)
            if i == 4:
                output = Func.hflip(output)
            if i == 5:
                output = Func.vflip(output)
            if i == 6:
                print(output.size())
                # dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
                output = rot_img(output, np.pi, dtype=torch.FloatTensor)
                # output = Func.rotate(output, 270)
            if i == 6:
                output_select = output
            output_all[i, :, :, :] = output
        # 求得多尺度输入平均结果
        output_avg = torch.sum(output_all, dim=0).argmax(0)
        Metric.add_batch(target.cpu().numpy().astype(int), output_avg.cpu().numpy().astype(int))
        # 获取多尺度输入误差结果
        error_avg_img = (output_avg.numpy() == target.cpu().numpy()).astype('uint8')  # 计算误差曲线
        error_avg_img[error_avg_img > 0] = 255  # 将分类错误的像素给置为0
        error_avg_img = np.stack([error_avg_img] * 3, 2)
        output_avg_img = view_annotated(output_avg.detach(), class_len, False)

        # 重新获取原始输入便于比对
        target = Image.open(label_img_path)
        # target = torch.from_numpy(np.array(target, dtype=np.int32)).long().cuda()  # Target To Tensor 类型
        sample = {'image': input, 'label': target}
        sample = transform2(sample)
        input = sample['image'].unsqueeze(0).cuda()
        target = sample['label'].cuda()
        # 获取原始输入结果
        # output = model(input).cpu()  # [1,C,H,W]
        output = output_select
        output = output.argmax(1).squeeze(0)  # [H,W]
        # Metric.add_batch(target.cpu().numpy().astype(int), output.cpu().numpy().astype(int))
        error_img = (output.numpy() == target.cpu().numpy()).astype('uint8')  # 计算误差曲线
        error_img[error_img > 0] = 255  # 将分类错误的像素给置为0
        error_img = np.stack([error_img] * 3, 2)
        output_img = view_annotated(output.detach(), class_len, False)
        target_img = view_annotated(target.detach(), class_len, False)
        print(input_img.shape)
        plt.figure()
        plt.subplot(2, 4, 1)
        plt.imshow(input_img)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2, 4, 2)
        plt.imshow(target_img)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2, 4, 3)
        plt.imshow(output_img)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2, 4, 4)
        plt.imshow(error_img)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2, 4, 5)
        plt.imshow(input_img)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2, 4, 6)
        plt.imshow(target_img)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2, 4, 7)
        plt.imshow(output_avg_img)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2, 4, 8)
        plt.imshow(error_avg_img)
        plt.xticks([])
        plt.yticks([])

def predict_2(input_img_path, model, weights_path=None):
    # 预测单张图片

    if weights_path is not None:
        utils.load_weights(weights_path)
    model.eval()
    with torch.no_grad():
        # input_img_path = os.path.join(input_img_path, image)
        # predict(os.path.join(input_path, image), model)
        label_img_path = input_img_path.replace('images', 'GTs_no_boundary')
        input = Image.open(input_img_path)
        input_img = np.asarray(input)
        # input_img = input_img.reshape(input_img.shape[2], input_img.shape[0], input_img.shape[1])
        # input_img = np.swapaxes(input_img, 0, 2)
        # input_img = np.swapaxes(input_img, 1, 2)
        normalize = transforms.Normalize(mean=mean, std=std)
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        w, h = input.size
        input_narrow_1 = input.resize((int(w * 0.5), int(h * 0.5)), Image.BILINEAR)
        input_narrow_2 = input.resize((int(w * 0.75), int(h * 0.75)), Image.BILINEAR)
        input_expand_1 = input.resize((int(w * 1.25), int(h * 1.25)), Image.BILINEAR)
        input_expand_2 = input.resize((int(w * 1.5), int(h * 1.5)), Image.BILINEAR)
        input_expand_3 = input.resize((int(w * 1.75), int(h * 1.75)), Image.BILINEAR)
        input_Hflip = input.transpose(Image.FLIP_LEFT_RIGHT)
        input_Vflip = input.transpose(Image.FLIP_TOP_BOTTOM)
        input_rotate_90 = input.rotate(90, Image.BILINEAR)
        input_all = [input, input_narrow_2,  input_expand_2, input_Hflip, input_Vflip, input_rotate_90]
        output_all = torch.zeros([len(input_all), class_len, w, h])
        output_img_all = np.zeros([7, w, h, 3], dtype='uint')
        error_img_all = np.zeros([7, w, h, 3])
        # output_select = torch.zeros([1, class_len, w, h])
        # normalize = transforms.Normalize(mean=mean, std=std)
        # transform = transforms.Compose([transforms.ToTensor(), normalize])
        transform2 = transforms.Compose([
            tr.ToTensor(),
            tr.Normalize(mean=mean, std=mean)])
        softmax = nn.Softmax(dim=1)
        for i in range(len(output_all)):
            img = input_all[i]
            # input to Tensor 类型
            target = Image.open(label_img_path)
            sample = {'image': img, 'label': target}
            sample = transform2(sample)
            img = sample['image'].unsqueeze(0).cuda()
            target = sample['label'].cuda()
            # img = transform(img).unsqueeze(0).cuda()
            output = model(img).cpu()  # [1,C,H,W]
            output = softmax(output)

            # print(output.size())
            # output = output.argmax(1).squeeze(0)  # [H,W]
            if (output.size()[2] != w) or (output.size()[3] != h):
                output = F.interpolate(output, size=(w, h), mode="bilinear", align_corners=True)
            if i == 3:
                output = Func.hflip(output)
            if i == 4:
                output = Func.vflip(output)
            if i == 5:
                # print(output.size())
                # dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
                output = rot_img(output, np.pi / 2 * 3, dtype=torch.FloatTensor)
                # output = Func.rotate(output, 270)
            # if i == 6:
            #     output_select = output
            output_all[i, :, :, :] = output
            output = output.argmax(1).squeeze(0)  # [H,W]
            # Metric.add_batch(target.cpu().numpy().astype(int), output.cpu().numpy().astype(int))
            error_img = (output.numpy() == target.cpu().numpy()).astype('uint8')  # 计算误差曲线
            error_img[error_img > 0] = 255  # 将分类错误的像素给置为0
            error_img = np.stack([error_img] * 3, 2)
            output_img = view_annotated(output.detach(), class_len, False)
            error_img_all[i] = error_img
            output_img_all[i] = output_img
            # target_img = view_annotated(target.detach(), class_len, False)
        # 求得多尺度输入平均结果
        output_avg = torch.sum(output_all, dim=0).argmax(0)
        # Metric.add_batch(target.cpu().numpy().astype(int), output_avg.cpu().numpy().astype(int))
        # 获取多尺度输入误差结果
        error_avg_img = (output_avg.numpy() == target.cpu().numpy()).astype('uint8')  # 计算误差曲线
        error_avg_img[error_avg_img > 0] = 255  # 将分类错误的像素给置为0
        error_avg_img = np.stack([error_avg_img] * 3, 2)
        output_avg_img = view_annotated(output_avg.detach(), class_len, False)
        error_img_all[6] = error_avg_img
        output_img_all[6] = output_avg_img

        # 重新获取原始输入便于比对
        # target = Image.open(label_img_path)
        # # target = torch.from_numpy(np.array(target, dtype=np.int32)).long().cuda()  # Target To Tensor 类型
        # sample = {'image': input, 'label': target}
        # sample = transform2(sample)
        # input = sample['image'].unsqueeze(0).cuda()
        # target = sample['label'].cuda()
        # # 获取原始输入结果
        # # output = model(input).cpu()  # [1,C,H,W]
        # # output = output_select
        # output = output.argmax(1).squeeze(0)  # [H,W]
        # # Metric.add_batch(target.cpu().numpy().astype(int), output.cpu().numpy().astype(int))
        # error_img = (output.numpy() == target.cpu().numpy()).astype('uint8')  # 计算误差曲线
        # error_img[error_img > 0] = 255  # 将分类错误的像素给置为0
        # error_img = np.stack([error_img] * 3, 2)
        # output_img = view_annotated(output.detach(), class_len, False)
        target_img = view_annotated(target.detach(), class_len, False)
        # print(input_img.shape)
        plt.figure()
        plt.subplot(2, 4, 1)
        plt.imshow(target_img)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2, 4, 2)
        plt.imshow(output_img_all[0])
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2, 4, 3)
        plt.imshow(output_img_all[1])
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2, 4, 4)
        plt.imshow(output_img_all[2])
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2, 4, 5)
        plt.imshow(output_img_all[3])
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2, 4, 6)
        plt.imshow(output_img_all[4])
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2, 4, 7)
        plt.imshow(output_img_all[5])
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2, 4, 8)
        plt.imshow(output_img_all[6])
        plt.xticks([])
        plt.yticks([])
        plt.figure()
        plt.subplot(2, 4, 1)
        plt.imshow(target_img)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2, 4, 2)
        plt.imshow(error_img_all[0])
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2, 4, 3)
        plt.imshow(error_img_all[1])
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2, 4, 4)
        plt.imshow(error_img_all[2])
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2, 4, 5)
        plt.imshow(error_img_all[3])
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2, 4, 6)
        plt.imshow(error_img_all[4])
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2, 4, 7)
        plt.imshow(error_img_all[5])
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2, 4, 8)
        plt.imshow(error_img_all[6])
        plt.xticks([])
        plt.yticks([])

        # data = data.cuda()
        # target = target.cuda()
        # output = model(data)
        # output = output.cpu().argmax(1)
        # Metric.add_batch(target.cpu().numpy().astype(int), output.cpu().numpy().astype(int))
        # error_img = (output.numpy() == target.cpu().numpy()).astype('uint8')  # 计算误差曲线
        # error_img[error_img > 0] = 255  # 将分类错误的像素给置为0
        # error_img = np.stack([error_img, error_img, error_img], 1)
        # output_img = view_annotated(output.detach(), class_len, False)
        # target_img = view_annotated(target.detach(), class_len, False)
        # input_img = cv.imread(input_img_path)
        # normalize = transforms.Normalize(mean=mean, std=std)
        # transform = transforms.Compose([transforms.ToTensor(), normalize])
        # input_img = transform(input_img).unsqueeze(0).cuda()
        # output_img = model(input_img).cpu().argmax(1).squeeze(0)
        # output_img = np.array(output_img.detach())
        # for i in range(class_len):
        #     output_img[output_img == i] = classes[i]
        # output_img = output_img.astype(np.uint16)
        # output_img_path = input_img_path.replace('input', 'results').replace('tif', 'png')
        # cv.imwrite(output_img_path, output_img)

def predict_all(input_path, model, Metric, weights_path=None):
    # 预测一个文件夹内的所有图片
    # input_path：传入图像文件夹
    # model：传入模型
    # weights_path：权重保存路径
    num = 0
    string = ''
    Metric.reset()
    viz = visdom.Visdom(env="Prediction")
    if weights_path is not None:
        model.load_weights(weights_path)
    model.eval()
    with torch.no_grad():
        for image in os.listdir(input_path):
            input_img_path = os.path.join(input_path, image)
            # predict(os.path.join(input_path, image), model)
            label_img_path = input_img_path.replace('images', 'GTs_no_boundary')
            input = Image.open(input_img_path)
            input_img = np.asarray(input)
            # input_img = input_img.reshape(input_img.shape[2], input_img.shape[0], input_img.shape[1])
            input_img = np.swapaxes(input_img, 0, 2)
            input_img = np.swapaxes(input_img, 1, 2)
            normalize = transforms.Normalize(mean=mean, std=std)
            transform = transforms.Compose([transforms.ToTensor(), normalize])
            w, h = input.size
            input_narrow_1 = input.resize((int(w * 0.5), int(h * 0.5)), Image.BILINEAR)
            input_narrow_2 = input.resize((int(w * 0.75), int(h * 0.75)), Image.BILINEAR)
            input_expand_1 = input.resize((int(w * 1.25), int(h * 1.25)), Image.BILINEAR)
            input_expand_2 = input.resize((int(w * 1.5), int(h * 1.5)), Image.BILINEAR)
            input_expand_3 = input.resize((int(w * 1.75), int(h * 1.75)), Image.BILINEAR)
            input_Hflip = input.transpose(Image.FLIP_LEFT_RIGHT)
            input_Vflip = input.transpose(Image.FLIP_TOP_BOTTOM)
            input_rotate_90 = input.rotate(90, Image.BILINEAR)
            input_all = [input_narrow_1, input_narrow_2, input, input_expand_1, input_expand_2, input_Hflip, input_Vflip]
            # input_all = [input_narrow_2, input, input_expand_1, input_Vflip]
            # input_all = [input_narrow_2, input, input_expand_1]
            # input_all = [input, input_rotate_90]
            output_all = torch.zeros([len(input_all), class_len, w, h])
            # normalize = transforms.Normalize(mean=mean, std=std)
            # transform = transforms.Compose([transforms.ToTensor(), normalize])
            transform2 = transforms.Compose([
                tr.ToTensor(),
                tr.Normalize(mean=mean, std=mean)])
            softmax = nn.Softmax(dim=1)
            for i in range(len(output_all)):
                img = input_all[i]
                # input to Tensor 类型
                target = Image.open(label_img_path)
                sample = {'image': img, 'label': target}
                sample = transform2(sample)
                img = sample['image'].unsqueeze(0).cuda()
                target = sample['label'].cuda()
                # img = transform(img).unsqueeze(0).cuda()
                output = model(img).cpu()  # [1,C,H,W]
                output = softmax(output)
                # output = output.argmax(1).squeeze(0)  # [H,W]
                if (output.size()[0] != w) or (output.size()[1] != h):
                    output = F.interpolate(output, size=(w, h), mode="bilinear", align_corners=True).squeeze(0)
                if i == 5:
                    output = Func.hflip(output)
                if i == 6:
                    output = Func.vflip(output)
                output_all[i, :, :, :] = output
            # 求得多尺度输入平均结果
            output_avg = torch.sum(output_all, dim=0).argmax(0)
            Metric.add_batch(target.cpu().numpy().astype(int), output_avg.cpu().numpy().astype(int))
            # 获取多尺度输入误差结果
            error_avg_img = (output_avg.numpy() == target.cpu().numpy()).astype('uint8')  # 计算误差曲线
            error_avg_img[error_avg_img > 0] = 255  # 将分类错误的像素给置为0
            error_avg_img = np.stack([error_avg_img] * 3, 0)
            output_avg_img = view_annotated(output_avg.detach(), class_len, False)

            # 重新获取原始输入便于比对
            target = Image.open(label_img_path)
            # target = torch.from_numpy(np.array(target, dtype=np.int32)).long().cuda()  # Target To Tensor 类型
            sample = {'image': input, 'label': target}
            sample = transform2(sample)
            input = sample['image'].unsqueeze(0).cuda()
            target = sample['label'].cuda()
            # 获取原始输入结果
            output = model(input).cpu()  # [1,C,H,W]
            output = output.argmax(1).squeeze(0)  # [H,W]
            # Metric.add_batch(target.cpu().numpy().astype(int), output.cpu().numpy().astype(int))
            error_img = (output.numpy() == target.cpu().numpy()).astype('uint8')  # 计算误差曲线
            error_img[error_img > 0] = 255  # 将分类错误的像素给置为0
            error_img = np.stack([error_img] * 3, 0)
            output_img = view_annotated(output.detach(), class_len, False)
            target_img = view_annotated(target.detach(), class_len, False)

            # visualization
            images_col_1 = np.stack([input_img, target_img, output_img, error_img], 0)
            images_col_2 = np.stack([input_img, target_img, output_avg_img, error_avg_img], 0)
            images = np.concatenate([images_col_1, images_col_2])
            if num % 5 == 0:
                MIOU = Metric.Mean_Intersection_over_Union()
                _, Avg_OA = Metric.Pixel_Accuracy_Class()
                OA = Metric.Pixel_Accuracy()
                IOU, FWIOU = Metric.Frequency_Weighted_Intersection_over_Union()
                F1_score, Avg_F1_score = Metric.F1_score()
                string = string + '{'+f"'iters': {num}, 'OA':{OA}, 'Avg_OA': {Avg_OA}, 'F1_score': {F1_score}, 'Avg_F1_score':{Avg_F1_score} MIOU': {MIOU}"+'}<br>'
                viz.text(string, win='_Metrics')
                viz.images(images, win='images', nrow=4, opts=dict(title='Visualization', custom='GT/Pred/Error', width=1200, height=600))
            num = num + 1
        MIOU = Metric.Mean_Intersection_over_Union()
        _, Avg_OA = Metric.Pixel_Accuracy_Class()
        OA = Metric.Pixel_Accuracy()
        IOU, FWIOU = Metric.Frequency_Weighted_Intersection_over_Union()
        F1_score, Avg_F1_score = Metric.F1_score()
        print("The final result:\n")
        print(f"The OA:{OA}  Avg_OA:{Avg_OA}  MIOU:{MIOU}  F1_score:{F1_score} Avg_F1_score:{Avg_F1_score}")
        print("The proportion of feature types:", np.sum(Metric.confusion_matrix, 1) / Metric.confusion_matrix.sum())
        print(Metric.confusion_matrix)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda')
    # model = DeepLab(num_classes=6, backbone='resnet-101', output_stride=16).to(device)
    model = DUNet(num_classes=6, backbone='resnet-101', output_stride=16).cuda()

    # model = ResNetUNet(8).to(device)
    # kwargs = {'map_location': lambda
    # storage, loc: storage.cuda(gpu_id)}
    # pthfile = './checkpoints/DeeplabV3+/resnet_101/Potsdam/weights-175--loss_0.7003--OA_0.8882--MIOU_0.8193--F1_score_0.8993.pth'
    pthfile = './checkpoints/DUNet/Vahingen/weights-71--loss_0.4785--OA_0.8738--MIOU_0.7565--F1_score_0.8590.pth'
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
    print("[Network %s] The total number of parameters : %.3f M" % ("DUNet_res101", num_params / 1e6))
    Metric = Evaluator(class_len)
    path = 'F:\数据集\ISPRS2D_Potsdam_Vahingen\Vahingen\Small_size\Size_256\\test\\images'
    # path = 'F:\数据集\ISPRS2D_Potsdam_Vahingen\Vahingen\Small_size\\test\\images'
    # predict_all(input_path=path, model=model, Metric=Metric)
    input_path_1 = 'F:\数据集\ISPRS2D_Potsdam_Vahingen\Vahingen\Small_size\Size_256\\test\images\\26.png'
    input_path_2 = 'F:\数据集\ISPRS2D_Potsdam_Vahingen\Vahingen\Small_size\Size_256\\test\images\\27.png'
    input_path_3 = 'F:\数据集\ISPRS2D_Potsdam_Vahingen\Vahingen\Small_size\Size_256\\test\images\\50.png'
    # input_path_1 = 'F:\数据集\ISPRS2D_Potsdam_Vahingen\Vahingen\Small_size\Size_256\\test\images\\3.png'
    # input_path_2 = 'F:\数据集\ISPRS2D_Potsdam_Vahingen\Vahingen\Small_size\Size_256\\test\images\\23.png'
    # input_path_3 = 'F:\数据集\ISPRS2D_Potsdam_Vahingen\Vahingen\Small_size\Size_256\\test\images\\212.png'
    predict_2(input_path_1, model=model)
    predict_2(input_path_2, model=model)
    predict_2(input_path_3, model=model)
    plt.show()

