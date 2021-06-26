import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageFilter
import cv2

# 定义地物类型
classes = ['impervious surfaces', 'building',
           'low vegetation', 'tree', 'car', 'clutter']
Imper_Surfaces = [255, 255, 255]
Building = [0, 0, 255]
Low_Vegetation = [0, 255, 255]
Tree = [0, 255, 0]
Car = [255, 255, 0]
Background = [255, 0, 0]
class_color = np.array([Imper_Surfaces, Building, Low_Vegetation, Tree, Car, Background])
# classes = [0, 255]
class_len = len(classes)
class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        label = sample['label']
        normalize = transforms.Normalize(mean=self.mean, std=self.std)
        img = normalize(img)
        return {'image': img,
                'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img_original = sample['image']
        label_original = sample['label']
        # img_original = np.asarray(img_original, dtype=np.float32)
        # h = img_original.shape[0]
        # w = img_original.shape[1]
        # img = np.zeros((h, w, 3)).astype(np.uint8)
        # for i in range(h):
        #     for j in range(w):
        #         img[i, j, :] = img_original[i, j]
        # img = np.concatenate((img, img, img), 0).reshape(512, 512, 3) # 将[512,512]扩充为[3,512,512]
        transform = transforms.ToTensor()
        img = transform(img_original)
        # label = np.array(label)
        # label = LabelToLongTensor(label)
        label = torch.from_numpy(np.array(label_original, dtype=np.int32)).long()
        return {'image': img,
                'label': label}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        label = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': label}

class RandomVerticalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        label = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            label = label.transpose(Image.FLIP_TOP_BOTTOM)

        return {'image': img,
                'label': label}

class RandomRotate(object):
    def __init__(self, degree=90):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        label = sample['label']
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        label = label.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'label': label}

class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        label = sample['label']
        if random.random() < 0.1:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img,
                'label': label}

# def LabelToLongTensor(label):
#     #将各个像素值转换为对应类，生成二维数组 label ，大小为 [ H , W ]
#     GT_label = np.copy(label)
#     for m in classes:
#         GT_label[GT_label == m] = classes.index(m)
#     GT_label = GT_label.astype(np.int16)
#     label = torch.from_numpy(GT_label).long()
#     return label
def LabelToLongTensor(label):
    #将各个像素值转换为对应类，生成二维数组 label ，大小为 [ H , W ]
    GT = np.array(label)
    GT_label = np.zeros((GT.shape[0], GT.shape[1]), dtype='int8')
    for i in range(GT.shape[0]):
        for j in range(GT.shape[1]):
            for k in range(class_len):
                if (GT[i, j] == class_color[k]).all():
                    GT_label[i, j] = k
                else:
                    continue
    label = torch.from_numpy(GT_label).long()
    return label






if __name__ == 'main':
    img_path = 'F:/machine_learning/Unet/dataset/train/images/0.tif'
    label_path = 'F:/machine_learning/Unet/dataset/train/label/0.tif'
    img = Image.open(img_path)
    label = Image.open(label_path)
    # img_rotate
    degree = 30
    rotate_degree = random.uniform(-1 * degree, degree)
    img_rotate = img.rotate(rotate_degree, Image.BILINEAR)
    label_rotate = label.rotate(rotate_degree, Image.NEAREST)

    # img_flip
    img_horizontal_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
    label_horizontal_flip = label.transpose(Image.FLIP_LEFT_RIGHT)
    img_vertical_flip = img.transpose(Image.FLIP_TOP_BOTTOM)
    label_vertical_flip = label.transpose(Image.FLIP_TOP_BOTTOM)

    # img_filter
    random = random.random()
    img_gauss_noise = img.filter(ImageFilter.GaussianBlur(radius=random))
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.subplot(2, 2, 2)
    plt.imshow(img_gauss_noise, cmap='gray')
    plt.subplot(2, 2, 3)
    plt.imshow(img_horizontal_flip, cmap='gray')
    plt.subplot(2, 2, 4)
    plt.imshow(img_vertical_flip, cmap='gray')
    plt.show()