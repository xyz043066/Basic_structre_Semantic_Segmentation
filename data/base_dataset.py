import os
import cv2
import random
import torch.utils.data as data
import torchvision.transforms as transforms
from data import custom_transforms as tr
import numpy as np
from PIL import Image
import hashlib

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

mean = [0.41189489566336, 0.4251328133025, 0.4326707089857]
std = [0.27413549931506, 0.28506257482912, 0.28284674400252]
class_len = len(class_color)
# classes = [0, 255]
# class_len = len(classes)
# mean = [0.41189489566336, 0.4251328133025, 0.4326707089857]
# std = [0.27413549931506, 0.28506257482912, 0.28284674400252]
class Base_dataset(data.Dataset):
    def __init__(self, opt):
        self.train = opt.isTrain
        self.images_dir = opt.dataroot
        self.is_transform = opt.transform
        self.ids = os.listdir(self.images_dir)
        # 读取数据集中各图像路径
        images_list = []
        for image_id in self.ids:
            if os.path.isfile(os.path.join(self.images_dir, image_id)):
                file_name, _ = os.path.splitext(image_id)
                images_list.append(self.images_dir + file_name + ".png")
        # 通过img_path的md5值对img_path进行排序
        images_list = self.get_img_hash_dict(images_list)
        self.images_len = len(images_list)
        if self.train:
            self.images_list = images_list[:int(0.8*self.images_len)]
        else:
            self.images_list = images_list[int(0.8*self.images_len):]
        # normalize = transforms.Normalize(mean=mean, std=std)
        if self.is_transform:
            self.transform = transforms.Compose([
                tr.RandomHorizontalFlip(),
                tr.RandomVerticalFlip(),
                # tr.RandomGaussianBlur(),
                tr.ToTensor(),
                tr.Normalize(mean=mean, std=mean)])
        else:
            self.transform = transforms.Compose([
                tr.ToTensor(),
                tr.Normalize(mean=mean, std=mean)])
    def __getitem__(self, index):
        img_item_path = self.images_list[index]
        label_item_path = img_item_path.replace('images', 'labels')
        img = Image.open(img_item_path)
        label = Image.open(label_item_path)
        sample = {'image': img, 'label': label}
        if self.is_transform is not None:
            sample = self.transform(sample)

        return sample['image'], sample['label']

    def __len__(self):
        return len(self.images_list)

    def _rotate(self, image, label):
        # Rotate the image with an angle between -10 and 10
        h, w, _ = image.shape
        angle = random.randint(-10, 10)
        center = (w / 2, h / 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, rot_matrix, (w, h), flags=cv2.INTER_CUBIC)  # , borderMode=cv2.BORDER_REFLECT)
        label = cv2.warpAffine(label, rot_matrix, (w, h), flags=cv2.INTER_NEAREST)  # ,  borderMode=cv2.BORDER_REFLECT)
        return image, label

    def _crop(self, image, label):
        # Padding to return the correct crop size
        if (isinstance(self.crop_size, list) or isinstance(self.crop_size, tuple)) and len(self.crop_size) == 2:
            crop_h, crop_w = self.crop_size
        elif isinstance(self.crop_size, int):
            crop_h, crop_w = self.crop_size, self.crop_size
        else:
            raise ValueError

        h, w, _ = image.shape
        pad_h = max(crop_h - h, 0)
        pad_w = max(crop_w - w, 0)
        pad_kwargs = {
            "top": 0,
            "bottom": pad_h,
            "left": 0,
            "right": pad_w,
            "borderType": cv2.BORDER_CONSTANT, }
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, value=self.image_padding, **pad_kwargs)
            label = cv2.copyMakeBorder(label, value=self.ignore_index, **pad_kwargs)

        # Cropping
        h, w, _ = image.shape
        start_h = random.randint(0, h - crop_h)
        start_w = random.randint(0, w - crop_w)
        end_h = start_h + crop_h
        end_w = start_w + crop_w
        image = image[start_h:end_h, start_w:end_w]
        label = label[start_h:end_h, start_w:end_w]
        return image, label

    def _blur(self, image, label):
        # Gaussian Blud (sigma between 0 and 1.5)
        sigma = random.random() * 1.5
        ksize = int(3.3 * sigma)
        ksize = ksize + 1 if ksize % 2 == 0 else ksize
        image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
        return image, label

    def _flip(self, image, label):
        # Random H flip
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            label = np.fliplr(label).copy()
        return image, label

    def _resize(self, image, label, bigger_side_to_base_size=True):
        if isinstance(self.base_size, int):
            h, w, _ = image.shape
            if self.scale:
                longside = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
                # longside = random.randint(int(self.base_size*0.5), int(self.base_size*1))
            else:
                longside = self.base_size

            if bigger_side_to_base_size:
                h, w = (longside, int(1.0 * longside * w / h + 0.5)) if h > w else (
                int(1.0 * longside * h / w + 0.5), longside)
            else:
                h, w = (longside, int(1.0 * longside * w / h + 0.5)) if h < w else (
                int(1.0 * longside * h / w + 0.5), longside)
            image = np.asarray(Image.fromarray(np.uint8(image)).resize((w, h), Image.BICUBIC))
            label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
            return image, label

        elif (isinstance(self.base_size, list) or isinstance(self.base_size, tuple)) and len(self.base_size) == 2:
            h, w, _ = image.shape
            if self.scale:
                scale = random.random() * 1.5 + 0.5  # Scaling between [0.5, 2]
                h, w = int(self.base_size[0] * scale), int(self.base_size[1] * scale)
            else:
                h, w = self.base_size
            image = np.asarray(Image.fromarray(np.uint8(image)).resize((w, h), Image.BICUBIC))
            label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
            return image, label

        else:
            raise ValueError

    def _val_augmentation(self, image, label):
        if self.base_size is not None:
            image, label = self._resize(image, label)
            image = self.normalize(self.to_tensor(Image.fromarray(np.uint8(image))))
            return image, label

        image = self.normalize(self.to_tensor(Image.fromarray(np.uint8(image))))
        return image, label

    def _augmentation(self, image, label):
        h, w, _ = image.shape

        if self.base_size is not None:
            image, label = self._resize(image, label)

        if self.crop_size is not None:
            image, label = self._crop(image, label)

        if self.flip:
            image, label = self._flip(image, label)

        image = Image.fromarray(np.uint8(image))
        image = self.jitter_tf(image) if self.jitter else image

        return self.normalize(self.to_tensor(image)), label

    @staticmethod
    def get_img_hash_dict(img_list):
        """
        将文件路径通过md5加密
        return [md5 encode],{md5 encode:'img_path'}
        """
        hash_list = []
        result_neg = {}
        md5 = hashlib.md5()
        for img_path in img_list:
            md5.update(img_path.encode('utf-8'))
            hash_list.append(md5.hexdigest())
            result_neg[md5.hexdigest()] = img_path
        sorted_img_dict = sorted(hash_list)
        images_list = [result_neg.get(hash_val) for hash_val in sorted_img_dict]
        return images_list