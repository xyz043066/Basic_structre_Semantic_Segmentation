import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data import custom_transforms as tr
import numpy as np
from PIL import Image
import cv2 as cv
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
class MyDataSet(data.Dataset):
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
        # if self.train:
        #     self.images_list = images_list[:int(0.8*self.images_len)]
        # else:
        #     self.images_list = images_list[int(0.8*self.images_len):]
        if self.train:
            self.images_list = images_list[int(0.25*self.images_len):]
        else:
            self.images_list = images_list[:int(0.25*self.images_len)]
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
        label_item_path = img_item_path.replace('images', 'GTs')
        img = Image.open(img_item_path)
        label = Image.open(label_item_path)
        sample = {'image': img, 'label': label}
        if self.is_transform is not None:
            sample = self.transform(sample)

        return sample['image'], sample['label']

    def __len__(self):
        return len(self.images_list)

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


class My_dataloader(DataLoader):
    def __init__(self, opt):
        self.dataset = MyDataSet(opt)
        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': opt.batch_size,
            'shuffle': opt.shuffle,
            'num_workers': opt.num_workers,
            'pin_memory': opt.pin_memory
        }
        super(My_dataloader, self).__init__(**self.init_kwargs)


if __name__ == "__main__":
    print("Creating dataset Successfully!")

# import os
# import torch
# import torch.utils.data as data
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# import numpy as np
# import cv2 as cv
# import hashlib
#
# classes = [100, 200, 300, 400, 500, 600, 700, 800]
# class_len = len(classes)
# mean = [0.41189489566336, 0.4251328133025, 0.4326707089857]
# std = [0.27413549931506, 0.28506257482912, 0.28284674400252]
#
# class MyDataSet(data.Dataset):
#
#     def __init__(self, self.images_dir, labels_dir, transform=True, train=True, val=False):
#         self.train = train
#         self.val = val
#         self.ids = os.listdir(self.images_dir)
#         images_list = []
#         for image_id in self.ids:
#             if os.path.isfile(os.path.join(self.images_dir, image_id)):
#                 file_name, _ = os.path.splitext(image_id)
#                 images_list.append(self.images_dir + file_name + ".tif")
#         images_hash_list, images_hash_dict = self.get_img_hash_dict(images_list)
#         # images_list.sort(key=lambda x: x.split('.')[-2].split('\\')[-1])
#         # labels_list.sort(key=lambda x: x.split('.')[-2].split('\\')[-1])
#         sorted_img_dict = sorted(images_hash_list)  # 通过img_path的md5值对img_path进行排序
#         images_list = [images_hash_dict.get(hash_val) for hash_val in sorted_img_dict]
#         images_len = len(images_list)
#         if self.train:
#             self.images_list = images_list[:int(0.99*images_len)]
#         elif self.val:
#             self.images_list = images_list[int(0.99*images_len):]
#         if transform:
#             normalize = transforms.Normalize(mean=mean, std=std)
#             self.transform = transforms.Compose([transforms.ToTensor(), normalize])
#     def __getitem__(self, index):
#         img_item_path = self.images_list[index]
#         label_item_path = img_item_path.replace('image', 'label').replace('tif', 'png')
#         img = cv.imread(img_item_path)
#         label = cv.imread(label_item_path, cv.IMREAD_UNCHANGED)
#         if self.transform is not None:
#             img = self.transform(img)
#         label = LabelToLongTensor(label)
#         return img, label
#
#     def __len__(self):
#         return len(self.images_list)
#
#     @staticmethod
#     def get_img_hash_dict(img_list):
#         """
#         将文件路径通过md5加密
#         return [md5 encode],{md5 encode:'img_path'}
#         """
#         hash_list = []
#         result_neg = {}
#         md5 = hashlib.md5()
#         for img_path in img_list:
#             md5.update(img_path.encode('utf-8'))
#             hash_list.append(md5.hexdigest())
#             result_neg[md5.hexdigest()] = img_path
#         return hash_list, result_neg
#
#
# def LabelToLongTensor(label):
#     #将各个像素值转换为对应类，生成二维数组 label ，大小为 [ H , W ]
#     GT_label = np.copy(label)
#     for m in classes:
#         GT_label[GT_label == m] = classes.index(m)
#     GT_label = GT_label.astype(np.int16)
#     label = torch.from_numpy(GT_label).long()
#     return label
#
# if __name__ == "__main__":
#     print("Successful!")
