from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
def LabelToLongTensor(label):
    #将各个像素值转换为对应类，生成二维数组 label ，大小为 [ H , W ]
    GT = np.copy(label)
    GT_label = np.zeros((GT.shape[0], GT.shape[1]), dtype='int8')
    for i in range(GT.shape[0]):
        for j in range(GT.shape[1]):
            for k in range(class_len):
                if (GT[i, j] == class_color[k]).all():
                    GT_label[i, j] = k
                else:
                    continue
    # label = torch.from_numpy(GT_label).long()
    return GT_label
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
img_dir = "F:\数据集\ISPRS2D_Potsdam_Vahingen\Vahingen\Small_size\Size_256\labels\\"
img_ids = os.listdir(img_dir)
len = len(img_ids)
img_list = []
i = 0
for image_id in img_ids:
    if os.path.isfile(os.path.join(img_dir, image_id)):
        file_name, _ = os.path.splitext(image_id)
        img_path = img_dir + file_name + ".png"
        img_list.append(img_path)
# img_path = img_list[1]
        img = np.asarray(Image.open(img_path))
        GT_label = LabelToLongTensor(img)
        img_path_2 = img_path.replace('labels', 'GTs')
        # cv.imwrite(img_path_2, GT_label)
        GT_label = Image.fromarray(GT_label)
        GT_label.save(img_path_2)
        if i % 50 == 0:
            print(f"Save the {i}/{len} succefully!")
        i = i + 1
# img_path = img_dir + "1.png"
# img = np.asarray(Image.open(img_path))
# img_path_2 = img_path.replace('labels', 'GTs')
# GT_label = np.asarray(Image.open(img_path_2))
# plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.imshow(GT_label)
# GT_label.show()
plt.show()

