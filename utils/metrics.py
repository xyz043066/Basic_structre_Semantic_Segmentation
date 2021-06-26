import numpy as np


class Evaluator(object):
    """
    Class to calculate mean-iou using fast_confusion_matrix method
    """
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)  # 开始定义时初始化混淆矩阵
        self.metric_dict = {'IOU': [], 'MIOU': [], 'FWIOU': [], 'F1_score': [], 'OA': [], 'Avg_F1': [], 'Loss': []}
    # 像素精度 PA
    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc
    # 均像素精度 PAC
    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc
    # 均交并比 MIoU
    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU
    # 加权均交并比 FWIoU
    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iou = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = np.nansum(freq[freq > 0] * iou[freq > 0])
        return iou, FWIoU
    # 计算F1-score
    def F1_score(self):
        Precision = np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, 0)
        Recall = np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, 1)
        F1_score = 2*Precision*Recall/(Precision+Recall)
        Avg_F1_score = np.nanmean(F1_score)
        return F1_score, Avg_F1_score
    # 计算混淆矩阵
    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def update_metrics(self, loss):
        iou, fwiou = self.Frequency_Weighted_Intersection_over_Union()
        miou = self.Mean_Intersection_over_Union()
        oa = self.Pixel_Accuracy_Class()
        f1_score, f1_score_avg = self.F1_score()
        self.metric_dict['IOU'].append(iou.tolist())
        self.metric_dict['MIOU'].append(miou.tolist())
        self.metric_dict['FWIOU'].append(fwiou.tolist())
        self.metric_dict['OA'].append(oa.tolist())
        self.metric_dict['F1_score'].append(f1_score.tolist())
        self.metric_dict['Avg_F1'].append(f1_score_avg.tolist())
        self.metric_dict['Loss'].append(loss)

    # 训练时添加每个Batch到混淆矩阵当中
    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image.flatten(), pre_image.flatten())
    # 重置混淆矩阵
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
