import os
import numpy as np
import torch
import torch.nn as nn

# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
device = torch.device('cuda')
# 精度指标
class Metric:
    """
    Class to calculate mean-iou using fast_confusion_matrix method
    """
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)  # 开始定义时初始化混淆矩阵
        self.metric_dict = {'IOU': [], 'MIOU': [], 'F1_score': [], 'OA': [], 'Avg_F1': []}
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

    def _update_metrics(self):
        iou, fwiou = self.Metric.Frequency_Weighted_Intersection_over_Union()
        miou = self.Mean_Intersection_over_Union()
        oa = self.Pixel_Accuracy_Class()
        f1_score, f1_score_avg = self.F1_score()
        self.metric_dict['IOU'].append(list(iou))
        self.metric_dict['MIOU'].append(list(miou))
        self.metric_dict['OA'].append(list(oa))
        self.metric_dict['F1_score'].append(list(f1_score))
        self.metric_dict['Avg_F1'].append(list(f1_score_avg))

    # 训练时添加每个Batch到混淆矩阵当中
    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image.flatten(), pre_image.flatten())
    # 重置混淆矩阵
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

# Dice loss
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1

        input_flat = input.view(-1)
        target_flat = target.view(-1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum() + smooth*N).item() / (input_flat.sum() + target_flat.sum() + smooth*N).item()
        loss = torch.tensor(1 - loss / N, requires_grad=True)
        # loss = loss.type(torch.float).requires_grad_()

        return loss
# 学习率调整
def adjust_learning_rate(lr, decay, optimizer, cur_epoch, n_epochs):
    """Sets the learning rate to the initially
        configured `lr` decayed by `decay` every `n_epochs`"""
    new_lr = lr
    if 50 < cur_epoch < 100:
        new_lr = 1e-4
    elif 100 < cur_epoch < 150:
        new_lr = 1e-5
    elif cur_epoch > 150:
        new_lr = 1e-6
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

# 预测输出
def get_predictions(output_batch):
    #bs, c, h, w = output_batch.size()
    tensor = output_batch.detach()
    # out = tensor.cpu().squeeze(1)
    # out[out >= 0.5] = 1
    # out[out < 0.5] = 0
    out = tensor.cpu().argmax(1)
    return out

# 保持加载模型及优化器参数
def save_weights(model, optimizer, epoch, loss, FWIOU):
    WEIGHTS_PATH = 'F:/machine_learning/Unet/weights/'
    weights_fname = 'weights-%d-%.4f-%.4f.pth' % (epoch, loss, FWIOU)
    weights_fpath = os.path.join(WEIGHTS_PATH, weights_fname)
    torch.save({
            'startEpoch': epoch,
            'loss': loss,
            'FWIOU': FWIOU,
            'Net': model.state_dict(),
            'optimizer': optimizer.state_dict()}, weights_fpath)

def load_weights(model, optimizer, fpath):
    print("loading weights '{}'".format(fpath))
    checkpoint = torch.load(fpath)
    startEpoch = checkpoint['startEpoch']
    model.load_state_dict(checkpoint['Net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("loaded weights (lastEpoch {}, loss {}, FWIOU {})".format(startEpoch, checkpoint['loss'], checkpoint['FWIOU']))
    return startEpoch

# 更新Metrics字典
def _update_metrics(Metric, metric_dict):
    iou, fwiou = Metric.Frequency_Weighted_Intersection_over_Union()
    miou = Metric.Mean_Intersection_over_Union()
    oa = Metric.Pixel_Accuracy_Class()
    f1_score, f1_score_avg = Metric.F1_score()
    metric_dict['IOU'].append(list(iou))
    metric_dict['MIOU'].append(list(miou))
    metric_dict['OA'].append(list(oa))
    metric_dict['F1_score'].append(list(f1_score))
    metric_dict['Avg_F1'].append(list(f1_score_avg))
    return metric_dict

# 模型训练函数
def train(model, train_loader, optimizer, criterion, epoch):
    model.train()
    metric_dict = {'IOU': [], 'MIOU': [], 'F1_score': [], 'OA': [], 'Avg_F1': []}
    FWIOU_Metric = Metric(6)
    train_batch_sum = len(train_loader)
    trn_loss = 0
    i = 0
    for idx, data in enumerate(train_loader):
        i = i + 1
        inputs = data[0].cuda()
        # targets = data[1].cuda()
        targets = torch.LongTensor(data[1]).cuda()
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        trn_loss += loss.item()
        pred = get_predictions(output)
        FWIOU_Metric.add_batch(targets.cpu().numpy().astype(int), pred.cpu().numpy().astype(int))
        metric_dict = _update_metrics(FWIOU_Metric, metric_dict)
        # 监测模型训练情况
        if i % 50 == 0:
            print("Epoch_idx:{}  batch_idx:{}/{}  AvgLoss:{:.4f}  AvgAcc:{:.4f}".format(epoch, i, train_batch_sum, trn_loss / i, FWIOU_Metric.Pixel_Accuracy()))
            IoU, FWIoU = FWIOU_Metric.Frequency_Weighted_Intersection_over_Union()
            for k in range(len(IoU)):
                print("IOU:Class {} => {:.4f}".format(k, IoU[k]))
            print("FWIOU: {:.4f}".format(FWIoU))
    # print('- - - - - - save {}th epoch weights - - - - - -'.format(epoch))
    # save_weights(model, optimizer, epoch, trn_loss / i, FWIoU)
    IoU, FWIoU = FWIOU_Metric.Frequency_Weighted_Intersection_over_Union()
    for k in range(len(IoU)):
        print("The {}th epoch IOU:Class {} => {:.4f}".format(epoch, k, IoU[k]))
    print("The {}th epoch FWIOU: {:.4f}".format(epoch, FWIoU))
    trn_loss /= len(train_loader)
    return trn_loss, FWIOU_Metric.Pixel_Accuracy(), FWIoU

# 模型测试函数
def val(model, val_loader, criterion, epoch=1):
    model.eval()
    FWIOU_Metric = Metric(6)
    val_batch_sum = len(val_loader)
    i = 0
    val_loss = 0
    with torch.no_grad():
      for data, target in val_loader:
        i = i + 1
        data = data.cuda()
        target = target.cuda()
        output = model(data)
        val_loss += criterion(output, target).item()
        pred = get_predictions(output)
        FWIOU_Metric.add_batch(target.cpu().numpy().astype(int), pred.cpu().numpy().astype(int))
        if i % 100 == 0:
            print("Epoch_idx:{}  batch_idx:{}/{}  AvgLoss:{:.4f}  AvgAcc:{:.4f}".format(epoch, i, val_batch_sum, val_loss / i, FWIOU_Metric.Pixel_Accuracy()))
            IoU, FWIoU = FWIOU_Metric.Frequency_Weighted_Intersection_over_Union()
            for k in range(len(IoU)):
                print("IOU:Class {} => {:.4f}".format(k, IoU[k]))
            print("FWIOU: {:.4f}".format(FWIoU))
    IoU, FWIoU = FWIOU_Metric.Frequency_Weighted_Intersection_over_Union()
    for k in range(len(IoU)):
        print("The {}th epoch IOU:Class {} => {:.4f}".format(epoch, k, IoU[k]))
    print("The {}th epoch FWIOU: {:.4f}".format(epoch, FWIoU))
    val_loss /= len(val_loader)
    return val_loss, FWIOU_Metric.Pixel_Accuracy(), FWIoU

if __name__ == '__main__':
    print('Successfully!')