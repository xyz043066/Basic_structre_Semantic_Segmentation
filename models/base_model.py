import os
import numpy as np
import torch
import torch.nn as nn
import time
import torch.optim as optim
from models.Deeplab.deeplab import *
from models.Deeplab.DUNet import *
from models.Deeplab.MSNet import *
from models.Deeplab.AFNet import *
from models.Deeplab.HMANet import *
from models.Deeplab.deeplab_ocr import *
from models.DenseNet.Net import *
from models.HRNet.seg_hrnet_ocr import *
from models.HRNet.seg_hrnet import *
from models.FCN.FCNs import *
from models.UNet.Unet import *
from models.PSPNet.pspnet import *
from utils.loss import *
from utils.lr_scheduler import *
from utils.metrics import *
from options.config import config
from options.config import update_config


class Model(object):
    def __init__(self, opt, train_loader, val_loader):
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
        # self.device = torch.device('cuda:' + str(opt.gpu_ids[0]))
        self.class_num = opt.class_num
        self.lr = opt.lr
        self.lr_policy = opt.lr_policy
        self.weight_decay = opt.weight_decay
        self.save_epoch_freq = opt.save_epoch_freq
        self.display_freq = opt.display_freq
        self.model_name = opt.model_name
        self.optimizer_name = opt.optimizer_name
        self.criterion_name = opt.criterion_name
        self.backbone = opt.backbone
        self.output_stride = opt.output_stride
        self.checkpoints_dir = opt.checkpoints_dir
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.n_epochs = opt.n_epochs
        self.train_batch_sum = len(self.train_loader)
        self.val_batch_sum = len(self.val_loader)
        self.best_F1_score = 0
        self.best_OA = 0
        self.best_MIOU = 0
        self.epoch = 0
        # update_config(config, opt.hrnet_model_set)
        self._create_model()
        self._create_optimizer()
        self._create_criterion()
        # if self.optimizer_name != 'adam':
        self._adjust_lr()
        if opt.fpath:
            self.epoch = self.load_model(opt.fpath)


    def _create_model(self):
        if self.model_name == 'FC_DenseNet':
            self.model = FCDenseNet67(n_classes=self.class_num).cuda()
        elif self.model_name == 'Deeplabv3+':
            self.model =  DeepLab(num_classes=self.class_num, backbone=self.backbone, output_stride=self.output_stride).cuda()
        elif self.model_name == 'OCRNet':
            self.model =  DeepLab_OCR(num_classes=self.class_num, backbone=self.backbone, output_stride=self.output_stride).cuda()
        elif self.model_name == 'DUNet':
            self.model =  DUNet(num_classes=self.class_num, backbone=self.backbone, output_stride=self.output_stride).cuda()
        elif self.model_name == 'MSNet':
            self.model =  MSNet(num_classes=self.class_num, backbone=self.backbone, output_stride=self.output_stride).cuda()
        elif self.model_name == 'AFNet':
            self.model = AFNet(num_classes=self.class_num, backbone=self.backbone, output_stride=self.output_stride).cuda()
        elif self.model_name == 'HMANet':
            self.model = HMANet(num_classes=self.class_num, backbone=self.backbone, output_stride=self.output_stride).cuda()
        elif self.model_name == 'HRNet':
            self.model = HRNet(config).cuda()
        elif self.model_name == 'PSPNet':
            self.model = PSPNet(num_classes=self.class_num, backbone=self.backbone).cuda()
        elif self.model_name == "HRNet_OCR":
            self.model = HRNet_OCR(config).cuda()
        elif self.model_name == 'FCNs':
            self.model = FCNs(n_class=self.class_num).cuda()
        elif self.model_name == 'UNet':
            self.model = ResNetUNet(n_class=self.class_num).cuda()
        else:
            raise NameError("Please input the proper name of used model")

    def _create_optimizer(self):
        if self.optimizer_name == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=0.9)
        else:
            raise NameError("Please input the proper name of used optimizer")

    def _create_criterion(self):
        if self.criterion_name == 'CE_loss':
            self.criterion = CE_loss()
        elif self.criterion_name == 'Focal_loss':
            alpha = get_alpha(self.train_loader)
            self.criterion = FocalLoss(apply_nonlin=softmax_helper, alpha=alpha, gamma=2, smooth=1e-5)
        else:
            raise NameError("Please input the proper name of used criterion")

    def save_model(self, epoch, loss, OA, MIOU, F1_score):
        WEIGHTS_PATH = self.checkpoints_dir
        weights_fname = 'weights-%d--loss_%.4f--OA_%.4f--MIOU_%.4f--F1_score_%.4f.pth' % (epoch, loss, OA, MIOU, F1_score)
        weights_fpath = os.path.join(WEIGHTS_PATH, weights_fname)
        torch.save({
            'startEpoch': epoch,
            'loss': loss,
            'OA': OA,
            'MIOU': MIOU,
            'F1_score': F1_score,
            'Net': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()}, weights_fpath)

    def load_model(self, fpath):
        print("loading weights '{}'".format(fpath))
        checkpoint = torch.load(fpath)
        startEpoch = checkpoint['startEpoch']
        self.model.load_state_dict(checkpoint['Net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"loaded weights (lastEpoch {startEpoch}, loss {checkpoint['loss']}, OA {checkpoint['OA']}"
              f" MIOU {checkpoint['MIOU']} F1_score {checkpoint['F1_score']})")
        return startEpoch

    def _adjust_lr(self):
        if self.lr_policy == 'multiStep':
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[20, 40, 80, 150], gamma=0.1, last_epoch=-1)
        elif self.lr_policy == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10)
        elif self.lr_policy in ['step', 'cos', 'poly']:
            self.scheduler = LR_Scheduler(self.lr_policy, self.lr, self.n_epochs, self.train_batch_sum, lr_step=20)
        else:
            raise NameError("Please input the proper name of used scheduler")
    # ????????????
    def get_prediction(self, output_batch):
        # bs, c, h, w = output_batch.size()
        tensor = output_batch.detach()
        # out = tensor.cpu().squeeze(1)
        # out[out >= 0.5] = 1
        # out[out < 0.5] = 0
        out = tensor.cpu().argmax(1)
        return out

    # ????????????
    def train(self, Metric, visualizer, epoch):
        # self.epoch = self.epoch + epoch
        self.model.train()
        Metric.reset()
        # train_batch_sum = len(self.train_loader)
        since = time.time()
        trn_loss = 0
        i = 0
        for idx, data in enumerate(self.train_loader):
            t_begin = time.time()
            i = i + 1
            inputs = data[0].cuda()
            # targets = data[1].cuda()
            targets = torch.LongTensor(data[1]).cuda()
            self.optimizer.zero_grad()
            output = self.model(inputs)
            loss = self.criterion(output, targets)
            loss.backward()
            self.optimizer.step()
            trn_loss += loss.item()
            pred = self.get_prediction(output)
            Metric.add_batch(targets.cpu().numpy().astype(int), pred.cpu().numpy().astype(int))
            MIOU = Metric.Mean_Intersection_over_Union()
            self.scheduler(self.optimizer, i, epoch, MIOU)
            # Metric.update_metrics(trn_loss/i)
            # ????????????????????????
            if i % (self.train_batch_sum//50) == 0:
                Metric.update_metrics(trn_loss / i)
                t_show = time.time()
                metric_dict = Metric.metric_dict
                # visualizer.print_current_metrics(self.epoch, i, metric_dict, t_show, t_begin)
                visualizer.plot_current_metrics(epoch, i/self.train_batch_sum, metric_dict, mode='train')
                visualizer.plot_current_images(target=targets, pred=pred, mode='train')
                F1_score, Avg_F1_score = Metric.F1_score()
                IoU, FWIoU = Metric.Frequency_Weighted_Intersection_over_Union()
                MIOU = Metric.Mean_Intersection_over_Union()
                OA = Metric.Pixel_Accuracy()
                print("Epoch_idx:{}  batch_idx:{}/{}  Loss:{:.4f}  OA:{:.4f} Avg_F1-score:{:.4f} MIOU:{:.4f} FWIOU:{:.4f}".
                      format(self.epoch + epoch, i, self.train_batch_sum, trn_loss / i, OA, Avg_F1_score, MIOU, FWIoU))
                # for k in range(len(IoU)):
                #     print("Class {} => IOU:{:.4f} F1-score:{:.4f}".format(k, IoU[k], F1_score[k]))
                # print("MIOU:{:.4f}  FWIOU: {:.4f}".format(MIOU, FWIoU))
        F1_score, Avg_F1_score = Metric.F1_score()
        IoU, FWIoU = Metric.Frequency_Weighted_Intersection_over_Union()
        MIOU = Metric.Mean_Intersection_over_Union()
        OA = Metric.Pixel_Accuracy()
        print("Epoch_idx:{}  Loss:{:.4f}  OA:{:.4f} Avg_F1-score:{:.4f} MIOU:{:.4f} FWIOU:{:.4f} ".
              format(self.epoch + epoch, trn_loss / i, OA, Avg_F1_score, MIOU, FWIoU))
        # for k in range(len(IoU)):
        #     print("Class {} => IOU:{:.4f} F1-score:{:.4f}".format(k, IoU[k], F1_score[k]))
        # print("MIOU:{:.4f}  FWIOU: {:.4f}".format(MIOU, FWIoU))
        # print('- - - - - - save {}th epoch weights - - - - - -'.format(epoch))
        # save_weights(model, optimizer, epoch, trn_loss / i, FWIoU)
        # IoU, FWIoU = Metric.Frequency_Weighted_Intersection_over_Union()
        # for k in range(len(IoU)):
        #     print("The {}th epoch IOU:Class {} => {:.4f}".format(self.epoch, k, IoU[k]))
        # print("The {}th epoch FWIOU: {:.4f}".format(self.epoch, FWIoU))
        # trn_loss /= len(self.train_loader)
        # OA = Metric.Pixel_Accuracy_Class()
        # MIOU = Metric.Mean_Intersection_over_Union()
        # F1_score, Avg_F1_score = Metric.F1_score()
        # lr = self.optimizer.param_groups[0]['lr']
        # if Avg_F1_score > self.best_F1_score:
        #     if self.epoch % 5 == 0
        #         self.save_model(trn_loss/self.train_batch_sum, OA, MIOU, Avg_F1_score)
        #     self.best_F1_score = Avg_F1_score
        #     print(f"The best results =>  epoch:{epoch}   OA:{OA}   MIOU:{MIOU}   Avg_F1_score:{Avg_F1_score}   lr:{lr}")
        time_elapsed = time.time() - since
        print('Train Time {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    def val(self, Metric, visualizer, epoch):
        # self.epoch = self.epoch + epoch
        Metric.reset()
        self.model.eval()
        since = time.time()
        i = 0
        val_loss = 0
        with torch.no_grad():
            for data, target in self.val_loader:
                i = i + 1
                data = data.cuda()
                target = target.cuda()
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                pred = self.get_prediction(output)
                Metric.add_batch(target.cpu().numpy().astype(int), pred.cpu().numpy().astype(int))
                if i % (self.val_batch_sum//4) == 0:
                    Metric.update_metrics(val_loss / i)
                    t_show = time.time()
                    metric_dict = Metric.metric_dict
                    # visualizer.print_current_metrics(self.epoch, i, metric_dict, t_show, t_begin)
                    visualizer.plot_current_metrics(epoch, i / self.val_batch_sum, metric_dict, mode='test')
                    visualizer.plot_current_images(target=target, pred=pred, mode='test')
                    F1_score, Avg_F1_score = Metric.F1_score()
                    IoU, FWIoU = Metric.Frequency_Weighted_Intersection_over_Union()
                    MIOU = Metric.Mean_Intersection_over_Union()
                    OA = Metric.Pixel_Accuracy()
                    print("Epoch_idx:{}  batch_idx:{}/{}  Loss:{:.4f}  OA:{:.4f} Avg_F1-score:{:.4f} MIOU:{:.4f} FWIoU:{:.4f}".
                          format(self.epoch+epoch, i, self.val_batch_sum, val_loss / i, OA, Avg_F1_score, MIOU, FWIoU))
                    # for k in range(len(IoU)):
                    #     print("Class {} => IOU:{:.4f} F1-score:{:.4f}".format(k, IoU[k], F1_score[k]))
                    # print("MIOU:{:.4f}  FWIOU: {:.4f}".format(MIOU, FWIoU))
        visualizer.print_current_metrics(epoch, Metric)
        F1_score, Avg_F1_score = Metric.F1_score()
        IoU, FWIoU = Metric.Frequency_Weighted_Intersection_over_Union()
        MIOU = Metric.Mean_Intersection_over_Union()
        OA = Metric.Pixel_Accuracy()
        print("Epoch_idx:{}  Loss:{:.4f}  OA:{:.4f} Avg_F1-score:{:.4f} MIOU:{:.4f} FWIOU:{:.4f} ".
              format(self.epoch+epoch, val_loss / i, OA, Avg_F1_score, MIOU, FWIoU))
        lr = self.optimizer.param_groups[0]['lr']
        if MIOU > self.best_MIOU:
            if(epoch < (self.n_epochs//10)) and (MIOU - self.best_MIOU > 0.003):
                self.save_model(self.epoch+epoch, val_loss/self.val_batch_sum, OA, MIOU, Avg_F1_score)
                self.best_MIOU = MIOU
                print(f"The best results =>  epoch:{self.epoch + epoch}   OA:{OA}   MIOU:{MIOU}   Avg_F1_score:{Avg_F1_score}   FWIoU:{FWIoU} lr:{lr}")
                print("The confusion matrix:\n", Metric.confusion_matrix)
            elif (epoch > (self.n_epochs//10)) and (epoch < (self.n_epochs//8)) and (MIOU - self.best_MIOU > 0.001):
                self.save_model(self.epoch+epoch, val_loss / self.val_batch_sum, OA, MIOU, Avg_F1_score)
                self.best_MIOU = MIOU
                print(f"The best results =>  epoch:{self.epoch + epoch}   OA:{OA}   MIOU:{MIOU}   Avg_F1_score:{Avg_F1_score}  FWIoU:{FWIoU}  lr:{lr}")
                print("The confusion matrix:\n", Metric.confusion_matrix)
            elif (epoch > (self.n_epochs//8)) and (epoch < (self.n_epochs//4)) and (MIOU - self.best_MIOU > 0.0003):
                self.save_model(self.epoch+epoch, val_loss / self.val_batch_sum, OA, MIOU, Avg_F1_score)
                self.best_MIOU = MIOU
                print(f"The best results =>  epoch:{self.epoch + epoch}   OA:{OA}   MIOU:{MIOU}   Avg_F1_score:{Avg_F1_score}  FWIoU:{FWIoU} lr:{lr}")
                print("The confusion matrix:\n", Metric.confusion_matrix)
            elif epoch > (self.n_epochs // 4):
                self.save_model(self.epoch+epoch, val_loss / self.val_batch_sum, OA, MIOU, Avg_F1_score)
                self.best_MIOU = MIOU
                print(f"The best results =>  epoch:{self.epoch + epoch}   OA:{OA}   MIOU:{MIOU}   Avg_F1_score:{Avg_F1_score}  FWIoU:{FWIoU} lr:{lr}")
                print("The confusion matrix:\n", Metric.confusion_matrix)
            # self.best_F1_score = Avg_F1_score
            # print(f"The best results =>  epoch:{self.epoch+epoch}   OA:{Avg_OA}   MIOU:{MIOU}   Avg_F1_score:{Avg_F1_score}   lr:{lr}")
            # print("The confusion matrix:\n", Metric.confusion_matrix)
        if epoch == self.n_epochs - 1:
            self.save_model(self.epoch+epoch, val_loss / self.val_batch_sum, OA, MIOU, Avg_F1_score)
        time_elapsed = time.time() - since
        print('val Time {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
