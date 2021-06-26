import time
import torch.optim as optim
from utils import utils
from data.base_dataloader import *
from models.Deeplab.deeplab import *
from options.val_options import *
from options.train_val_options import *
from models.DenseNet.Net import *
from models.base_model import *
from utils.metrics import *
from utils.visualizer import *


def main():
    # Set the basic options for train and validation
    opt_train = TrainOptions(is_train=True).parse()
    opt_val = TrainOptions(is_train=False).parse()
    # Set the metrics for the evaluation of accuracy
    Metric_train = Evaluator(opt_train.class_num)
    Metric_val = Evaluator(opt_train.class_num)
    # Use the visdom for visualizaiton
    visualizer = Visualizer(opt_train)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt_train.gpu_ids)
    # device = torch.device('cuda')
    train_loader = My_dataloader(opt_train)
    val_loader = My_dataloader(opt_val)
    model = Model(opt_train, train_loader, val_loader)
    #-------------显示数据集信息-----------------

    print("Train: %d" % len(train_loader.dataset))
    print("Validation: %d" % len(val_loader.dataset))
    print("Classes: %d" % class_len)
    inputs, targets = next(iter(train_loader))
    print("Inputs size: ", inputs.size())
    print("Targets size: ", targets.size())
    print("Inputs type:", inputs.type())
    print("Targets type:", targets.type())


    #------------加载模型、损失函数、优化器----------------

    # vgg_model = VGGNet(requires_grad=True)
    # model = FCN32s(pretrained_net=vgg_model, n_class=8).to(device)
    # model = DeepLab(num_classes=6, backbone='resnet-18', output_stride=16).to(device)
    # model = FCDenseNet67(n_classes=6).to(device)
    # # model = nn.DataParallel(model, device_ids=[0, 1])
    # criterion = nn.CrossEntropyLoss().to(device)
    # # optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    print(model.model)

    # weights_path = '/gs/home/xuyizhe/Basic_Structure/weights/weights-188-0.1914-0.8659.pth'
    # epoch_open = utils.load_weights(model, optimizer, weights_path)
    epoch_open = 0
    val_best = 0

    #writer.add_graph(model, (inputs.to(device),))
    for epoch in range(0, opt_train.n_epochs):
        ### Train ###
        print('- - - - - - - - - - - -  Beginning Train - - - - - - - - - - - - ')
        model.train(Metric_train, visualizer, epoch)
        print('- - - - - - - - - - - -  Beginning validation - - - - - - - - - - - - ')
        model.val(Metric_val, visualizer, epoch)

if __name__ == '__main__':
    main()
    print("Successfully train and test !!")


