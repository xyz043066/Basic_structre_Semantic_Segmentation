import numpy as np
import os
import sys
import ntpath
import time
from . import util, html
from subprocess import Popen, PIPE
from data.base_dataloader import *
import matplotlib.pyplot as plt

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


class Visualizer(object):
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test_256 options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training metrics
        """
        self.opt = opt  # cache the option
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.port = opt.display_port
        self.saved = False
        self.string = ''
        if self.display_id > 0:  # connect to a visdom server given <display_port> and <display_server>
            import visdom
            self.ncols = opt.display_ncols
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env)
            if not self.vis.check_connection():
                self.create_visdom_connections()

        if self.use_html:  # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        # create a logging file to store training metrics
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, epoch, save_result):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        if self.display_id > 0:  # show images in the browser using visdom
            ncols = self.ncols
            if ncols > 0:        # show all the images in one visdom panel
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                        table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)  # create a table css
                # create a table of images.
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                for label, image in visuals.items():
                    image_numpy = util.tensor2im(image)
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                try:
                    self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                    padding=2, opts=dict(title=title + ' images'))
                    label_html = '<table>%s</table>' % label_html
                    self.vis.text(table_css + label_html, win=self.display_id + 2,
                                  opts=dict(title=title + ' labels'))
                except VisdomExceptionBase:
                    self.create_visdom_connections()

            else:     # show each image in a separate visdom panel;
                idx = 1
                try:
                    for label, image in visuals.items():
                        image_numpy = util.tensor2im(image)
                        self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                       win=self.display_id + idx)
                        idx += 1
                except VisdomExceptionBase:
                    self.create_visdom_connections()

        if self.use_html and (save_result or not self.saved):  # save images to an HTML file if they haven't been saved.
            self.saved = True
            # save images to the disk
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, image_numpy in visuals.items():
                    image_numpy = util.tensor2im(image)
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    def view_annotated(self, tensor, nc=class_len, plot=True):
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

    def plot_current_images(self, target, pred, mode): # 此时target,pred shape皆为 [b, h, w]
        error_img = (pred.numpy() == target.cpu().numpy()).astype('uint8')  # 计算误差曲线
        error_img[error_img > 0] = 255  # 将分类错误的像素给置为0 shape: [b, h, w]
        error_img = np.stack([error_img, error_img, error_img], 1)   # shape: [b, 3, h, w]
        output_img = self.view_annotated(pred.detach(), class_len, False)  # shape: [b, 3, h, w]
        target_img = self.view_annotated(target.detach(), class_len, False) # shape: [b, 3, h, w]
        # 在这里显示每个batch的前两张图像
        images_col_1 = np.stack([target_img[0], output_img[0], error_img[0]], 0)  # shape: [1, 3, h, w] 每一行依次为target/pred/error_img
        images_col_2 = np.stack([target_img[1], output_img[1], error_img[1]], 0)
        images = np.concatenate([images_col_1, images_col_2])  # shape: [2, 3, h, w]
        # self.vis.images(images_col_1, win=mode + 'images1', nrow=3, opts=dict(title=mode + '_Visualization'))
        # self.vis.images(images_col_2, win=mode + 'images2', nrow=3, opts=dict(title=mode + '_Visualization'))
        self.vis.images(images, win=mode + 'images', nrow=3, opts=dict(title=mode+'_Visualization', width=900, height=600))
        # self.vis.images(images, win=mode + 'images', nrow=3, opts=dict(title=mode + '_Visualization'))

    def plot_current_metrics(self, epoch, counter_ratio, metrics, mode='train'):
        """display the current metrics on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            metrics (OrderedDict)  -- training metrics stored in the format of (name, float) pairs
        """
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'X_train': [], 'X_val': [], 'IOU': [], 'MIOU': [], 'FWIOU': [], 'F1_score': [], 'OA': [], 'Avg_F1': [], 'Loss': [], 'legend': list(metrics.keys())}
        if mode == 'train':
            self.plot_data['X_train'].append(epoch + counter_ratio)
            self.plot_data['X'] = self.plot_data['X_train']
        elif mode == 'val':
            self.plot_data['X_val'].append(epoch + counter_ratio)
            self.plot_data['X'] = self.plot_data['X_val']
        self.plot_data['IOU'] = metrics['IOU']
        self.plot_data['MIOU'] = metrics['MIOU']
        self.plot_data['FWIOU'] = metrics['FWIOU']
        self.plot_data['F1_score'] = metrics['F1_score']
        self.plot_data['OA'] = metrics['OA']
        self.plot_data['Loss'] = metrics['Loss']
        self.plot_data['Avg_F1'] = metrics['Avg_F1']
        num_Class = np.array(self.plot_data['IOU']).shape[1]
        try:
            self.vis.line(
                X=np.array(self.plot_data['X']),
                Y=np.array(self.plot_data['Loss']),
                opts={
                    'title': mode + ' Loss over time',
                    # 'legend': ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6'],
                    'xlabel': 'epoch',
                    'ylabel': 'Loss'},
                win=mode + ' Loss')
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * num_Class, 1),
                Y=np.array(self.plot_data['IOU']),
                opts={
                    'title': mode + ' IOU over time',
                    'legend': ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6'],
                    'xlabel': 'epoch',
                    'ylabel': 'IOU'},
                win=mode + ' IOU')
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * num_Class, 1),
                Y=np.array(self.plot_data['F1_score']),
                opts={
                    'title': mode + ' F1_score over time',
                    'legend': ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6'],
                    'xlabel': 'epoch',
                    'ylabel': 'F1_score'},
                win=mode + ' F1_score')
            y = np.stack([np.array(self.plot_data['MIOU']), np.array(self.plot_data['FWIOU']),
                          np.array(self.plot_data['OA']), np.array(self.plot_data['Avg_F1'])], 1)
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * 4, 1),
                Y=y,
                # Y=np.array(self.plot_data['F1_score']),
                opts={
                    'title': mode + ' Metrics over time',
                    'legend': ['MIOU', 'FWIOU', 'OA', 'Avg_F1'],
                    'xlabel': 'epoch',
                    'ylabel': 'Metrics'},
                win=mode + ' Metrics')
            # self.vis.line(
            #     X=np.array(self.plot_data['X']),
            #     Y=np.array(self.plot_data['OA']),
            #     opts={
            #         'title': mode + ' OA over time',
            #         # 'legend': ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6'],
            #         'xlabel': 'epoch',
            #         'ylabel': 'OA'},
            #     win=mode + ' OA')
            # self.vis.line(
            #     X=np.array(self.plot_data['X']),
            #     Y=np.array(self.plot_data['MIOU']),
            #     opts={
            #         'title': mode + ' MIOU over time',
            #         # 'legend': ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6'],
            #         'xlabel': 'epoch',
            #         'ylabel': 'MIOU'},
            #     win=mode + ' MIOU')
            # self.vis.line(
            #     X=np.array(self.plot_data['X']),
            #     Y=np.array(self.plot_data['Avg_F1']),
            #     opts={
            #         'title': mode + ' Avg_F1 over time',
            #         # 'legend': ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6'],
            #         'xlabel': 'epoch',
            #         'ylabel': 'Avg_F1'},
            #     win=mode + ' Avg_F1')
            # self.vis.line(
            #     X=np.array(self.plot_data['X']),
            #     Y=np.array(self.plot_data['FWIOU']),
            #     opts={
            #         'title': mode + ' FWIOU over time',
            #         # 'legend': ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6'],
            #         'xlabel': 'epoch',
            #         'ylabel': 'FWIOU'},
            #     win=mode + ' FWIOU')

        except VisdomExceptionBase:
            self.create_visdom_connections()

    def print_current_metrics(self, epoch, Metric):
        MIOU = Metric.Mean_Intersection_over_Union()
        OA = Metric.Pixel_Accuracy()
        IOU, FWIOU = Metric.Frequency_Weighted_Intersection_over_Union()
        F1_score, Avg_F1_score = Metric.F1_score()
        self.string = self.string + '{' + f"'iters': {epoch}, 'OA': {OA}, 'F1_score': {Avg_F1_score}, 'MIOU': {MIOU}, 'FWIOU': {FWIOU}" + '}<br>'
        self.vis.text(self.string, win='_Metrics')

    # # metrics: same format as |metrics| of plot_current_metrics
    # def print_current_metrics(self, epoch, iters, metrics, t_comp, t_data):
    #     """print current metrics on console; also save the metrics to the disk
    #
    #     Parameters:
    #         epoch (int) -- current epoch
    #         iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
    #         metrics (OrderedDict) -- training metrics stored in the format of (name, float) pairs
    #         t_comp (float) -- computational time per data point (normalized by batch_size)
    #         t_data (float) -- data loading time per data point (normalized by batch_size)
    #     """
    #     message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
    #     for k, v in metrics.items():
    #         message += '%s: %.3f ' % (k, v)
    #
    #     print(message)  # print the message
    #     with open(self.log_name, "a") as log_file:
    #         log_file.write('%s\n' % message)  # save the message
