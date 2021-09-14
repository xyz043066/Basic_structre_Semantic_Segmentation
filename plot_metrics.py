import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
# 改变坐标轴字体
def axis_font_change(ax):
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

# 绘制图像
def show_figure(data_source,  figure_sets):
    font_title = {'family': 'Times New Roman', 'weight': 'normal', 'size': 30}
    font_label = {'family': 'Times New Roman', 'weight': 'normal', 'size': 23}
    if figure_sets['type'] == 'xlsx':
        data = pd.read_excel(data_source)
    elif figure_sets['type'] == 'csv':
        data = pd.read_csv(data_source)
    x = pd.to_datetime(data.iloc[:, 1])
    y = data.iloc[:, 2]
    plt.figure(figsize=[11, 9])
    ax = plt.subplot()
    axis_font_change(ax)
    plt.plot(x, y, figure_sets['maker_set'], linewidth=2, markersize=2)
    plt.title(figure_sets['title'], fontdict=font_title)
    plt.xlabel(figure_sets['x_label'], fontdict=font_label)
    plt.ylabel(figure_sets['y_label'], fontdict=font_label)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
# f1 = open("visdom_text.txt", 'r')
#
# line = f1.readline()
# line = line.replace('<br>', '\n')
# print(line)
# f2 = open("visdom_text.txt", 'w')
# f2.write(line)
# f2.close()
epoch = []
OA = []
F1_score = []
MIOU = []
FWIOU = []
with open("visdom_text.txt") as f:
    lines = f.readlines()
    for line in lines:
        print(line)
        line = eval(line)
        epoch.append(line['epoch'])
        OA.append(line['OA'])
        F1_score.append(line['F1_score'])
        MIOU.append(line['MIOU'])
        FWIOU.append(line['FWIOU'])
        # print(line['epoch'])
epoch = np.array(epoch)
OA = np.array(OA)
F1_score = np.array(F1_score)
MIOU = np.array(MIOU)
FWIOU = np.array(FWIOU)
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}
font_label = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
"""
    将四个指标绘制于一张图中
"""
plt.figure()
ax = plt.subplot()
plt.plot(epoch, OA)
plt.plot(epoch, F1_score)
plt.plot(epoch, MIOU)
plt.plot(epoch, FWIOU)
# plt.ylim([0.7, 0.91])
plt.grid(alpha=0.2)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel('epoch', fontdict=font_label)
plt.ylabel('metrics', fontdict=font_label)
plt.title('The change of metrics', fontdict=font)
plt.legend(['OA', 'F1_score', 'MIOU', 'FWIOU'], prop=font_label)
axis_font_change(ax)
"""
    分开绘制四个指标
"""
font1 = font
font1['size'] = 15
plt.figure()
ax = plt.subplot(2, 2, 1)
axis_font_change(ax)
plt.plot(epoch, OA, color='b', linewidth=1, markersize=15)
# plt.ylim([0.7, 0.95])
plt.title("OA", fontdict=font1)
ax = plt.subplot(2, 2, 2)
axis_font_change(ax)
plt.plot(epoch, F1_score, color='g', linewidth=1, markersize=15)
# plt.ylim([0, 0.95])
plt.title("F1_score", fontdict=font1)
ax = plt.subplot(2, 2, 3)
axis_font_change(ax)
plt.plot(epoch, MIOU, color='r', linewidth=1, markersize=15)
# plt.ylim([0, 0.95])
plt.title("MIOU", fontdict=font1)
ax = plt.subplot(2, 2, 4)
axis_font_change(ax)
plt.plot(epoch, FWIOU, color='y', linewidth=1, markersize=15)
# plt.ylim([0, 0.95])
plt.title("FWIOU", fontdict=font1)
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()

