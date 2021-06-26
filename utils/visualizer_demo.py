import visdom
import numpy as np
import time
import torch
viz = visdom.Visdom(env='test1')
# 单张/多张图像显示与更新demo
image = viz.image(np.random.rand(3, 256, 256), win='img')
for i in range(10):
    viz.images(np.random.randn(16, 3, 64, 64), nrow=4, win='imgs', opts={'title': 'imgs'})
    time.sleep(0.1)

# 绘制曲线
x = np.arange(1, 100, 0.1)
y = np.cos(x)
viz.line(X=x, Y=y, win='sinx', opts={'title': 'y=cos(x)'})

# 绘制多条曲线
N = np.linspace(-10, 10, 100)
M = np.random.rand(100, 3)
N_2 = np.stack([N]*3, 1)
viz.line(
    # X=np.column_stack((N, N)),
    # Y=np.column_stack((N*N, N*N+20)),
    X=np.stack([N]*3, 1),
    Y=M,
    win='lines',
    opts=dict(legend=['curve1', 'curve2', 'curve3'],
              title='line demo',
              xlabel='time',
              ylabel='Volume')
)