from torchviz import make_dot
from network_building import MyModel
import torch
import hiddenlayer as hl
import os
os.environ["PATH"] += os.pathsep + 'D:/Graphviz/bin/'
import sys
import torch.nn as nn
from tensorboardX import SummaryWriter


model = MyModel()
optimizer = torch.optim.Adam(model.parameters())
checkpoint = torch.load('model_weights_opt1.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
model.train()
print('Loading model complete')
x = torch.randn(11).requires_grad_(True)
y = model(x)

with SummaryWriter(comment='ANN') as w:
    w.add_graph(model, (x, ))
    # 写一个新的数值序列到logs内的文件里，比如sin正弦波。
    for i in range(100):
        x = torch.tensor(i / 10, dtype=torch.float)
        y = torch.sin(x)
        # 写入数据的标注指定为 data/sin, 写入数据是y, 当前已迭代的步数是i。
        w.add_scalar('data/sin', y, i)













# MyConvNetVis = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))
# MyConvNetVis.format = "png"
# MyConvNetVis.directory = "data"
# MyConvNetVis.view()

# transforms = [hl.transforms.Fold("MatMul > Add > Relu", "Relu")] # Removes Constant nodes from graph.
# graph = hl.build_graph(model, x, transforms=transforms)
# graph.theme = hl.graph.THEMES['blue'].copy()
# graph.save('rnn_hiddenlayer', format='png')
