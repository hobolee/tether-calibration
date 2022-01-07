import numpy as np
import matplotlib.pyplot as plt
import math
import torch
from torch.utils.data import DataLoader, Dataset
import scipy.io as scio
from network_building import MyModel
from tensorboardX import SummaryWriter


max_first = [84, 75, 83, 77, 95, 90, 71, 98, 86, 75, 65, 72]
for i in range(12):
    index = str(i+1).zfill(2)
    data_path = '../Videos/%s.txt' % index
    with open(data_path) as f:
        num = f.read().split()
        num = [float(x) for x in num]
    data = np.array(num).reshape(int(len(num)/11), 11)
    # 去除第一段
    data = data[max_first[i]:, :]
    for j in range(8, -1, -2):
        data[:, j] = data[:, j] - data[:, 0]
    for j in range(9, 0, -2):
        data[:, j] = data[:, j] - data[:, 1]
    # deg_0 = np.subtract(deg_0, deg_0[0, :])
    # deg_0 = np.transpose(np.subtract(np.transpose(deg_0), np.transpose(deg_0[:, 1])))
    # plt.plot(range(1515), deg_0[:, -1])
    # plt.show()

    # R(x, angle) = [[1 0 0], [0 ca -sa], [0 sa ca]]
    angle = math.pi * (i / 6)
    x1 = data[:, 2]
    y1 = data[:, 3] * -math.sin(angle)
    z1 = data[:, 3] * math.cos(angle)
    x2 = data[:, 4]
    y2 = data[:, 5] * -math.sin(angle)
    z2 = data[:, 5] * math.cos(angle)
    x3 = data[:, 6]
    y3 = data[:, 7] * -math.sin(angle)
    z3 = data[:, 7] * math.cos(angle)
    x4 = data[:, 8]
    y4 = data[:, 9] * -math.sin(angle)
    z4 = data[:, 9] * math.cos(angle)

    tan_theta = (y4 - y3)/(x4 - x3)
    theta = np.array([math.atan(i) for i in tan_theta])
    m = (y4 - y3)**2 + (x4 - x3)**2
    sqrt_m = [math.sqrt(x) for x in m]
    tan_phi = (z4 - z3) / sqrt_m
    phi = np.array([math.atan(i) for i in tan_phi])

    # 加初始力
    fz = (data[:, 10] + 0.13) * math.cos(angle)
    fy = (data[:, 10] + 0.13) * math.sin(angle)
    if i == 0:
        # all_data = np.array([x1, y1, z1, x2, y2, z2, x3, y3, z3, theta, phi, fz, fy]).transpose(1, 0)
        all_data = np.array([theta, phi, fz, fy]).transpose(1, 0)
        # all_data = np.array([theta, phi, x3, y3, z3]).transpose(1, 0)
        all_data = torch.from_numpy(all_data)
    else:
        # all_data = np.array([x1, y1, z1, x2, y2, z2, x3, y3, z3, theta, phi, fz, fy]).transpose(1, 0)
        tmp_data = np.array([theta, phi, fz, fy]).transpose(1, 0)
        # tmp_data = np.array([theta, phi, x3, y3, z3]).transpose(1, 0)
        tmp_data = torch.from_numpy(tmp_data)
        all_data = torch.cat((all_data, tmp_data), 0)


class MyDataset(Dataset):
    def __init__(self, dataset):
        self.X = dataset[:, :2]
        self.y = dataset[:, 2:]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


all_data = MyDataset(all_data)
train_size = int(0.9 * len(all_data))
test_size = len(all_data) - train_size
train_data, test_data = torch.utils.data.random_split(all_data, [train_size, test_size])
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=True)

writer = SummaryWriter(comment='MyModel')
model = MyModel()
epochs = 5000
loss_fn = torch.nn.MSELoss()
learning_rate = 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50)
ifModel = False
if ifModel:
    model = MyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    checkpoint = torch.load('model_weights_opt1.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.train()
    print('Loading model complete')


def train_loop(train_loader, model, loss_fn, optimizer):
    size = len(train_loader.dataset)
    for batch, (X, y) in enumerate(train_loader):
        pred = model(X.float())
        loss = loss_fn(pred, y.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return loss


def test_loop(test_loader, model, loss_fn):
    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    t_loss = 0
    with torch.no_grad():
        for X, y in test_loader:
            pred = model(X.float())
            t_loss += loss_fn(pred, y.float()).item()
    t_loss /= num_batches
    test_loss.append(t_loss)
    print(f"Avg loss: {t_loss:>8f} \n")
    return t_loss


test_loss = []
train_loss = []
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    loss = train_loop(train_loader, model, loss_fn, optimizer)
    t_loss = test_loop(test_loader, model, loss_fn)
    writer.add_scalars('Loss', {'train_loss': loss, 'test_loss': t_loss}, t)
    train_loss.append(loss)
    if t % 100 == 99:
        torch.save({
            'epoch': t,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, 'model_weights_opt_mac.pth')

test_loss = torch.tensor(test_loss)
train_loss = torch.tensor(train_loss)
torch.save(test_loss, 'test_loss')
torch.save(train_loss, 'train_loss')

# plt.figure()
# plt.plot(test_loss.float())
# plt.plot(train_loss.float())
# plt.show()

x = torch.randn(2).requires_grad_(True)
writer.add_graph(model, x)
writer.close



