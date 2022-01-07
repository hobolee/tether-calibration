from network_building import MyModel
import torch
import math
import numpy as np
import scipy.io as io

learning_rate = 1e-5
model = MyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
checkpoint = torch.load('model_weights_opt_mac.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
model.train()
print('Loading model complete')

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

    tan_theta = (y4 - y3) / (x4 - x3)
    theta = np.array([math.atan(i) for i in tan_theta])
    m = (y4 - y3) ** 2 + (x4 - x3) ** 2
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

table = []
for i in range(len(all_data)):
    x = all_data[i][:-2].float()
    y = model(x)
    tmp = [np.array(x[0]), np.array(x[1]), y[0].detach().numpy(), y[1].detach().numpy()]
    # tmp_loss = math.sqrt((y[0] - all_data[i][11])**2 + (y[1] - all_data[i][12])**2)
    table.append(tmp)

# np.save('table.txt', table)
io.savemat('table_f.mat', {'x': table})
