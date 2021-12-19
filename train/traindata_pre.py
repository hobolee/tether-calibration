import numpy as np
import matplotlib.pyplot as plt
import math

with open('../Videos/01.txt') as f:
    num = f.read().split()
    num = [float(x) for x in num]
data = np.array(num).reshape(int(len(num)/11), 11)
for i in range(8, -1, -2):
    data[:, i] = data[:, i] - data[:, 0]
for i in range(9, 0, -2):
    data[:, i] = data[:, i] - data[:, 1]
# deg_0 = np.subtract(deg_0, deg_0[0, :])
# deg_0 = np.transpose(np.subtract(np.transpose(deg_0), np.transpose(deg_0[:, 1])))
# plt.plot(range(1515), deg_0[:, -1])
# plt.show()

# R(x, angle) = [[1 0 0], [0 ca -sa], [0 sa ca]]
angle = 0
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

theta = math.atan((z4 - z3)/(x4 - x3))
phi = math.atan((y4 - y3)/(x4 - x3))

x_train = np.array(data[:, 2:8])



pass

