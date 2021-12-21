import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.animation as animation
import matplotlib
from scipy.interpolate import make_interp_spline
from scipy.interpolate import splprep, splev
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

max_first = [84, 75, 83, 77, 95, 90, 71, 98, 86, 75, 65, 72]
fig = plt.figure()
for index in range(12):
    # open file
    index = str(index + 1).zfill(2)
    with open('../Videos/%s.txt' % index) as f:
        num = f.read().split()
        num = [float(x) for x in num]
    data = np.array(num).reshape(int(len(num)/11), 11)
    # 减去固定点
    for i in range(8, -1, -2):
        data[:, i] = data[:, i] - data[:, 0]
    for i in range(9, 0, -2):
        data[:, i] = data[:, i] - data[:, 1]

    # sort all data
    # data = data[data[:, 9].argsort()]

    # generate 3-D coordinate
    angle = math.pi * ((int(index) - 1) / 6)
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
    fz = (data[:, 10] + 0.13) * math.cos(angle)
    fy = -(data[:, 10] + 0.13) * math.sin(angle)
    data_name = 'data_%s' % index
    tmp = np.concatenate((x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4, fz, fy)).reshape(14, len(x1))
    tmp = np.transpose(tmp)
    exec (data_name+ '= tmp')

    # create plt object
    length = max_first[int(index) - 1] #len(data)
    for i in range(length):
        ax1 = plt.axes(projection='3d')

        # 画5个点
        ax1.scatter3D(0, 0, 0)
        ax1.scatter3D(x1[i], y1[i], z1[i], c='#000000')
        ax1.scatter3D(x2[i], y2[i], z2[i], c='#0000FF')
        ax1.scatter3D(x3[i], y3[i], z3[i], c='#008000')
        ax1.scatter3D(x4[i], y4[i], z4[i], c='#FF0000')
        # 画四个点的轨迹
        ax1.plot3D(x1[:i + 1], y1[:i + 1], z1[:i + 1], 'k')
        ax1.plot3D(x2[:i + 1], y2[:i + 1], z2[:i + 1], 'b')
        ax1.plot3D(x3[:i + 1], y3[:i + 1], z3[:i + 1], 'g')
        ax1.plot3D(x4[:i + 1], y4[:i + 1], z4[:i + 1], 'r')
        # 画5个点的样条曲线
        x_5 = [0, x1[i], x2[i], x3[i], x4[i]]
        y_5 = [0, y1[i], y2[i], y3[i], y4[i]]
        z_5 = [0, z1[i], z2[i], z3[i], z4[i]]
        tck, u = splprep([x_5, y_5, z_5], s=5)
        new_points = splev(u, tck)
        ax1.plot3D(new_points[0], new_points[1], new_points[2], '#1f77b4')
        # 画F
        # F偏移
        x_f = 0
        y_f = fy[i] * 50
        z_f = fz[i] * 50
        ax1.quiver(x4[i], y4[i], z4[i], x_f, y_f, z_f, length=1)

        #画之前系列的最后一个点
        tmp = int(index) - 1
        while tmp > 0:
            tmp_str = str(tmp).zfill(2)
            exec('data_tmp = data_%s' % tmp_str)
            data_tmp = data_tmp
            # 画5个点
            ax1.scatter3D(0, 0, 0)
            a = max_first[tmp - 1]
            ax1.scatter3D(data_tmp[max_first[tmp-1], 0], data_tmp[max_first[tmp-1], 1], data_tmp[max_first[tmp-1], 2], c='#000000')
            ax1.scatter3D(data_tmp[max_first[tmp-1], 3], data_tmp[max_first[tmp-1], 4], data_tmp[max_first[tmp-1], 5], c='#0000FF')
            ax1.scatter3D(data_tmp[max_first[tmp-1], 6], data_tmp[max_first[tmp-1], 7], data_tmp[max_first[tmp-1], 8], c='#008000')
            ax1.scatter3D(data_tmp[max_first[tmp-1], 9], data_tmp[max_first[tmp-1], 10], data_tmp[max_first[tmp-1], 11], c='#FF0000')
            # 画四个点的轨迹
            # ax1.plot3D(data_tmp[:, 0], data_tmp[:, 1], data_tmp[:, 2], 'k')
            # ax1.plot3D(data_tmp[:, 3], data_tmp[:, 4], data_tmp[:, 5], 'b')
            # ax1.plot3D(data_tmp[:, 6], data_tmp[:, 7], data_tmp[:, 8], 'g')
            # ax1.plot3D(data_tmp[:, 9], data_tmp[:, 10], data_tmp[:, 11], 'r')
            # 画5个点的样条曲线
            x_5 = [0, data_tmp[max_first[tmp-1], 0], data_tmp[max_first[tmp-1], 3], data_tmp[max_first[tmp-1], 6], data_tmp[max_first[tmp-1], 9]]
            y_5 = [0, data_tmp[max_first[tmp-1], 1], data_tmp[max_first[tmp-1], 4], data_tmp[max_first[tmp-1], 7], data_tmp[max_first[tmp-1], 10]]
            z_5 = [0, data_tmp[max_first[tmp-1], 2], data_tmp[max_first[tmp-1], 5], data_tmp[max_first[tmp-1], 8], data_tmp[max_first[tmp-1], 11]]
            tck, u = splprep([x_5, y_5, z_5], s=5)
            new_points = splev(u, tck)
            ax1.plot3D(new_points[0], new_points[1], new_points[2], '#1f77b4')

            # 画F
            # F偏移
            x_f = 0
            y_f = data_tmp[max_first[tmp-1], 13] * 50
            z_f = data_tmp[max_first[tmp-1], 12] * 50
            ax1.quiver(data_tmp[max_first[tmp-1], 9], data_tmp[max_first[tmp-1], 10], data_tmp[max_first[tmp-1], 11], x_f, y_f, z_f, length=1)
            tmp -= 1

        ax1.set_xlabel('X')
        ax1.set_xlim(0, 100)
        ax1.set_ylabel('Y')
        ax1.set_ylim(-80, 80)
        ax1.set_zlabel('Z')
        ax1.set_zlim(-80, 80)
        ax1.view_init(elev=25, azim=315)
        img_path = "../Videos/img%s_3d/%i.jpg" % (index, i)
        plt.savefig(img_path)
        # plt.show()
        plt.clf()
        # plt.close(fig)

fps = 5
size = (640, 480)
output_path = "../Videos/output2_3d.avi"
video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
for index in range(12):
    index = str(index + 1).zfill(2)
    # output_path = "../Videos/output_%s_3d.avi" % index
    # video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
    length = max_first[int(index) - 1]
    for i in range(length):
        image_path = "../Videos/img%s_3d/%i.jpg" % (index, i)
        print(image_path)
        img = cv2.imread(image_path)
        video.write(img)

video.release()
cv2.destroyAllWindows()
