import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.animation as animation
import matplotlib
from scipy.interpolate import make_interp_spline

index = 3
with open('../Videos/0%i.txt' % index) as f:
    num = f.read().split()
    num = [float(x) for x in num]
deg_0 = np.array(num).reshape(int(len(num)/11), 11)
for i in range(8, -1, -2):
    deg_0[:, i] = deg_0[:, i] - deg_0[:, 0]
for i in range(9, 0, -2):
    deg_0[:, i] = deg_0[:, i] - deg_0[:, 1]

# deg_0 = deg_0[deg_0[:, 9].argsort()]
#84, 75
length = 90
plt.figure()
plt.tight_layout()
plt.ion()
for i in range(length):
    plt.plot(deg_0[i, 0], -deg_0[i, 1], 'ok')
    plt.plot(deg_0[i, 2], -deg_0[i, 3], 'ob')
    plt.plot(deg_0[i, 4], -deg_0[i, 5], 'og')
    plt.plot(deg_0[i, 6], -deg_0[i, 7], 'oy')
    plt.plot(deg_0[i, 8], -deg_0[i, 9], 'or')
    plt.plot(deg_0[:i+1, 0], -deg_0[:i+1, 1], 'k')
    plt.plot(deg_0[:i+1, 2], -deg_0[:i+1, 3], 'b')
    plt.plot(deg_0[:i+1, 4], -deg_0[:i+1, 5], 'g')
    plt.plot(deg_0[:i+1, 6], -deg_0[:i+1, 7], 'y')
    plt.plot(deg_0[:i+1, 8], -deg_0[:i+1, 9], 'r')
    plt.arrow(deg_0[i, 8], -deg_0[i, 9], 0, -deg_0[i, 10]*30, head_width=1, head_length=deg_0[i, 10]*6)
    plt.text(deg_0[i, 8]+1, -deg_0[i, 9]-deg_0[i, 10]*30, "F")
    x = deg_0[i, 0:9:2]
    y = -deg_0[i, 1:10:2]
    model = make_interp_spline(x, y)
    xs = np.linspace(0, deg_0[i, 8])
    ys = model(xs)
    plt.plot(xs, ys)
    plt.ylim(-60, 10)
    plt.show()
    img_path = "../Videos/img%i/%i.jpg" % (index, i)
    plt.savefig(img_path)
    plt.clf()

fps = 5
size = (640, 480)
output_path = "../Videos/output_%i.avi" % index
video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
for i in range(length):
    image_path = "../Videos/img%i/%i.jpg" % (index, i)
    print(image_path)
    img = cv2.imread(image_path)
    video.write(img)

video.release()
cv2.destroyAllWindows()




# def draw_line(i, deg_0):
#     plt.plot(deg_0[i, 0], deg_0[i, 1], 'ok')
#     plt.plot(deg_0[i, 2], deg_0[i, 3], 'ob')
#     plt.plot(deg_0[i, 4], deg_0[i, 5], 'og')
#     plt.plot(deg_0[i, 6], deg_0[i, 7], 'oy')
#     plt.plot(deg_0[i, 8], deg_0[i, 9], 'or')
#     plt.plot(deg_0[:i+1, 0], deg_0[:i+1, 1], 'k')
#     plt.plot(deg_0[:i+1, 2], deg_0[:i+1, 3], 'b')
#     plt.plot(deg_0[:i+1, 4], deg_0[:i+1, 5], 'g')
#     plt.plot(deg_0[:i+1, 6], deg_0[:i+1, 7], 'y')
#     plt.plot(deg_0[:i+1, 8], deg_0[:i+1, 9], 'r')
#
# fig = plt.figure()
# # plt.tight_layout()
# # plt.ion()
#
# line_ani = animation.FuncAnimation(
#     fig, draw_line, 100,fargs=(deg_0, 1), interval=50, blit=False)
# matplotlib.rc('animation', html='html5')
# line_ani
