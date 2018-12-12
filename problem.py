import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

LIM = 500
FUNC_EVAL_LIM = 10000


def f(x):
    '''
    :param x: Numpy array of n-dimension
    :return: The Schwefel function in n dimensions for x
    '''
    return -np.dot(x, np.sin(np.sqrt(np.abs(x))))


x1 = np.linspace(-LIM, LIM, 1001)
x2 = np.linspace(-LIM, LIM, 1001)
xx1, xx2 = np.meshgrid(x1, x2)

f_eval = np.zeros([len(xx1), len(xx2)])

for i in range(0, len(xx1)):
    for j in range(0, len(xx2)):
        f_eval[i, j] = f([xx1[i, j], xx2[i, j]])


def plot_points(points):
    plt.contourf(xx1, xx2, f_eval, zdir='z', offset=np.min(f_eval), cmap=cm.coolwarm)
    plt.colorbar()

    plt.scatter(*points, c='m')

    plt.show()


def plot_walk(points):
    plt.contourf(xx1, xx2, f_eval, zdir='z', offset=np.min(f_eval), cmap=cm.coolwarm)
    plt.colorbar()

    plt.plot(*points, c='m')

    plt.show()


if __name__ == '__main__':

    def plot_f_2d():
        x1 = np.linspace(-LIM, LIM, 1001)
        x2 = np.linspace(-LIM, LIM, 1001)
        xx1, xx2 = np.meshgrid(x1, x2)

        f_eval = np.zeros([len(xx1), len(xx2)])

        for i in range(0, len(xx1)):
            for j in range(0, len(xx2)):
                f_eval[i, j] = f([xx1[i, j], xx2[i, j]])

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        ax.plot_surface(xx1, xx2, f_eval, rstride=5, cstride=5, alpha=1, shade=True)
        # ax.plot_wireframe(xx1, xx2, f_eval, rstride=50, cstride=50, colors=(0,0,0), linewidths=1)

        cset = ax.contourf(xx1, xx2, f_eval, zdir='z', offset=np.min(f_eval), cmap=cm.coolwarm)

        plt.show()

        fig = plt.figure()

        plt.contourf(xx1, xx2, f_eval, zdir='z', offset=np.min(f_eval), cmap=cm.coolwarm)

        plt.show()

    plot_f_2d()
