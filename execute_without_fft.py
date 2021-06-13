import learning as learn
import numpy as np

import penny_files as p
import carmen_files as c
import menta_files as m
import functools
import matplotlib.pyplot as plt
from statistics import stdev


def get_x_y(lst):
    test_1, test_2, test_3, test_4 = lst
    x = learn.getX(test_1, test_2, test_3, test_4)
    y = learn.getY(test_1, test_2, test_3, test_4)
    return x, y


def plot_scatter_helper(data, f1, f2, index, perc):
    x_temp, x = [], []
    y_temp, y = [], []
    for i in range(len(data)):
        x_temp.append(data[i][f1])
        y_temp.append(data[i][f2])
    x_mean, x_stdv = np.mean(x_temp), np.std(x_temp)
    for i in range(len(data)):
        if x_mean - x_stdv * perc < x_temp[i] < x_mean + x_stdv * perc:
            x.append(x_temp[i])
            y.append(y_temp[i])

    # plt.yscale('log'
    # print("y: ", y)
    # print("x: ", x)
    # print("------------------------")

    plt.scatter(x, y, alpha=0.7, label="area" + str(index))
    # plt.show()
    # plt.scatter(x, y, linewidths=4, s=1)
    # plt.plot(x,y)


def plot_scatter(data, monkey, f1, f2):
    perc = 0.9
    for index, i in enumerate(data):
        plot_scatter_helper(i, f1, f2, index + 1, perc)
    plt.title(str(monkey) + "- feature " + str(f1) + " against  feature " + str(f2) + ". stdv perc: " + str(perc * 100))
    plt.legend(title='areas')
    # plt.legend(labels=['area1', 'area2', 'area3', 'area4'], title='areas')
    plt.xlabel("feature " + str(f1))
    plt.ylabel("feature " + str(f2))
    plt.yscale('log')
    # plt.xscale('log')
    plt.show()


px_train, py_train = get_x_y(m.menta_train_features)
px_test, py_test = get_x_y(m.menta_validation_features)
learn.KNeighbors(px_train, py_train, px_test, py_test, "menta" )

# f1, f2 = learn.mutualInfo(px_train, py_train)
# plot_scatter(p.penny_train_features, 'PENNY', f1, f2)
#
# mx_train, my_train = get_x_y(m.menta_train_features)
# f1, f2 = learn.mutualInfo(mx_train, my_train)
# plot_scatter(m.menta_train_features, 'MENTA', f1, f2)
#
# cx_train, cy_train = get_x_y(c.carmen_train_features)
# f1, f2 = learn.mutualInfo(cx_train, cy_train)
# plot_scatter(c.carmen_train_features, 'CARMEN', f1, f2)



# mx_train, my_train = get_x_y(c.carmen_train)
# f1, f2 = learn.mutualInfo(mx_train, my_train)
# train1, train2, train3, train4 = c.carmen_
# scatter = [c.train1, c.train2, c.train3, c.train4]
# monkey = 'Carmen'
# plot_scatter(scatter, monkey, f1, f2)
