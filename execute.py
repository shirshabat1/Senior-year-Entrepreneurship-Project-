import learning as learn
import numpy as np

from files import penny_files as p
from files import carmen_files as c
from files import menta_files as m



def get_x_y(x1, x2, x3, x4):
    test_1 = np.concatenate(x1, axis=1)
    test_2 = np.concatenate(x2, axis=1)
    test_3 = np.concatenate(x3, axis=1)
    test_4 = np.concatenate(x4, axis=1)
    x = learn.getX(test_1, test_2, test_3, test_4)
    y = learn.getY(test_1, test_2, test_3, test_4)
    return x, y

def get_result(x_train, y_train, x_test, y_test, monkey):
    print("-----" + str(monkey) + "--------")
    learn.randomForest(x_train, y_train, x_test, y_test)
    learn.KNeighbors(x_train, y_train, x_test, y_test)
    learn.qda(x_train, y_train, x_test, y_test)
    learn.lda(x_train, y_train, x_test, y_test)
    learn.adaBoost(x_train, y_train, x_test, y_test)
    learn.linearSVC(x_train, y_train, x_test, y_test)
    learn.mutualInfo(x_test, y_test, str(monkey) + " - test")
    learn.mutualInfo(x_train, y_train, str(monkey) + " - train")

############ CARMEN ######################

cy1 = (c.train1, c.train_1_welch, c.train_1_psd_welch, c.train_1_psd_multitaper, c.train_1_cor)
cy2 = (c.train2, c.train_2_welch, c.train_2_psd_welch, c.train_2_psd_multitaper, c.train_2_cor)
cy3 = (c.train3, c.train_3_welch, c.train_3_psd_welch, c.train_3_psd_multitaper, c.train_3_cor)
cy4 = (c.train4, c.train_4_welch, c.train_4_psd_welch, c.train_4_psd_multitaper, c.train_4_cor)

cx1 = (c.test1, c.test_1_welch, c.test_1_psd_welch, c.test_1_psd_multitaper, c.test_1_cor)
cx2 = (c.test2, c.test_2_welch, c.test_2_psd_welch, c.test_2_psd_multitaper, c.test_2_cor)
cx3 = (c.test3, c.test_3_welch, c.test_3_psd_welch, c.test_3_psd_multitaper, c.test_3_cor)
cx4 = (c.test4, c.test_4_welch, c.test_4_psd_welch, c.test_4_psd_multitaper, c.test_4_cor)

############ MENTA ######################

my1 = (m.train1, m.train_1_welch, m.train_1_psd_welch, m.train_1_psd_multitaper, m.train_1_cor)
my2 = (m.train2, m.train_2_welch, m.train_2_psd_welch, m.train_2_psd_multitaper, m.train_2_cor)
my3 = (m.train3, m.train_3_welch, m.train_3_psd_welch, m.train_3_psd_multitaper, m.train_3_cor)
my4 = (m.train4, m.train_4_welch, m.train_4_psd_welch, m.train_4_psd_multitaper, m.train_4_cor)

mx1 = (m.test1, m.test_1_welch, m.test_1_psd_welch, m.test_1_psd_multitaper,  m.test_1_cor)
mx2 = (m.test2, m.test_2_welch, m.test_2_psd_welch, m.test_2_psd_multitaper,  m.test_2_cor)
mx3 = (m.test3, m.test_3_welch, m.test_3_psd_welch, m.test_3_psd_multitaper,  m.test_3_cor)
mx4 = (m.test4, m.test_4_welch, m.test_4_psd_welch, m.test_4_psd_multitaper,  m.test_4_cor)

############ PENNY ######################

py1 = (p.train1, p.train_1_welch, p.train_1_psd_welch, p.train_1_psd_multitaper, p.train_1_cor)
py2 = (p.train2, p.train_2_welch, p.train_2_psd_welch, p.train_2_psd_multitaper, p.train_2_cor)
py3 = (p.train3, p.train_3_welch, p.train_3_psd_welch, p.train_3_psd_multitaper, p.train_3_cor)
py4 = (p.train4, p.train_4_welch, p.train_4_psd_welch, p.train_4_psd_multitaper, p.train_4_cor)

px1 = (p.test1, p.test_1_welch,p.test_1_psd_welch, p.test_1_psd_multitaper,  p.test_1_cor)
px2 = (p.test2, p.test_2_welch, p.test_2_psd_welch, p.test_2_psd_multitaper,  p.test_2_cor)
px3 = (p.test3, p.test_3_welch, p.test_3_psd_welch, p.test_3_psd_multitaper,  p.test_3_cor)
px4 = (p.test4, p.test_4_welch, p.test_4_psd_welch, p.test_4_psd_multitaper,  p.test_4_cor)


# if __name__ == '__main__':
#     cx_train, cy_train = get_x_y(cy1, cy2, cy3, cy4)
#     cx_test, cy_test = get_x_y(cx1, cx2, cx3,cx4)
#     get_result(cx_train, cy_train, cx_test, cy_test, "carmen")
#
# if __name__ == '__main__':
#     mx_train, my_train = get_x_y(my1, my2, my3, my4)
#     mx_test, my_test = get_x_y(mx1, mx2, mx3,mx4)
#     get_result(mx_train, my_train, mx_test, my_test, "menta")
#
# if __name__ == '__main__':
#     mx_train, my_train = get_x_y(py1, py2, py3, py4)
#     mx_test, my_test = get_x_y(px1, px2, px3,px4)
#     get_result(mx_train, my_train, mx_test, my_test, "penny")


