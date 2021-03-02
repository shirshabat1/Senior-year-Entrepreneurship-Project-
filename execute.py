import learning as learn
import numpy as np

import penny_files as p
import carmen_files as c
import menta_files as m


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


cy1 = (c.train1, c.train_1_welch, c.train_1_psd_welch, c.train_1_psd_multitaper)
cy2 = (c.train2, c.train_2_welch, c.train_2_psd_welch, c.train_2_psd_multitaper)
cy3 = (c.train3, c.train_3_welch, c.train_3_psd_welch, c.train_3_psd_multitaper)
cy4 = (c.train4, c.train_4_welch, c.train_4_psd_welch, c.train_4_psd_multitaper)

cx1 = (c.test1, c.test_1_welch, c.test_1_psd_welch, c.test_1_psd_multitaper)
cx2 = (c.test2, c.test_2_welch, c.test_2_psd_welch, c.test_2_psd_multitaper)
cx3 = (c.test3, c.test_3_welch, c.test_3_psd_welch, c.test_3_psd_multitaper)
cx4 = (c.test4, c.test_4_welch, c.test_4_psd_welch, c.test_4_psd_multitaper)

cz1 = (c.valid1, c.valid_1_welch, c.valid_1_psd_welch, c.valid_1_psd_multitaper)
cz2 = (c.valid2, c.valid_2_welch, c.valid_2_psd_welch, c.valid_2_psd_multitaper)
cz3 = (c.valid3, c.valid_3_welch, c.valid_3_psd_welch, c.valid_3_psd_multitaper)
cz4 = (c.valid4, c.valid_4_welch, c.valid_4_psd_welch, c.valid_4_psd_multitaper)

############ MENTA ######################
#

my1 = (m.train1, m.train_1_welch, m.train_1_psd_welch, m.train_1_psd_multitaper)
my2 = (m.train2, m.train_2_welch, m.train_2_psd_welch, m.train_2_psd_multitaper)
my3 = (m.train3, m.train_3_welch, m.train_3_psd_welch, m.train_3_psd_multitaper)
my4 = (m.train4, m.train_4_welch, m.train_4_psd_welch, m.train_4_psd_multitaper)

mx1 = (m.test1, m.test_1_welch, m.test_1_psd_welch, m.test_1_psd_multitaper)
mx2 = (m.test2, m.test_2_welch, m.test_2_psd_welch, m.test_2_psd_multitaper)
mx3 = (m.test3, m.test_3_welch, m.test_3_psd_welch, m.test_3_psd_multitaper)
mx4 = (m.test4, m.test_4_welch, m.test_4_psd_welch, m.test_4_psd_multitaper)

mz1 = (m.valid1, m.valid_1_welch, m.valid_1_psd_welch, m.valid_1_psd_multitaper)
mz2 = (m.valid2, m.valid_2_welch, m.valid_2_psd_welch, m.valid_2_psd_multitaper)
mz3 = (m.valid3, m.valid_3_welch, m.valid_3_psd_welch, m.valid_3_psd_multitaper)
mz4 = (m.valid4, m.valid_4_welch, m.valid_4_psd_welch, m.valid_4_psd_multitaper)




############ PENNY ######################

py1 = (p.train1, p.train_1_welch, p.train_1_psd_welch, p.train_1_psd_multitaper)
py2 = (p.train2, p.train_2_welch, p.train_2_psd_welch, p.train_2_psd_multitaper)
py3 = (p.train3, p.train_3_welch, p.train_3_psd_welch, p.train_3_psd_multitaper)
py4 = (p.train4, p.train_4_welch, p.train_4_psd_welch, p.train_4_psd_multitaper)

px1 = (p.test1, p.test_1_welch, p.test_1_psd_welch, p.test_1_psd_multitaper)
px2 = (p.test2, p.test_2_welch, p.test_2_psd_welch, p.test_2_psd_multitaper)
px3 = (p.test3, p.test_3_welch, p.test_3_psd_welch, p.test_3_psd_multitaper)
px4 = (p.test4, p.test_4_welch, p.test_4_psd_welch, p.test_4_psd_multitaper)

pz1 = (p.valid1, p.valid_1_welch, p.valid_1_psd_welch, p.valid_1_psd_multitaper)
pz2 = (p.valid2, p.valid_2_welch, p.valid_2_psd_welch, p.valid_2_psd_multitaper)
pz3 = (p.valid3, p.valid_3_welch, p.valid_3_psd_welch, p.valid_3_psd_multitaper)
pz4 = (p.valid4, p.valid_4_welch, p.valid_4_psd_welch, p.valid_4_psd_multitaper)



def add_feat(monkey1, monkey2):
    monkey_len = min(len(monkey1), len(monkey2))
    monkey = monkey2
    if len(monkey1) == monkey_len:
        monkey = monkey1
    for i in range(len(monkey)):
        monkey[i] = monkey1[i] + monkey2[i]
    return monkey

# p_m_x_1_1, p_m_x_1_2, p_m_x_1_3, p_m_x_1_4, p_m_x_1_5 = add_feat(m.test1, p.test1), add_feat(m.test_1_welch, p.test_1_welch), add_feat(m.test_1_psd_welch, p.test_1_psd_welch),\
#                              add_feat(m.test_1_psd_multitaper, p.test_1_psd_multitaper), add_feat(m.test_1_cor, p.test_1_cor)
# p_m_x_2_1, p_m_x_2_2, p_m_x_2_3, p_m_x_2_4, p_m_x_2_5 = add_feat(m.test2, p.test2), add_feat(m.test_2_welch, p.test_2_welch), add_feat(m.test_2_psd_welch, p.test_2_psd_welch),\
#                              add_feat(m.test_2_psd_multitaper, p.test_2_psd_multitaper), add_feat(m.test_2_cor, p.test_2_cor)
#
# p_m_x_3_1, p_m_x_3_2, p_m_x_3_3, p_m_x_3_4, p_m_x_3_5 = add_feat(m.test3, p.test3), add_feat(m.test_3_welch, p.test_3_welch), add_feat(m.test_3_psd_welch, p.test_3_psd_welch),\
#                              add_feat(m.test_3_psd_multitaper, p.test_3_psd_multitaper), add_feat(m.test_3_cor, p.test_3_cor)
#
# p_m_x_4_1, p_m_x_4_2, p_m_x_4_3, p_m_x_4_4, p_m_x_4_5 = add_feat(m.test4, p.test4), add_feat(m.test_4_welch, p.test_4_welch), add_feat(m.test_4_psd_welch, p.test_4_psd_welch),\
#                              add_feat(m.test_4_psd_multitaper, p.test_4_psd_multitaper), add_feat(m.test_4_cor, p.test_4_cor)
#
#
# p_m_y_1_1, p_m_y_1_2, p_m_y_1_3, p_m_y_1_4, p_m_y_1_5 = add_feat(m.train1, p.train1), add_feat(m.train_1_welch, p.train_1_welch), add_feat(m.train_1_psd_welch, p.train_1_psd_welch),\
#                              add_feat(m.train_1_psd_multitaper, p.train_1_psd_multitaper), add_feat(m.train_1_cor, p.train_1_cor)
# p_m_y_2_1, p_m_y_2_2, p_m_y_2_3, p_m_y_2_4, p_m_y_2_5 = add_feat(m.train2, p.train2), add_feat(m.train_2_welch, p.train_2_welch), add_feat(m.train_2_psd_welch, p.train_2_psd_welch),\
#                              add_feat(m.train_2_psd_multitaper, p.train_2_psd_multitaper), add_feat(m.train_2_cor,p.train_2_cor)
#
# p_m_y_3_1, p_m_y_3_2, p_m_y_3_3, p_m_y_3_4, p_m_y_3_5 = add_feat(m.train3, p.train3), add_feat(m.train_3_welch, p.train_3_welch), add_feat(m.train_3_psd_welch, p.train_3_psd_welch),\
#                              add_feat(m.train_3_psd_multitaper, p.train_3_psd_multitaper), add_feat(m.train_3_cor, p.train_3_cor)
#
# p_m_y_4_1, p_m_y_4_2, p_m_y_4_3, p_m_y_4_4, p_m_y_4_5 = add_feat(m.train4, p.train4), add_feat(m.train_4_welch, p.train_4_welch), add_feat(m.train_4_psd_welch, p.train_4_psd_welch),\
#                              add_feat(m.train_4_psd_multitaper, p.train_4_psd_multitaper), add_feat(m.train_4_cor, p.train_4_cor)
#
#
#
# c_p_x_1_1, c_p_x_1_2, c_p_x_1_3, c_p_x_1_4, c_p_x_1_5 = add_feat(c.test1, p.test1), add_feat(c.test_1_welch, p.test_1_welch), add_feat(c.test_1_psd_welch, p.test_1_psd_welch),\
#                              add_feat(c.test_1_psd_multitaper, p.test_1_psd_multitaper), add_feat(c.test_1_cor, p.test_1_cor)
# c_p_x_2_1, c_p_x_2_2, c_p_x_2_3, c_p_x_2_4, c_p_x_2_5 = add_feat(c.test2, p.test2), add_feat(c.test_2_welch, p.test_2_welch), add_feat(c.test_2_psd_welch, p.test_2_psd_welch),\
#                              add_feat(c.test_2_psd_multitaper, p.test_2_psd_multitaper), add_feat(c.test_2_cor, p.test_2_cor)
#
# c_p_x_3_1, c_p_x_3_2, c_p_x_3_3, c_p_x_3_4, c_p_x_3_5 = add_feat(c.test3, p.test3), add_feat(c.test_3_welch, p.test_3_welch), add_feat(c.test_3_psd_welch, p.test_3_psd_welch),\
#                              add_feat(c.test_3_psd_multitaper, p.test_3_psd_multitaper), add_feat(c.test_3_cor, p.test_3_cor)
#
# c_p_x_4_1, c_p_x_4_2, c_p_x_4_3, c_p_x_4_4, c_p_x_4_5 = add_feat(c.test4, p.test4), add_feat(c.test_4_welch, p.test_4_welch), add_feat(c.test_4_psd_welch, p.test_4_psd_welch),\
#                              add_feat(c.test_4_psd_multitaper, p.test_4_psd_multitaper), add_feat(c.test_4_cor, p.test_4_cor)
#
#
# c_p_y_1_1, c_p_y_1_2, c_p_y_1_3, c_p_y_1_4, c_p_y_1_5 = add_feat(c.train1, p.train1), add_feat(c.train_1_welch, p.train_1_welch), add_feat(c.train_1_psd_welch, p.train_1_psd_welch),\
#                              add_feat(c.train_1_psd_multitaper, p.train_1_psd_multitaper), add_feat(c.train_1_cor, p.train_1_cor)
# c_p_y_2_1, c_p_y_2_2, c_p_y_2_3, c_p_y_2_4, c_p_y_2_5 = add_feat(c.train2, p.train2), add_feat(c.train_2_welch, p.train_2_welch), add_feat(c.train_2_psd_welch, p.train_2_psd_welch),\
#                              add_feat(c.train_2_psd_multitaper, p.train_2_psd_multitaper), add_feat(c.train_2_cor,p.train_2_cor)
#
# c_p_y_3_1, c_p_y_3_2, c_p_y_3_3, c_p_y_3_4, c_p_y_3_5 = add_feat(c.train3, p.train3), add_feat(c.train_3_welch, p.train_3_welch), add_feat(c.train_3_psd_welch, p.train_3_psd_welch),\
#                              add_feat(c.train_3_psd_multitaper, p.train_3_psd_multitaper), add_feat(c.train_3_cor, p.train_3_cor)
#
# c_p_y_4_1, c_p_y_4_2, c_p_y_4_3, c_p_y_4_4, c_p_y_4_5 = add_feat(c.train4, p.train4), add_feat(c.train_4_welch, p.train_4_welch), add_feat(c.train_4_psd_welch, p.train_4_psd_welch),\
#                              add_feat(c.train_4_psd_multitaper, p.train_4_psd_multitaper), add_feat(c.train_4_cor, p.train_4_cor)
#
#
#
# c_m_x_1_1, c_m_x_1_2, c_m_x_1_3, c_m_x_1_4, c_m_x_1_5 = add_feat(c.test1, m.test1), add_feat(c.test_1_welch, m.test_1_welch), add_feat(c.test_1_psd_welch, m.test_1_psd_welch),\
#                              add_feat(c.test_1_psd_multitaper, m.test_1_psd_multitaper), add_feat(c.test_1_cor, m.test_1_cor)
# c_m_x_2_1, c_m_x_2_2, c_m_x_2_3, c_m_x_2_4, c_m_x_2_5 = add_feat(c.test2, m.test2), add_feat(c.test_2_welch, m.test_2_welch), add_feat(c.test_2_psd_welch, m.test_2_psd_welch),\
#                              add_feat(c.test_2_psd_multitaper, m.test_2_psd_multitaper), add_feat(c.test_2_cor, m.test_2_cor)
#
# c_m_x_3_1, c_m_x_3_2, c_m_x_3_3, c_m_x_3_4, c_m_x_3_5 = add_feat(c.test3, m.test3), add_feat(c.test_3_welch, m.test_3_welch), add_feat(c.test_3_psd_welch, m.test_3_psd_welch),\
#                              add_feat(c.test_3_psd_multitaper, m.test_3_psd_multitaper), add_feat(c.test_3_cor, m.test_3_cor)
#
# c_m_x_4_1, c_m_x_4_2, c_m_x_4_3, c_m_x_4_4, c_m_x_4_5 = add_feat(c.test4, m.test4), add_feat(c.test_4_welch, m.test_4_welch), add_feat(c.test_4_psd_welch, m.test_4_psd_welch),\
#                              add_feat(c.test_4_psd_multitaper, m.test_4_psd_multitaper), add_feat(c.test_4_cor, m.test_4_cor)
#
#
# c_m_y_1_1, c_m_y_1_2, c_m_y_1_3, c_m_y_1_4, c_m_y_1_5 = add_feat(c.train1, m.train1), add_feat(c.train_1_welch, m.train_1_welch), add_feat(c.train_1_psd_welch, m.train_1_psd_welch),\
#                              add_feat(c.train_1_psd_multitaper, m.train_1_psd_multitaper), add_feat(c.train_1_cor, m.train_1_cor)
# c_m_y_2_1, c_m_y_2_2, c_m_y_2_3, c_m_y_2_4, c_m_y_2_5 = add_feat(c.train2, m.train2), add_feat(c.train_2_welch, m.train_2_welch), add_feat(c.train_2_psd_welch, m.train_2_psd_welch),\
#                              add_feat(c.train_2_psd_multitaper, m.train_2_psd_multitaper), add_feat(c.train_2_cor, m.train_2_cor)
#
# c_m_y_3_1, c_m_y_3_2, c_m_y_3_3, c_m_y_3_4, c_m_y_3_5 = add_feat(c.train3, m.train3), add_feat(c.train_3_welch, m.train_3_welch), add_feat(c.train_3_psd_welch, m.train_3_psd_welch),\
#                              add_feat(c.train_3_psd_multitaper, m.train_3_psd_multitaper), add_feat(c.train_3_cor, m.train_3_cor)
#
# c_m_y_4_1, c_m_y_4_2, c_m_y_4_3, c_m_y_4_4, c_m_y_4_5 = add_feat(c.train4, m.train4), add_feat(c.train_4_welch, m.train_4_welch), add_feat(c.train_4_psd_welch, m.train_4_psd_welch),\
#                              add_feat(c.train_4_psd_multitaper, m.train_4_psd_multitaper), add_feat(c.train_4_cor, m.train_4_cor)


#
# p_mx1 = (p_m_x_1_1, p_m_x_1_2, p_m_x_1_3, p_m_x_1_4, p_m_x_1_5)
# p_mx2 = (p_m_x_2_1, p_m_x_2_2, p_m_x_2_3, p_m_x_2_4, p_m_x_2_5)
# p_mx3 = (p_m_x_3_1, p_m_x_3_2, p_m_x_3_3, p_m_x_3_4, p_m_x_3_5)
# p_px4 = (p_m_x_4_1, p_m_x_4_2, p_m_x_4_3, p_m_x_4_4,  p_m_x_4_5)
#
# p_my1 = (p_m_y_1_1,p_m_y_1_2, p_m_y_1_3, p_m_y_1_4, p_m_y_1_5)
# p_my2 = (p_m_y_2_1, p_m_y_2_2, p_m_y_2_3, p_m_y_2_4, p_m_y_2_5)
# p_my3 = (p_m_y_3_1, p_m_y_3_2,p_m_y_3_3, p_m_y_3_4, p_m_y_3_5)
# p_my4 = (p_m_y_4_1, p_m_y_4_2, p_m_y_4_3, p_m_y_4_4,  p_m_y_4_5)
#
#
# c_mx1 = (c_m_x_1_1, c_m_x_1_2, c_m_x_1_3, c_m_x_1_4, c_m_x_1_5)
# c_mx2 = (c_m_x_2_1, c_m_x_2_2, c_m_x_2_3, c_m_x_2_4, c_m_x_2_5)
# c_mx3 = (c_m_x_3_1, c_m_x_3_2, c_m_x_3_3, c_m_x_3_4, c_m_x_3_5)
# c_mx4 = (c_m_x_4_1, c_m_x_4_2, c_m_x_4_3, c_m_x_4_4,  c_m_x_4_5)
#
# c_my1 = (c_m_y_1_1,c_m_y_1_2, c_m_y_1_3, c_m_y_1_4, c_m_y_1_5)
# c_my2 = (c_m_y_2_1, c_m_y_2_2, c_m_y_2_3, c_m_y_2_4, c_m_y_2_5)
# c_my3 = (c_m_y_3_1, c_m_y_3_2,c_m_y_3_3, c_m_y_3_4, c_m_y_3_5)
# c_my4 = (c_m_y_4_1, c_m_y_4_2, c_m_y_4_3, c_m_y_4_4,  c_m_y_4_5)
#
#
# c_px1 = (c_p_x_1_1, c_p_x_1_2, c_p_x_1_3, c_p_x_1_4, c_p_x_1_5)
# c_px2 = (c_p_x_2_1, c_p_x_2_2, c_p_x_2_3, c_p_x_2_4, c_p_x_2_5)
# c_px3 = (c_p_x_3_1, c_p_x_3_2, c_p_x_3_3, c_p_x_3_4, c_p_x_3_5)
# c_px4 = (c_p_x_4_1, c_p_x_4_2, c_p_x_4_3, c_p_x_4_4,  c_p_x_4_5)
#
# c_py1 = (c_p_y_1_1,c_p_y_1_2, c_p_y_1_3, c_p_y_1_4, c_p_y_1_5)
# c_py2 = (c_p_y_2_1, c_p_y_2_2, c_p_y_2_3, c_p_y_2_4, c_p_y_2_5)
# c_py3 = (c_p_y_3_1, c_p_y_3_2,c_p_y_3_3, c_p_y_3_4, c_p_y_3_5)
# c_py4 = (c_p_y_4_1, c_p_y_4_2, c_p_y_4_3, c_p_y_4_4,  c_p_y_4_5)
#



# c_m_py1 =
# c_m_py2 =
# c_m_py3 =
# c_m_py4 =
# if __name__ == '__main__':
# #     # carmen + menta
#     c_m_x_train, c_m_y_train = get_x_y(c_my1, c_my2, c_my3, c_my4)
#     # c_m_x_test, c_m_y_test = get_x_y(c_mx1, c_mx2, c_mx3, c_mx4)
#     px_test, py_test = get_x_y(px1, px2, px3,px4)
#     get_result(c_m_x_train, c_m_y_train, px_test, py_test, "carmen + menta")

# if __name__ == '__main__':
# #     # penny+ menta
#     p_m_x_train, c_m_y_train = get_x_y(p_my1, p_my2, p_my3, p_my4)
#     # c_m_x_test, c_m_y_test = get_x_y(c_mx1, c_mx2, c_mx3, c_mx4)
#     cx_test, cy_test = get_x_y(cx1, cx2, cx3,cx4)
# #     get_result(p_m_x_train, c_m_y_train, cx_test, cy_test, "penny + menta")
#
# if __name__ == '__main__':
# #     menta to carmen
#     cx_train, cy_train = get_x_y(my1, my2, my3, my4)
#     cx_test, cy_test = get_x_y(cx1, cx2, cx3, cx4)
#     get_result(cx_train, cy_train, cx_test, cy_test, "menta to carmen")





# if __name__ == '__main__':
#     # carmen
#     cx_train, cy_train = get_x_y(cy1, cy2, cy3, cy4)
#     cx_test, cy_test = get_x_y(cx1, cx2, cx3, cx4)
#     cx_valid, cy_valid = get_x_y(cz1, cz2, cz3, cz4)
#     get_result(cx_train, cy_train, cx_valid, cy_valid, "carmen - validation")
    # get_result(cx_train, cy_train, cx_test, cy_test, "carmen")
# # # #
if __name__ == '__main__':
# # # menta
    mx_train, my_train = get_x_y(my1, my2, my3, my4)
    mx_test, my_test = get_x_y(mx1, mx2, mx3,mx4)
    mx_valid, my_valid = get_x_y(mz1, mz2, mz3, mz4)
    get_result(mx_train, my_train, mx_valid, my_valid, "menta - validation")
    # get_result(mx_train, my_train, mx_test, my_test, "menta")
#
# if __name__ == '__main__':
# # penny
#     mx_train, my_train = get_x_y(py1, py2, py3, py4)
#     mx_test, my_test = get_x_y(px1, px2, px3,px4)
#     mx_valid, my_valid = get_x_y(pz1, pz2, pz3,pz4)
#     # get_result(mx_train, my_train, mx_valid, my_valid, "penny - validation")
#     get_result(mx_train, my_train, mx_test, my_test, "penny")
#
