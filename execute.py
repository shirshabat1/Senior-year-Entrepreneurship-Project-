import learning as learn
import numpy as np

import penny_files as p
import carmen_files as c
import menta_files as m
import functools
import matplotlib.pyplot as plt





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
    learn.randomForest(x_train, y_train, x_test, y_test, monkey)
    learn.KNeighbors(x_train, y_train, x_test, y_test, monkey)
    learn.qda(x_train, y_train, x_test, y_test, monkey)
    learn.lda(x_train, y_train, x_test, y_test, monkey)
    # learn.adaBoost(x_train, y_train, x_test, y_test)
    # learn.linearSVC(x_train, y_train, x_test, y_test, monkey)
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


def combine_monkey(m1, m2, m3=None):
    if m3 is None:
        for i in range(len(m1)):
            m1[0][i] = m1[0][i] + m2[0][i]
            m1[1][i] = m1[1][i] + m2[1][i]
            m1[2][i] = m1[2][i] + m2[2][i]
            m1[3][i] = m1[3][i] + m2[3][i]
    else:
        for i in range(len(m1)):
            m1[0][i] = m1[0][i] + m2[0][i] + m3[0][i]
            m1[1][i] = m1[1][i] + m2[1][i] + m3[1][i]
            m1[2][i] = m1[2][i] + m2[2][i] + m3[2][i]
            m1[3][i] = m1[3][i] + m2[3][i] + m3[3][i]
    return [m1[0], m1[1], m1[2], m1[3]]


mte1 = [m.test_small1, m.test_small_1_welch, m.test_small_1_psd_welch, m.test_small_1_psd_multitaper]
mte2 = [m.test_small2, m.test_small_2_welch, m.test_small_2_psd_welch, m.test_small_2_psd_multitaper]
mte3 = [m.test_small3, m.test_small_3_welch, m.test_small_3_psd_welch, m.test_small_3_psd_multitaper]
mte4 = [m.test_small4, m.test_small_4_welch, m.test_small_4_psd_welch, m.test_small_4_psd_multitaper]

mtr1 = [m.train_small1, m.train_small_1_welch, m.train_small_1_psd_welch, m.train_small_1_psd_multitaper]
mtr2 = [m.train_small2, m.train_small_2_welch, m.train_small_2_psd_welch, m.train_small_2_psd_multitaper]
mtr3 = [m.train_small3, m.train_small_3_welch, m.train_small_3_psd_welch, m.train_small_3_psd_multitaper]
mtr4 = [m.train_small4, m.train_small_4_welch, m.train_small_4_psd_welch, m.train_small_4_psd_multitaper]

mv1 = [m.valid_small1, m.valid_small_1_welch, m.valid_small_1_psd_welch, m.valid_small_1_psd_multitaper]
mv2 = [m.valid_small2, m.valid_small_2_welch, m.valid_small_2_psd_welch, m.valid_small_2_psd_multitaper]
mv3 = [m.valid_small3, m.valid_small_3_welch, m.valid_small_3_psd_welch, m.valid_small_3_psd_multitaper]
mv4 = [m.valid_small4, m.valid_small_4_welch, m.valid_small_4_psd_welch, m.valid_small_4_psd_multitaper]

pte1 = [p.test_small1, p.test_small_1_welch, p.test_small_1_psd_welch, p.test_small_1_psd_multitaper]
pte2 = [p.test_small2, p.test_small_2_welch, p.test_small_2_psd_welch, p.test_small_2_psd_multitaper]
pte3 = [p.test_small3, p.test_small_3_welch, p.test_small_3_psd_welch, p.test_small_3_psd_multitaper]
pte4 = [p.test_small4, p.test_small_4_welch, p.test_small_4_psd_welch, p.test_small_4_psd_multitaper]

ptr1 = [p.train_small1, p.train_small_1_welch, p.train_small_1_psd_welch, p.train_small_1_psd_multitaper]
ptr2 = [p.train_small2, p.train_small_2_welch, p.train_small_2_psd_welch, p.train_small_2_psd_multitaper]
ptr3 = [p.train_small3, p.train_small_3_welch, p.train_small_3_psd_welch, p.train_small_3_psd_multitaper]
ptr4 = [p.train_small4, p.train_small_4_welch, p.train_small_4_psd_welch, p.train_small_4_psd_multitaper]

pv1 = [p.valid_small1, p.valid_small_1_welch, p.valid_small_1_psd_welch, p.valid_small_1_psd_multitaper]
pv2 = [p.valid_small2, p.valid_small_2_welch, p.valid_small_2_psd_welch, p.valid_small_2_psd_multitaper]
pv3 = [p.valid_small3, p.valid_small_3_welch, p.valid_small_3_psd_welch, p.valid_small_3_psd_multitaper]
pv4 = [p.valid_small4, p.valid_small_4_welch, p.valid_small_4_psd_welch, p.valid_small_4_psd_multitaper]

cte1 = [c.test_small1, c.test_small_1_welch, c.test_small_1_psd_welch, c.test_small_1_psd_multitaper]
cte2 = [c.test_small2, c.test_small_2_welch, c.test_small_2_psd_welch, c.test_small_2_psd_multitaper]
cte3 = [c.test_small3, c.test_small_3_welch, c.test_small_3_psd_welch, c.test_small_3_psd_multitaper]
cte4 = [c.test_small4, c.test_small_4_welch, c.test_small_4_psd_welch, c.test_small_4_psd_multitaper]

ctr1 = [c.train_small1, c.train_small_1_welch, c.train_small_1_psd_welch, c.train_small_1_psd_multitaper]
ctr2 = [c.train_small2, c.train_small_2_welch, c.train_small_2_psd_welch, c.train_small_2_psd_multitaper]
ctr3 = [c.train_small3, c.train_small_3_welch, c.train_small_3_psd_welch, c.train_small_3_psd_multitaper]
ctr4 = [c.train_small4, c.train_small_4_welch, c.train_small_4_psd_welch, c.train_small_4_psd_multitaper]

cv1 = [c.valid_small1, c.valid_small_1_welch, c.valid_small_1_psd_welch, c.valid_small_1_psd_multitaper]
cv2 = [c.valid_small2, c.valid_small_2_welch, c.valid_small_2_psd_welch, c.valid_small_2_psd_multitaper]
cv3 = [c.valid_small3, c.valid_small_3_welch, c.valid_small_3_psd_welch, c.valid_small_3_psd_multitaper]
cv4 = [c.valid_small4, c.valid_small_4_welch, c.valid_small_4_psd_welch, c.valid_small_4_psd_multitaper]

p_m_x_1_1, p_m_x_1_2, p_m_x_1_3, p_m_x_1_4 = combine_monkey(mte1, pte1)
p_m_x_2_1, p_m_x_2_2, p_m_x_2_3, p_m_x_2_4 = combine_monkey(mte2, pte2)
p_m_x_3_1, p_m_x_3_2, p_m_x_3_3, p_m_x_3_4 = combine_monkey(mte3, pte3)
p_m_x_4_1, p_m_x_4_2, p_m_x_4_3, p_m_x_4_4 = combine_monkey(mte4, pte4)

p_m_y_1_1, p_m_y_1_2, p_m_y_1_3, p_m_y_1_4 = combine_monkey(mtr1, ptr1)
p_m_y_2_1, p_m_y_2_2, p_m_y_2_3, p_m_y_2_4 = combine_monkey(mtr2, ptr2)
p_m_y_3_1, p_m_y_3_2, p_m_y_3_3, p_m_y_3_4 = combine_monkey(mtr3, ptr3)
p_m_y_4_1, p_m_y_4_2, p_m_y_4_3, p_m_y_4_4 = combine_monkey(mtr4, ptr4)

p_m_z_1_1, p_m_z_1_2, p_m_z_1_3, p_m_z_1_4 = combine_monkey(mv1, pv1)
p_m_z_2_1, p_m_z_2_2, p_m_z_2_3, p_m_z_2_4 = combine_monkey(mv2, pv2)
p_m_z_3_1, p_m_z_3_2, p_m_z_3_3, p_m_z_3_4 = combine_monkey(mv3, pv3)
p_m_z_4_1, p_m_z_4_2, p_m_z_4_3, p_m_z_4_4 = combine_monkey(mv4, pv4)

c_m_x_1_1, c_m_x_1_2, c_m_x_1_3, c_m_x_1_4 = combine_monkey(mte1, cte1)
c_m_x_2_1, c_m_x_2_2, c_m_x_2_3, c_m_x_2_4 = combine_monkey(mte2, cte2)
c_m_x_3_1, c_m_x_3_2, c_m_x_3_3, c_m_x_3_4 = combine_monkey(mte3, cte3)
c_m_x_4_1, c_m_x_4_2, c_m_x_4_3, c_m_x_4_4 = combine_monkey(mte4, cte4)

c_m_y_1_1, c_m_y_1_2, c_m_y_1_3, c_m_y_1_4 = combine_monkey(mtr1, ctr1)
c_m_y_2_1, c_m_y_2_2, c_m_y_2_3, c_m_y_2_4 = combine_monkey(mtr2, ctr2)
c_m_y_3_1, c_m_y_3_2, c_m_y_3_3, c_m_y_3_4 = combine_monkey(mtr3, ctr3)
c_m_y_4_1, c_m_y_4_2, c_m_y_4_3, c_m_y_4_4 = combine_monkey(mtr4, ctr4)

c_m_z_1_1, c_m_z_1_2, c_m_z_1_3, c_m_z_1_4 = combine_monkey(mv1, cv1)
c_m_z_2_1, c_m_z_2_2, c_m_z_2_3, c_m_z_2_4 = combine_monkey(mv2, cv2)
c_m_z_3_1, c_m_z_3_2, c_m_z_3_3, c_m_z_3_4 = combine_monkey(mv3, cv3)
c_m_z_4_1, c_m_z_4_2, c_m_z_4_3, c_m_z_4_4 = combine_monkey(mv4, cv4)

c_p_x_1_1, c_p_x_1_2, c_p_x_1_3, c_p_x_1_4 = combine_monkey(pte1, cte1)
c_p_x_2_1, c_p_x_2_2, c_p_x_2_3, c_p_x_2_4 = combine_monkey(pte2, cte2)
c_p_x_3_1, c_p_x_3_2, c_p_x_3_3, c_p_x_3_4 = combine_monkey(pte3, cte3)
c_p_x_4_1, c_p_x_4_2, c_p_x_4_3, c_p_x_4_4 = combine_monkey(pte4, cte4)

c_p_y_1_1, c_p_y_1_2, c_p_y_1_3, c_p_y_1_4 = combine_monkey(ptr1, ctr1)
c_p_y_2_1, c_p_y_2_2, c_p_y_2_3, c_p_y_2_4 = combine_monkey(ptr2, ctr2)
c_p_y_3_1, c_p_y_3_2, c_p_y_3_3, c_p_y_3_4 = combine_monkey(ptr3, ctr3)
c_p_y_4_1, c_p_y_4_2, c_p_y_4_3, c_p_y_4_4 = combine_monkey(ptr4, ctr4)

c_p_z_1_1, c_p_z_1_2, c_p_z_1_3, c_p_z_1_4 = combine_monkey(pv1, cv1)
c_p_z_2_1, c_p_z_2_2, c_p_z_2_3, c_p_z_2_4 = combine_monkey(pv2, cv2)
c_p_z_3_1, c_p_z_3_2, c_p_z_3_3, c_p_z_3_4 = combine_monkey(pv3, cv3)
c_p_z_4_1, c_p_z_4_2, c_p_z_4_3, c_p_z_4_4 = combine_monkey(pv4, cv4)

m_c_p_x_1_1, m_c_p_x_1_2, m_c_p_x_1_3, m_c_p_x_1_4 = combine_monkey(pte1, cte1, mte1)
m_c_p_x_2_1, m_c_p_x_2_2, m_c_p_x_2_3, m_c_p_x_2_4 = combine_monkey(pte2, cte2, mte2)
m_c_p_x_3_1, m_c_p_x_3_2, m_c_p_x_3_3, m_c_p_x_3_4 = combine_monkey(pte3, cte3, mte3)
m_c_p_x_4_1, m_c_p_x_4_2, m_c_p_x_4_3, m_c_p_x_4_4 = combine_monkey(pte4, cte4, mte4)

m_c_p_y_1_1, m_c_p_y_1_2, m_c_p_y_1_3, m_c_p_y_1_4 = combine_monkey(ptr1, ctr1, mtr1)
m_c_p_y_2_1, m_c_p_y_2_2, m_c_p_y_2_3, m_c_p_y_2_4 = combine_monkey(ptr2, ctr2, mtr2)
m_c_p_y_3_1, m_c_p_y_3_2, m_c_p_y_3_3, m_c_p_y_3_4 = combine_monkey(ptr3, ctr3, mtr3)
m_c_p_y_4_1, m_c_p_y_4_2, m_c_p_y_4_3, m_c_p_y_4_4 = combine_monkey(ptr4, ctr4, mtr4)

m_c_p_z_1_1, m_c_p_z_1_2, m_c_p_z_1_3, m_c_p_z_1_4 = combine_monkey(pv1, cv1, mv1)
m_c_p_z_2_1, m_c_p_z_2_2, m_c_p_z_2_3, m_c_p_z_2_4 = combine_monkey(pv2, cv2, mv2)
m_c_p_z_3_1, m_c_p_z_3_2, m_c_p_z_3_3, m_c_p_z_3_4 = combine_monkey(pv3, cv3, mv3)
m_c_p_z_4_1, m_c_p_z_4_2, m_c_p_z_4_3, m_c_p_z_4_4 = combine_monkey(pv4, cv4, mv4)

p_mx1 = (p_m_x_1_1, p_m_x_1_2, p_m_x_1_3, p_m_x_1_4)
p_mx2 = (p_m_x_2_1, p_m_x_2_2, p_m_x_2_3, p_m_x_2_4)
p_mx3 = (p_m_x_3_1, p_m_x_3_2, p_m_x_3_3, p_m_x_3_4)
p_px4 = (p_m_x_4_1, p_m_x_4_2, p_m_x_4_3, p_m_x_4_4)

p_my1 = (p_m_y_1_1, p_m_y_1_2, p_m_y_1_3, p_m_y_1_4)
p_my2 = (p_m_y_2_1, p_m_y_2_2, p_m_y_2_3, p_m_y_2_4)
p_my3 = (p_m_y_3_1, p_m_y_3_2, p_m_y_3_3, p_m_y_3_4)
p_my4 = (p_m_y_4_1, p_m_y_4_2, p_m_y_4_3, p_m_y_4_4)

c_mx1 = (c_m_x_1_1, c_m_x_1_2, c_m_x_1_3, c_m_x_1_4)
c_mx2 = (c_m_x_2_1, c_m_x_2_2, c_m_x_2_3, c_m_x_2_4)
c_mx3 = (c_m_x_3_1, c_m_x_3_2, c_m_x_3_3, c_m_x_3_4)
c_mx4 = (c_m_x_4_1, c_m_x_4_2, c_m_x_4_3, c_m_x_4_4)

c_my1 = (c_m_y_1_1, c_m_y_1_2, c_m_y_1_3, c_m_y_1_4)
c_my2 = (c_m_y_2_1, c_m_y_2_2, c_m_y_2_3, c_m_y_2_4)
c_my3 = (c_m_y_3_1, c_m_y_3_2, c_m_y_3_3, c_m_y_3_4)
c_my4 = (c_m_y_4_1, c_m_y_4_2, c_m_y_4_3, c_m_y_4_4)

c_px1 = (c_p_x_1_1, c_p_x_1_2, c_p_x_1_3, c_p_x_1_4)
c_px2 = (c_p_x_2_1, c_p_x_2_2, c_p_x_2_3, c_p_x_2_4)
c_px3 = (c_p_x_3_1, c_p_x_3_2, c_p_x_3_3, c_p_x_3_4)
c_px4 = (c_p_x_4_1, c_p_x_4_2, c_p_x_4_3, c_p_x_4_4)

c_py1 = (c_p_y_1_1, c_p_y_1_2, c_p_y_1_3, c_p_y_1_4)
c_py2 = (c_p_y_2_1, c_p_y_2_2, c_p_y_2_3, c_p_y_2_4)
c_py3 = (c_p_y_3_1, c_p_y_3_2, c_p_y_3_3, c_p_y_3_4)
c_py4 = (c_p_y_4_1, c_p_y_4_2, c_p_y_4_3, c_p_y_4_4)

m_c_px1 = (c_p_x_1_1, c_p_x_1_2, c_p_x_1_3, c_p_x_1_4)
m_c_px2 = (c_p_x_2_1, c_p_x_2_2, c_p_x_2_3, c_p_x_2_4)
m_c_px3 = (c_p_x_3_1, c_p_x_3_2, c_p_x_3_3, c_p_x_3_4)
m_c_px4 = (c_p_x_4_1, c_p_x_4_2, c_p_x_4_3, c_p_x_4_4)

m_c_py1 = (c_p_y_1_1, c_p_y_1_2, c_p_y_1_3, c_p_y_1_4)
m_c_py2 = (c_p_y_2_1, c_p_y_2_2, c_p_y_2_3, c_p_y_2_4)
m_c_py3 = (c_p_y_3_1, c_p_y_3_2, c_p_y_3_3, c_p_y_3_4)
m_c_py4 = (c_p_y_4_1, c_p_y_4_2, c_p_y_4_3, c_p_y_4_4)

m_c_pz1 = (c_p_z_1_1, c_p_z_1_2, c_p_z_1_3, c_p_z_1_4)
m_c_pz2 = (c_p_z_2_1, c_p_z_2_2, c_p_z_2_3, c_p_z_2_4)
m_c_pz3 = (c_p_z_3_1, c_p_z_3_2, c_p_z_3_3, c_p_z_3_4)
m_c_pz4 = (c_p_z_4_1, c_p_z_4_2, c_p_z_4_3, c_p_z_4_4)


def plot_scatter_helper(data, f1, f2):
    x = []
    y = []
    for i in range(len(data)):
        x.append(data[i][f1])
        y.append(data[i][f2])
    # plt.yscale('log')
    plt.scatter(x, y, linewidths=0.5, s=1)


def plot_scatter(data, monkey, f1, f2):
    for i in data:
        print(i)
        plot_scatter_helper(i, f1, f2)
    plt.title(str(monkey) + "- feature " + str(f1)+ " against  feature "+ str(f2))
    plt.legend(labels=['area1', 'area2', 'area3', 'area4'], title='areas')
    plt.xlabel("feature " + str(f1))
    plt.ylabel("feature " + str(f2))
    plt.yscale('log')
    plt.xscale('log')
    plt.show()

