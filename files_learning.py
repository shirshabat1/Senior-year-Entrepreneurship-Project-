import learning as learn
import numpy as np

import penny_files as p
import carmen_files as c
import menta_files as m


def run_features(signals, monkey, feature, TYPE, function, small =0):
    type_m = TYPE
    if small == 1:
        type_m = 'small\\' + TYPE
    for i in range(1, 5):
        if function == 1:
            x = learn.get_additional_features(signals[i - 1], str(feature))
        elif function == 2:
            x = learn.get_features(signals[i - 1])
        elif function == 3:
            x = learn.get_specific_additional_features(signals[i - 1], str(feature))
            np.save(type_m + '\\' + str(monkey) + '\\features\\psd_' + str(feature) + '\\area_' + str(i) + '_' + str(
                monkey) + '_features_extra_' + TYPE + '_' + str(len(x)) + '_signals.npy', np.array(x))
            continue
        np.save(type_m + '\\' + str(monkey) + '\\features\\' + str(feature) + '\\area_' + str(i) + '_' + str(
            monkey) + '_features_extra_' + TYPE + '_' + str(len(x)) + '_signals.npy', np.array(x))
    return




# monkey = 'menta'
# signals_test = [m.sig_test1, m.sig_test2, m.sig_test3, m.sig_test4]
# signals_train = [m.sig_train1, m.sig_train2, m.sig_train3, m.sig_train4]
# signals_valid = [m.sig_validation1, m.sig_validation2, m.sig_validation3, m.sig_validation4]


# monkey = 'penny'
# signals_test = [p.sig_test1, p.sig_test2, p.sig_test3, p.sig_test4]
# signals_train = [p.sig_train1, p.sig_train2, p.sig_train3, p.sig_train4]
# signals_valid = [p.sig_validation1, p.sig_validation2, p.sig_validation3, p.sig_validation4]


# monkey = 'carmen'
# signals_test = [c.sig_test1, c.sig_test2, c.sig_test3, c.sig_test4]
# signals_train = [c.sig_train1, c.sig_train2, c.sig_train3, c.sig_train4]
# signals_valid = [c.sig_validation1, c.sig_validation2, c.sig_validation3, c.sig_validation4]






# ------------ base features -------------
# run_features(signals_test, monkey, 'regular', 'test', 2)
# run_features(signals_train, monkey, 'regular', 'train', 2)
# run_features(signals_valid, monkey, 'regular', 'validation', 2)

# ---------- special ------------------------
#
# run_features(signals_test, monkey, 'multitapers', 'test', 3)
# run_features(signals_train, monkey, 'multitapers', 'train', 3)
# run_features(signals_valid, monkey, 'multitapers', 'validation', 3)
#
# run_features(signals_test, monkey, 'welch', 'test', 3)
# run_features(signals_train, monkey, 'welch', 'train', 3)
# run_features(signals_valid, monkey, 'welch', 'validation', 3)
#

# run_features(signals_test, monkey, 'welch', 'test', 1)
# run_features(signals_train, monkey, 'welch', 'train', 1)
# run_features(signals_valid, monkey, 'welch', 'validation', 1)



#
# monkey = 'menta'
# signals_small_test = [m.sig_small_test1, m.sig_small_test2, m.sig_small_test3, m.sig_small_test4]
# signals_small_train = [m.sig_small_train1, m.sig_small_train2, m.sig_small_train3, m.sig_small_train4]
# signals_small_valid = [m.sig_small_validation1, m.sig_small_validation2, m.sig_small_validation3, m.sig_small_validation4]

#
# monkey = 'penny'
# signals_small_test = [p.sig_small_test1, p.sig_small_test2, p.sig_small_test3, p.sig_small_test4]
# signals_small_train = [p.sig_small_train1, p.sig_small_train2, p.sig_small_train3, p.sig_small_train4]
# signals_small_valid = [p.sig_small_validation1, p.sig_small_validation2, p.sig_small_validation3, p.sig_small_validation4]
#
# monkey = 'carmen'
# signals_small_test = [c.sig_small_test1, c.sig_small_test2, c.sig_small_test3, c.sig_small_test4]
# signals_small_train = [c.sig_small_train1, c.sig_small_train2, c.sig_small_train3, c.sig_small_train4]
# signals_small_valid = [c.sig_small_validation1, c.sig_small_validation2, c.sig_small_validation3, c.sig_small_validation4]

#
# signals_test = signals_small_test
# signals_train = signals_small_train
# signals_valid = signals_small_valid
#


#
#
# # ------------ base features -------------
# run_features(signals_test, monkey, 'regular', 'test', 2,1)
# run_features(signals_train, monkey, 'regular', 'train', 2,1)
# run_features(signals_valid, monkey, 'regular', 'validation', 2,1)
#
# # ---------- special ------------------------
#
# run_features(signals_test, monkey, 'multitapers', 'test', 3,1)
# run_features(signals_train, monkey, 'multitapers', 'train', 3,1)
# run_features(signals_valid, monkey, 'multitapers', 'validation', 3,1)
#
# run_features(signals_test, monkey, 'welch', 'test', 3,1)
# run_features(signals_train, monkey, 'welch', 'train', 3,1)
# run_features(signals_valid, monkey, 'welch', 'validation', 3,1)
#
#
# run_features(signals_test, monkey, 'welch', 'test', 1,1)
# run_features(signals_train, monkey, 'welch', 'train', 1,1)
# run_features(signals_valid, monkey, 'welch', 'validation', 1,1)
#
