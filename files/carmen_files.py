import numpy as np

START_TRAIN = 'train\\carmen\\features\\'
START_TEST = 'test\\carmen\\features\\'

AREA_1_TRAIN = '\\area_1_carmen_features_extra_train5017_signals.npy'
AREA_2_TRAIN = '\\area_2_carmen_features_extra_train5017_signals.npy'
AREA_3_TRAIN = '\\area_3_carmen_features_extra_train5017_signals.npy'
AREA_4_TRAIN = '\\area_4_carmen_features_extra_train5017_signals.npy'

AREA_1_TEST = '\\area_1_carmen_features_extra_test_1075_signals.npy'
AREA_2_TEST = '\\area_2_carmen_features_extra_test_1075_signals.npy'
AREA_3_TEST = '\\area_3_carmen_features_extra_test_1075_signals.npy'
AREA_4_TEST = '\\area_4_carmen_features_extra_test_1075_signals.npy'


def load_test_feature(feature):
    test_1 = np.load(START_TEST+ str(feature) + AREA_1_TEST, allow_pickle=True)
    test_2 = np.load(START_TEST + str(feature) + AREA_2_TEST, allow_pickle=True)
    test_3 = np.load(START_TEST + str(feature) + AREA_3_TEST, allow_pickle=True)
    test_4 = np.load(START_TEST + str(feature) + AREA_4_TEST, allow_pickle=True)
    return test_1, test_2, test_3, test_4


def load_train_feature(feature):
    train_1 = np.load(START_TRAIN + str(feature) + AREA_1_TRAIN, allow_pickle=True)
    train_2 = np.load(START_TRAIN + str(feature) + AREA_2_TRAIN, allow_pickle=True)
    train_3 = np.load(START_TRAIN + str(feature) + AREA_3_TRAIN, allow_pickle=True)
    train_4 = np.load(START_TRAIN + str(feature) + AREA_4_TRAIN, allow_pickle=True)
    return train_1, train_2, train_3, train_4


train1, train2, train3, train4 = load_train_feature('regular')
test1, test2, test3, test4 = load_test_feature('regular')

train_1_psd_burg, train_2_psd_burg, train_3_psd_burg, train_4_psd_burg = load_train_feature('psd_burg')
test_1_psd_burg, test_2_psd_burg, test_3_psd_burg, test_4_psd_burg = load_test_feature('psd_burg')

train_1_psd_welch, train_2_psd_welch, train_3_psd_welch, train_4_psd_welch = load_train_feature('psd_welch')
test_1_psd_welch, test_2_psd_welch, test_3_psd_welch, test_4_psd_welch = load_test_feature('psd_welch')

train_1_psd_multitaper, train_2_psd_multitaper, train_3_psd_multitaper, train_4_psd_multitaper = load_train_feature('psd_multitaper')
test_1_psd_multitaper, test_2_psd_multitaper, test_3_psd_multitaper, test_4_psd_multitaper = load_test_feature('psd_multitaper')

train_1_burg, train_2_burg, train_3_burg, train_4_burg =  load_train_feature('burg')
test_1_burg, test_2_burg, test_3_burg, test_4_burg = load_test_feature('burg')

train_1_cor, train_2_cor, train_3_cor, train_4_cor = load_train_feature('cor')
test_1_cor, test_2_cor,test_3_cor,test_4_cor = load_test_feature('cor')

train_1_entropy, train_2_entropy, train_3_entropy, train_4_entropy =  load_train_feature('entropy')
test_1_entropy, test_2_entropy, test_3_entropy, test_4_entropy = load_test_feature('entropy')

train_1_hdi, train_2_hdi, train_3_hdi, train_4_hdi = load_train_feature('hdi')
test_1_hdi, test_2_hdi, test_3_hdi, test_4_hdi = load_test_feature('hdi')

train_1_hilbert, train_2_hilbert, train_3_hilbert, train_4_hilbert = load_train_feature('hilbert')
test_1_hilbert, test_2_hilbert, test_3_hilbert, test_4_hilbert = load_test_feature('hilbert')

train_1_welch, train_2_welch, train_3_welch, train_4_welch = load_train_feature('welch')
test_1_welch, test_2_welch, test_3_welch, test_4_welch = load_test_feature('welch')

train_1_multitaper, train_2_multitaper, train_3_multitaper, train_4_multitaper  = load_train_feature('multitaper')
test_1_multitaper, test_2_multitaper, test_3_multitaper, test_4_multitaper = load_test_feature('multitaper')


sig_train1 = np.load('train\\carmen\\signals\\area_1_carmen_train_5017_signals.npy', allow_pickle=True)
sig_train2 = np.load('train\\carmen\\signals\\area_2_carmen_train_5017_signals.npy', allow_pickle=True)
sig_train3 = np.load('train\\carmen\\signals\\area_3_carmen_train_5017_signals.npy', allow_pickle=True)
sig_train4 = np.load('train\\carmen\\signals\\area_4_carmen_train_5017_signals.npy', allow_pickle=True)

sig_test1 = np.load('test\\carmen\\signals\\area_1_carmen_test_1075_signals.npy', allow_pickle=True)
sig_test2 = np.load('test\\carmen\\signals\\area_2_carmen_test_1075_signals.npy', allow_pickle=True)
sig_test3 = np.load('test\\carmen\\signals\\area_3_carmen_test_1075_signals.npy', allow_pickle=True)
sig_test4 = np.load('test\\carmen\\signals\\area_4_carmen_test_1075_signals.npy', allow_pickle=True)


sig_valid1 = np.load('validation\\carmen\\signals\\area_1_carmen_validation_1075_signals.npy', allow_pickle=True)
sig_valid2 = np.load('validation\\carmen\\signals\\area_2_carmen_validation_1075_signals.npy', allow_pickle=True)
sig_valid3 = np.load('validation\\carmen\\signals\\area_3_carmen_validation_1075_signals.npy', allow_pickle=True)
sig_valid4 = np.load('validation\\carmen\\signals\\area_4_carmen_validation_1075_signals.npy', allow_pickle=True)
