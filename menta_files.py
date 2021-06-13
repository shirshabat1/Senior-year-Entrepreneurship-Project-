import numpy as np

START_TRAIN = 'train\\menta\\features\\all\\'
START_TEST = 'test\\menta\\features\\all\\'
START_VALID = 'validation\\menta\\features\\all\\'

START_TRAIN_SIGNALS = 'train\\menta\\signals'
START_TEST_SIGNALS = 'test\\menta\\signals'
START_VALID_SIGNALS = 'validation\\menta\\signals'

AREA_1_TRAIN = '\\area_1_menta_train_2876_signals.npy'
AREA_2_TRAIN = '\\area_2_menta_train_2876_signals.npy'
AREA_3_TRAIN = '\\area_3_menta_train_2876_signals.npy'
AREA_4_TRAIN = '\\area_4_menta_train_2876_signals.npy'

AREA_1_TEST = '\\area_1_menta_test_616_signals.npy'
AREA_2_TEST = '\\area_2_menta_test_616_signals.npy'
AREA_3_TEST = '\\area_3_menta_test_616_signals.npy'
AREA_4_TEST = '\\area_4_menta_test_616_signals.npy'

AREA_1_VALID = '\\area_1_menta_validation_617_signals.npy'
AREA_2_VALID = '\\area_2_menta_validation_617_signals.npy'
AREA_3_VALID = '\\area_3_menta_validation_617_signals.npy'
AREA_4_VALID = '\\area_4_menta_validation_617_signals.npy'


def load_data(area1, area2, area3, area4, path):
    data_area1 = np.load(path + area1, allow_pickle=True)
    data_area2 = np.load(path +area2, allow_pickle=True)
    data_area3 = np.load(path +area3, allow_pickle=True)
    data_area4 = np.load(path +area4, allow_pickle=True)
    return data_area1, data_area2, data_area3, data_area4



def load_files(start_path, monkey, type):
    files = []
    for i in range(1, 5):
        files.append(np.load(start_path + str(monkey) + '_area_' + str(i) + '_' + str(type) + '_features.npy',
                             allow_pickle=True))
    return files


menta_validation_features = load_files(START_VALID, 'menta', 'validation')
menta_train_features = load_files(START_TRAIN, 'menta', 'train')
menta_test_features = load_files(START_TEST, 'menta', 'test')


train1, train2, train3, train4 = load_data(AREA_1_TRAIN, AREA_2_TRAIN, AREA_3_TRAIN, AREA_4_TRAIN, START_TRAIN_SIGNALS)
test1, test2, test3, test4 = load_data(AREA_1_TEST, AREA_2_TEST, AREA_3_TEST, AREA_4_TEST, START_TEST_SIGNALS)
valid1, valid2, valid3, valid4 = load_data(AREA_1_VALID, AREA_2_VALID, AREA_3_VALID, AREA_4_VALID, START_VALID_SIGNALS)


