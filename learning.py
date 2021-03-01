import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
import librosa
from sklearn import svm
from sklearn import metrics


import matplotlib.pyplot as plt
import featureResearch as feat
import neurokit2 as nk
from scipy.signal import hilbert, chirp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import RandomForestClassifier
SAMPLE_RATE = 1000



def extract_mel_chroma_first_fft_feature(X):
    X = X.astype(float)
    stft = np.abs(librosa.stft(X))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=SAMPLE_RATE).T, axis=0)
    # print("chroma: ", np.shape(chroma))
    mel = np.mean(librosa.feature.melspectrogram(X, sr=SAMPLE_RATE).T, axis=0)
    # print("mel :", np.shape(mel))
    fft = np.fft.fft(X) / len(X)
    fft = np.abs(fft[: len(X) // 7])
    fft = [fft[0]]

    return chroma, mel, fft
def extract_welch(X):
    X = X.astype(float)
    welch = nk.signal_psd(X, method="welch", min_frequency=1, max_frequency=20, show=False)["Power"]
    welch = np.array(welch)
    return welch

def extract_multitaper(X):
    multitaper = nk.signal_psd(X, method="multitapers", max_frequency=20, show=False)["Power"]
    multitaper = np.array(multitaper)
    return multitaper


def extract_hilbert(X):
    X = X.astype(float)
    analytic_signal = hilbert(X)
    analytic_signal = [np.linalg.norm(analytic_signal)]
    return analytic_signal

def extract_cor(X):
    X = X.astype(float)
    cor = np.corrcoef(X, X)
    cor = np.array([cor[0][1], cor[1][0]])
    # print("cor: ", np.shape(cor))
    return cor

def extract_entropy(X):
    sample = nk.entropy_sample(X)
    entropy = nk.entropy_approximate(X)
    return [sample, entropy]

def extract_hdi(X):
    X = X.astype(float)
    ci_min, ci_max = nk.hdi(X, ci=0.95, show=True)
    hdi = np.array([ci_max, ci_min])
    return hdi




def extract_burg(X):
    X = X.astype(float)
    burg = nk.signal_psd(X, method="burg", min_frequency=1, max_frequency=20, order=10, show=False)["Power"]
    burg = np.array(burg)
    return  burg

def extract_band(X, method):
    X = X.astype(float)
    return get_band(X, method)

def get_features(area):
    """
    this method takes only mel, chroma and the first fft features
    """
    features = []
    for x in tqdm(area):
        chroma, mel, fft = extract_mel_chroma_first_fft_feature(x)
        feature = np.concatenate((chroma, mel, fft))
        features.append(feature)
    return np.array(features)


def get_additional_features(area, feature):
    features = []
    for x in tqdm(area):
        if feature == 'cor':
            y = extract_cor(x)
        elif feature == 'welch':
            y = extract_welch(x)
        elif feature == 'entropy':
            y = extract_entropy(x)
        elif feature == 'hdi':
            y = extract_hdi(x)
        elif feature == 'burg':
            y = extract_burg(x)
        features.append(y)
    return np.array(features)

def get_specific_additional_features(area, method):
    features = []
    for x in tqdm(area):
        y = extract_band(x, method)
        features.append(y)
    return np.array(features)

def bandpower(data, sf, band, window_sec=None, relative=False):
    """Compute the average power of the signal x in a specific frequency band.

    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds.
        If None, window_sec = (1 / min(band)) * 2
    relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.

    Return
    ------
    bp : float
        Absolute or relative band power.
    """
    from scipy.signal import welch
    from scipy.integrate import simps
    band = np.asarray(band)
    low, high = band

    # Define window length
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        nperseg = (2 / low) * sf

    # Compute the modified periodogram (Welch)
    freqs, psd = welch(data, sf, nperseg=nperseg)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule.
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp
#
def get_band(X, method):
    fft = np.fft.fft(X) / len(X)
    fft = np.abs(fft[: len(X) // 7])
    psd_average = []
    y = np.mean(np.array(nk.signal_psd(fft, method=method, min_frequency=1, max_frequency=3, show=False)["Power"]))
    psd_average.append(y)
    y = np.mean(np.array(nk.signal_psd(fft, method=method, min_frequency=4, max_frequency=7, show=False)["Power"]))
    psd_average.append(y)
    y = np.mean( np.array(nk.signal_psd(fft, method=method, min_frequency=8, max_frequency=12, show=False)["Power"]))
    psd_average.append(y)
    y = np.mean(np.array(nk.signal_psd(fft, method=method, min_frequency=13, max_frequency=22, show=False)["Power"]))
    psd_average.append(y)
    y = np.mean(np.array(nk.signal_psd(fft, method=method, min_frequency=23, max_frequency=34, show=False)["Power"]))
    psd_average.append(y)
    y = np.mean(np.array(nk.signal_psd(fft, method=method, min_frequency=35, max_frequency=45, show=False)["Power"]))
    psd_average.append(y)
    y = np.mean(np.array(nk.signal_psd(fft, method=method, min_frequency=56, max_frequency=95, show=False)["Power"]))
    psd_average.append(y)
    y = np.mean(np.array(nk.signal_psd(fft, method=method, min_frequency=105, max_frequency=195, show=False)["Power"]))
    psd_average.append(y)
    return psd_average


def save_data(res1, res2, res3, res4, path, monkey):
    np.save(
        path + 'area_' + str(1) + '_' + str(monkey) + '_features_extra_validation_' + str(len(res1)) + '_signals.npy',
        np.array(res1))
    np.save(
        path + 'area_' + str(2) + '_' + str(monkey) + '_features_extra_validation_' + str(len(res2)) + '_signals.npy',
        np.array(res2))
    np.save(
        path + 'area_' + str(3) + '_' + str(monkey) + '_features_extra_validation_' + str(len(res3)) + '_signals.npy',
        np.array(res3))
    np.save(
        path + 'area_' + str(4) + '_' + str(monkey) + '_features_extra_validation_' + str(len(res4)) + '_signals.npy',
        np.array(res4))


def getY(area1, area2, area3, area4):
    y_1 = [0] * len(area1)
    y_2 = [1] * len(area2)
    y_3 = [2] * len(area3)
    y_4 = [3] * len(area4)
    y = y_1 + y_2 + y_3 + y_4
    return y


def getX(area1, area2, area3, area4):
    x = np.concatenate((area1, area2, area3, area4))
    return x


def SVM(x_train, y_train, x_test, y_test):
    clf = svm.SVC(C=0.1, kernel='linear')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("SVM Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("SVM confusion: ")
    print(metrics.confusion_matrix(y_test, y_pred))




def SVMovr(x_train, y_train, x_test, y_test):
    clf = svm.SVC(decision_function_shape='ovr')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("SVC ONE VS REST Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("SVMovr confusion: ")
    print(metrics.confusion_matrix(y_test, y_pred))



def linearSVC(x_train, y_train, x_test, y_test):
    clf = svm.LinearSVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(len(x_train))
    print(len(y_train))
    print("Linear SVC Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("linearSVC confusion: ")
    print(metrics.confusion_matrix(y_test, y_pred))




def randomForest(x_train, y_train, x_test, y_test):
    """Random Forest Accuracy: 0.6769767441860465"""
    clf = RandomForestClassifier(
        max_depth=50,
        min_samples_leaf=200,
        n_estimators=200,
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        random_state=50,
        max_features='auto')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("Random Forest Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("confusion_matrix Random Forest Accuracy:")
    print(metrics.confusion_matrix(y_test, y_pred))



def adaBoost(x_train, y_train, x_test, y_test):
    abc = AdaBoostClassifier(n_estimators=200,learning_rate=1)
    abc.fit(x_train, y_train)
    y_pred = abc.predict(x_test)
    print("boost Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("boost confusion: ")
    print(metrics.confusion_matrix(y_test, y_pred))




def GradientBoosting(x_train, y_train, x_test, y_test):
    clf = GradientBoostingClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("Gradient Boosting Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Gradient confusion: ")
    print(metrics.confusion_matrix(y_test, y_pred))




def KNeighbors(x_train, y_train, x_test, y_test):
    """
    KNeighbors Accuracy: 0.6909302325581396
    """
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    # y_pred = clf.predict(x_train)
    print("KNeighbors Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("KNeighbors confusion: ")
    print(metrics.confusion_matrix(y_test, y_pred))



def lda(x_train, y_train, x_test, y_test):
    clf = LDA()
    clf.fit(x_train, y_train)
    LDA(n_components=None, priors=None, shrinkage=None, solver='svd',store_covariance=False, tol=0.0001)
    y_pred = clf.predict(x_test)
    print("lda Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("lda confusion: ")
    print(metrics.confusion_matrix(y_test, y_pred))



def qda(x_train, y_train, x_test, y_test):
    clf = QDA()
    clf.fit(x_train, y_train)
    QDA(priors=None, reg_param=0.0)
    y_pred = clf.predict(x_test)
    print(len(x_train))
    print(len(y_train))
    print("qda Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("qda confusion: ")
    print(metrics.confusion_matrix(y_test, y_pred))



def mutualInfo(x, y, monkey):
    mi = mutual_info_classif(x, y, discrete_features=False)
    index1 = np.argmax(mi)
    old = mi[index1]
    mi[index1] = -1
    index2 = np.argmax(mi)

    mi[index1] = old

    plt.scatter(range(len(mi)), mi)
    plt.xlabel("feature index")
    plt.ylabel("mutual info")
    plt.title(monkey)
    plt.show()

    print(index1)
    print(index2)



def getFeaturesAndNormalize(data):
    """
    Doesn't help. The results are not good with this one
    """
    features = []
    for sig in tqdm(data):
        feature = feat.extract_feature(feat.dc_normalize(sig))
        features.append(feature)
    return np.array(features)

