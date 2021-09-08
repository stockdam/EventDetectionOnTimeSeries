from tools.load_tools import load_Json
from tools.tsfel_tools import load_featuresbydomain, featuresTsfelMat
from tools.processing_tools_final import normalize_feature_sequence_z
from tools.tsfel_tools import chunk_data

from sklearn.metrics.pairwise import euclidean_distances

import numpy as np

import matplotlib.pyplot as plt

def featuresExtraction(signal, fs, win_size, overlap_size, features):
    """
    Process of extracting features with methods from tsfel. It returns two dictionnarues with
    1) the feature file where the original signal and all the feature components are stored
    with the name of the features
    2) The feature name dictionnary where the tag of each feature is stored

    :param signal:  Original signal(s) from which the features will be extracted
    :param fs: sampling frequency (int)
    :param win_size: size of the sliding window
    :param overlap_size: overlaping size, if int, the value, if between 0-1, the percentage
    :return: 2 dictionnaries:
    1 - feature_file: Array with dicts for each signal from which features are extracted
    np.array(
			[{"signal": original signal, "features": matrix with features}])
	2 - feature_dict:
	dict(featurebydomain={"temp": [], "stat": [], "spec": []}, allfeatures=[])
	3 - feature_names:
	dict(featurebydomain={"temp": [], "stat": [], "spec": []}, allfeatures=[])

	TODO: Not yet consolidated the multisignal purposes
    """
    feature_file = featuresTsfelMat(signal, fs, win_size, overlap_size, features)

    if (np.ndim(signal) > 1):
        #first case
        feature_dict, featuredict_names = load_featuresbydomain(feature_file[0]["features"], "all")
        feature_dict={"allfeatures":feature_dict["allfeatures"]}
        featuredict_names={"allfeatures":featuredict_names["allfeatures"]}
        for i in range(1, len(feature_file)):
            feature_dict_, featuredict_names_ = load_featuresbydomain(feature_file[i]["features"], "all")
            # print(np.shape(feature_dict["allfeatures"]))
            feature_dict["allfeatures"] = np.vstack([feature_dict["allfeatures"], feature_dict_["allfeatures"]])
            featuredict_names["allfeatures"] = np.hstack([featuredict_names["allfeatures"], featuredict_names_["allfeatures"]])
    else:
        feature_dict, featuredict_names = load_featuresbydomain(feature_file[0]["features"], "all")
    return feature_file, feature_dict, featuredict_names

# def tsfel_feature_extraction(or_signal, fs, time_scale, perc_overlap):
#

def ExtractFeatureMatrix(or_signal, fs, time_scale, perc_overlap=0.9):
    """
    Computes the extraction of featues of a signal or a group of signals
    :param or_signal: signal or signals from which features will be extracted
    :param fs: sampling frequency
    :param time_scale: time scale at which features will be extracted. It defines the sliding window size, which is half the time scale times the sampling frequency
    :param perc_overlap: overlap percentage of the sliding window
    :return: Feature matrix, feature dataframe with features by name and group, and sampling frequency of the extracted features
    """

    # temporal adjustments
    win_len = fs * time_scale // 2
    # sampling frequency of reduced dimension
    new_fs = fs / (win_len * (1 - perc_overlap))
    sens = 1

    # set features to extract
    features = load_Json("config1.json")["features"]
    # features = ["stat_c_std"]
    # extract features
    feat_file, featMat, feature_names = featuresExtraction(or_signal, fs, int(win_len), int(perc_overlap * win_len),
                                                           features)

    return featMat, new_fs

def EuclideanMatrix(sig, fs, time_scale, perc_overlap=0.9):
    # inputSignal = inputSignal - np.mean(inputSignal)

    # temporal adjustments
    win_len = fs * time_scale // 2
    overlap_size = int(perc_overlap * win_len)
    # sampling frequency of reduced dimension
    new_fs = fs / (win_len * (1 - perc_overlap))
    if(np.ndim(sig)>1):
        sens = 1
        s_matrix = []
        WinRange = int(win_len / 2)
        sig = np.r_[sig[WinRange:0:-1,:], sig, sig[-1:len(sig) - WinRange:-1,:]].transpose()
        for s_i in range(np.shape(sig)[0]):
            s_t = sig[s_i]
            s_temp = np.copy(s_t)
            # sig_a = chunk_data(s_temp, window_size=window_len, overlap_size=overlap_size)*(win/win.sum())
            sig_a = chunk_data(s_temp, window_size=int(win_len), overlap_size=overlap_size)
            dist_mat = euclidean_distances(sig_a, sig_a)

            s_matrix.append(dist_mat)

        return s_matrix

    else:
        s_temp = np.copy(sig)
        sig_a = chunk_data(s_temp, window_size=int(win_len), overlap_size=overlap_size)
        dist_mat = euclidean_distances(sig_a, sig_a)
        return dist_mat



def EventDetection(signal, time_scale, kernel_size, fs, perc_overlap, method="features"):
    """
    Computes the function that applies the functions to detect the events on the signal. It returns the novelty function
    after extracting the features from the signal or set of signals.
    :param signal: 1-D signal or N-D signal
    :param time_scale: time scale at which features will be extracted. It defines the sliding window size, which is half the time scale times the sampling frequency
    :param kernel_size: size of the sliding kernel that will compute the novelty function
    :param fs: sampling frequency
    :param perc_overlap: overlap percentage of the sliding window
    :param labels: json with set of features to be extracted
    :return S: SSM matrix
    :return nov_ssm: novelty function
    :return new_fs: sampling frequency of extracted features
    """
    if(method=="features"):
        # Extract the feature matrix, which is represented by the PCA
        print("extract features")
        feat_Mat, new_fs = ExtractFeatureMatrix(signal, fs, time_scale, perc_overlap=perc_overlap)

        print("Computing SSM")
        # Extract Novelty Function from the Self-Similarity Matrix computed with the X_pca
        S = ComputerSSM(np.array(feat_Mat["allfeatures"]), normalization=True)

    elif(method=="euclidean"):
        S = EuclideanMatrix(signal, fs, time_scale, perc_overlap)
        new_fs = 0

    print("Computing novelty function")
    nov_ssm = NoveltyFunctionfromFeatures(S, Kernel_size=int(kernel_size))

    print("plotting...")

    return S, nov_ssm, new_fs


def compute_SM_dot(X, Y):
    """Computes similarty matrix from feature sequences using dot (inner) product
        Notebook: C4/C4S2_SSM.ipynb
        """
    S = np.dot(np.transpose(Y), X)
    return S


def compute_kernel_checkerboard_Gaussian(L, var=1, normalize=True):
    """
    This code is used from https://www.audiolabs-erlangen.de/resources/MIR/FMP/C4/C4S4_NoveltySegmentation.html
    Check the Notebook C4 from Fundamentals of Music Processing (FMP).
    Compute Guassian-like checkerboard kernel
    Uses bivariate guassian distribution to define the kernel
    See also: https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/
    :param L: Parameter specifying the kernel size M=2*L+1
    :param var: Variance parameter determing the tapering (epsilon)
    :return kernel: Kernel matrix of size M x M
    """
    taper = np.sqrt(1 / 2) / (L * var)
    axis = np.arange(-L, L + 1)
    gaussian1D = np.exp(-taper ** 2 * (axis ** 2))
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    kernel_box = np.outer(np.sign(axis), np.sign(axis))
    kernel = kernel_box * gaussian2D

    if normalize:
        kernel = kernel / np.sum(np.abs(kernel))
    return kernel


def compute_novelty_SSM(S, kernel=None, L=10, var=0.5, exclude=False):
    """This code is used from https://www.audiolabs-erlangen.de/resources/MIR/FMP/C4/C4S4_NoveltySegmentation.html
    Check the Notebook C4 from Fundamentals of Music Processing (FMP).
    :param S: SSM
    :param kernel: Checkerboard kernel (if kernel==None, it will be computed)
    :param L: Parameter specifying the kernel size M=2*L+1
    :param var: Variance parameter determing the tapering (epsilon)
    :param exclude: Sets the first L and last L values of novelty function to zero
    :return nov: Novelty function
    """
    if kernel is None:
        kernel = compute_kernel_checkerboard_Gaussian(L=L, var=var)
        # kernel = compute_kernel_checkerboard_box(L=L)
    N = S.shape[0]
    M = 2 * L + 1
    nov = np.zeros(N)
    # np.pad does not work with numba/jit
    S_padded = np.pad(S, L, mode='constant')

    for n in range(N):
        # Does not work with numba/jit
        nov[n] = np.sum(S_padded[n:n + M, n:n + M] * kernel)
    if exclude:
        right = np.min([L, N])
        left = np.max([0, N - L])
        nov[0:right] = 0
        nov[left:N] = 0

    return nov

def PeriodicEventDetection(signal, time_scale, perc_overlap, fs):
    print("extract features")
    feat_Mat, new_fs = ExtractFeatureMatrix(signal, fs, time_scale, perc_overlap=perc_overlap)

    print("Computing SSM")
    # Extract Novelty Function from the Self-Similarity Matrix computed with the X_pca
    S = ComputerSSM(np.array(feat_Mat["allfeatures"]), normalization=True)

    #periodic_info
    sim_function = np.sum(S, axis=0)

    return S, sim_function

def ComputerSSM(F_set, normalization=False):
    #Normalization
    if(normalization):
        F_set = normalize_feature_sequence_z(F_set)

    #Compute S matrix
    S = np.transpose(compute_SM_dot(F_set, F_set))

    return S


def NoveltyFunctionfromFeatures(SSM, Kernel_size=10):
    """
    :param F_set:
    :param Fs:
    :param Kernel_size:
    :param filt_len:
    :param downsampling_ratio:
    :param normalization:
    :param downsampling:
    :return:
    """

    #compute novelty function
    nov_ssm = compute_novelty_SSM(SSM, L=Kernel_size, var=0.5)

    return nov_ssm
