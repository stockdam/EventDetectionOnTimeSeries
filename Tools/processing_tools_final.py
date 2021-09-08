import numpy as np
from numpy.lib.stride_tricks import as_strided as ast

def normalize_feature_sequence_z(X, threshold=0.0001, v=None):
    K, N = X.shape
    X_norm = np.zeros((K, N))

    if v is None:
        v = np.zeros(K)

    for n in range(N):
        mu = np.sum(X[:, n]) / K
        sigma = np.sqrt(np.sum((X[:, n] - mu) ** 2) / (K - 1))
        if sigma > threshold:
            X_norm[:, n] = (X[:, n] - mu) / sigma
        else:
            X_norm[:, n] = v

    return X_norm

def chunk_data(data,window_size,overlap_size=0,flatten_inside_window=True):
    """
    Gives a matrix with all the windows of the signal separated by window size and overlap size.
    :param data:
    :param window_size:
    :param overlap_size:
    :param flatten_inside_window:
    :return: matrix with signal windowed based on window_size and overlap_size
    """
    assert data.ndim == 1 or data.ndim == 2
    if data.ndim == 1:
        data = data.reshape((-1,1))

    # get the number of overlapping windows that fit into the data
    num_windows = (data.shape[0] - window_size) // (window_size - overlap_size) + 1
    overhang = data.shape[0] - (num_windows*window_size - (num_windows-1)*overlap_size)

    # if there's overhang, need an extra window and a zero pad on the data
    # (numpy 1.7 has a nice pad function I'm not using here)
    if overhang != 0:
        num_windows += 1
        newdata = np.zeros((num_windows*window_size - (num_windows-1)*overlap_size,data.shape[1]))
        newdata[:data.shape[0]] = data
        data = newdata

    sz = data.dtype.itemsize
    ret = ast(
        data,
        shape=(num_windows,window_size*data.shape[1]),
        strides=((window_size-overlap_size)*data.shape[1]*sz, sz)
    )

    if flatten_inside_window:
        return ret
    else:
        return ret.reshape((num_windows,-1,data.shape[1]))


def mean_norm(sig):
	a = sig-np.mean(sig)
	return a/max(a)

def loadfeaturesbydomain_sub(features, featureSet, featureSet_names):

    for feature in features.keys():

        # print(np.where(np.isnan(features["features"][feature])))
        if (feature in ["spec_m_coeff", "temp_mslope"]):
            continue
        elif(len(np.where(np.isnan(np.array(features[feature])))[0])>0):
            print(feature)
            continue
        elif(np.sum(abs(features[feature]))==0):
            print(feature)
            continue
        else:
            # print(feature)
            # print(len(features["features"][feature]))
            signal_i = features[feature]
            signal_i = mean_norm(signal_i)

            featureSet['allfeatures'].append(signal_i)
            featureSet_names["allfeatures"].append(feature)
            if ("temp" in feature):
                featureSet['featurebydomain']["temp"].append(signal_i)
                featureSet_names['featurebydomain']["temp"].append(feature)
            elif ("spec" in feature):
                featureSet['featurebydomain']["spec"].append(signal_i)
                featureSet_names['featurebydomain']["spec"].append(feature)
            else:
                featureSet['featurebydomain']["stat"].append(signal_i)
                featureSet_names['featurebydomain']["stat"].append(feature)
    return featureSet, featureSet_names
