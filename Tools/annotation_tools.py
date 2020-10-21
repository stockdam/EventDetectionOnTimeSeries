def FeatureExtraction(or_signal, fs, time_scale, perc_overlap=0.9):
    """
    Computes the extraction of featues of a signal or a group of signals

    :param or_signal: signal or signals from which features will be extracted
    :param fs: sampling frequency
    :param time_scale: time scale at which features will be extracted. It defines the sliding window size, which is half the time scale times the sampling frequency
    :param perc_overlap: overlap percentage of the sliding window
    :return: Feature matrix, feature dataframe with features by name and group, and sampling frequency of the extracted features
    """
    
    #temporal adjustments
    win_len = fs * time_scale//2
    # sampling frequency of reduced dimension
    new_fs = fs / (win_len * (1 - perc_overlap))
    sens = 1

    # set features to extract
    features = load_Json("D:\PhD\Code\PhDProject\AnnotationConfigurationFiles\config1.json")["features"]

    # extract features
    feat_file, featMat, feature_names = featuresExtraction(or_signal, fs, int(win_len), int(perc_overlap * win_len),
                                                           features)

    return featMat, new_fs


def EventDetection(signal, time_scale, kernel_size, fs, perc_overlap, labels):
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

    #Extract the feature matrix, which is represented by the PCA
    print("extract features")
    X_pca, feat_Mat, new_fs = FeatureExtraction(signal, fs, time_scale, perc_overlap=perc_overlap)


    #Extract Novelty Function from the Self-Similarity Matrix computed with the X_pca
    S, nov_ssm = NoveltyFunctionfromFeatures(np.array(feat_Mat["allfeatures"]), new_fs, Kernel_size=int(new_fs*kernel_size), downsampling=False, normalization=True)


    return S, nov_ssm, new_fs

def compute_SM_dot(X,Y):
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
    taper = np.sqrt(1/2)/(L*var)
    axis = np.arange(-L,L+1)
    gaussian1D = np.exp(-taper**2 * (axis**2))
    gaussian2D = np.outer(gaussian1D,gaussian1D)
    kernel_box = np.outer(np.sign(axis),np.sign(axis))
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
