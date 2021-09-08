import h5py
import numpy as np
import os
import json
import csv
from scipy.io import arff
import wfdb


def physionet_data(file):
    """

    :param file: should not have the data name
    :return:
    """

    record = wfdb.rdrecord(file)

    return record.p_signal, record.fs

def physionet_data_with_ann(file, ann):
    """

    :param file: should not have the data name
    :return:
    """

    record = wfdb.rdrecord(file)

    if(len(ann)>1):
        ann_array = []
        ann_symbol_array = []
        for a in ann:
            print(a)
            ann_i = wfdb.rdann(file, a)
            ann_array.append(ann_i.sample)
            ann_symbol_array.append(ann_i.symbol)
    else:
        ann_array = wfdb.rdann(file, ann[0])
        ann_symbol_array = ann_array.symbol
        ann_array = ann_array.sample

    return record.p_signal, record.fs, ann_array, ann_symbol_array



def load_npz(filepath):
    npz_file = np.load(filepath, allow_pickle=True)

    return npz_file

def load_txtfile(filepath):
    #read annotation text file
    with open(filepath) as f:
        read_data = f.read()

    return read_data

def load_npz_featuresHui(filepath):
    """
    0 EMG1

    1 EMG2

    2 EMG3

    3 EMG4

    4 Airborne Microphone

    5 Piezoelectric Microphone (Respiration) - Please don't regard this
    channel. It's not stable.

    6 ACC Upper X

    7 ACC Upper Y

    8 ACC Upper Z

    9 Goniometer X

    10 ACC Lower X

    11 ACC Lower Y

    12 ACC Lower Z

    13 Goniometer Y

    14 Gyro Upper X

    15 Gyro Upper Y

    16 Gyro Upper Z

    17 Force Sensor - Please don't regard this channel. It's breakable and
    the signal quality is worse.

    18 Gyro Lower X

    19 Gyro Lower Y

    20 Gyro Lower Z

    21 Pushbutton

    :param filepath:
    :return:
    """
    npz_f = load_npz(filepath)
    object_f = npz_f["arr"]

    return object_f

def loadH52(file):
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%% Creation of h5py object %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    h5_object = h5py.File(file, mode="r")
    import matplotlib.pyplot as plt
    # %%%%%%%%%%%%%%%%%%%%%%%%% Data of the selected channels %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    out_dict = np.array([])
    channels = h5_object.get('00:07:80:46:E5:C4').get("raw")
    for chn in channels:
        print(chn)
        data_temp = h5_object.get('00:07:80:46:E5:C4').get("raw").get(chn)

        out_dict = np.append(out_dict, data_temp)
    out_dict = np.reshape(out_dict, (len(channels), len(data_temp)))
    print(out_dict.shape)
    return out_dict


def loadH5(file):

    dataFile = h5py.File(file, "r")
    dataSet = dataFile["data"][:]
    dataFile.close()
    signal = dataSet

    return signal

def load_Json(file):
    with open(file) as js_file:
        data = json.load(js_file)

    return data


def load_arff(file):
    data, meta = arff.loadarff(file)
    return data

def convert_dat2csv(filename, path):
    # read flash.dat to a list of lists
    datContent = [i.strip().split(";") for i in open(path+"\\"+filename+".dat").readlines()]
    datContent = [x for x in datContent if x != ['']]
    print(datContent)

    # write it as a new CSV file
    with open(path+"\\"+filename+".csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(datContent)

def load_sensor_IOTIP(path, station_type = "Green", sensor_type="Accelerometer"):
    """

    :param path: folder directory
    :param sensor_type: type of sensor to load
    :return: array, shape(N,5)
    """

    iotip_folders = os.listdir(path)

    for folder in iotip_folders:
        if(station_type in folder):
            return np.loadtxt(path+'/'+folder+'/'+sensor_type+'.txt', delimiter=",")
        else:
            print("No file found corresponding to this structure")
            print(iotip_folders)