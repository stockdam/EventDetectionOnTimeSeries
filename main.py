from Tools.annotation_tools_final import *
from Tools.load_tools import loadH5
from Tools.plot_tools import *

all_signals1 = loadH5("sig_example.h5")

sig = all_signals1[:, 8]
time_scale = 2 #in seconds
fs = 1000 #the fs is used combined with the time scale to define the time window
kernel_size = 2 #in samples
overlap_perc = 0.9 #percentage of the time window

#compute novelty function
S, nov_ssm, new_fs = EventDetection(signal=sig, time_scale=time_scale, kernel_size=kernel_size, fs=fs, perc_overlap=overlap_perc, method="features")
SSM_sig_plot(S, sig, nov_ssm)

#compute periodicity function
S, sim_function = PeriodicEventDetection(signal=sig, time_scale=time_scale, fs=fs, perc_overlap=overlap_perc)
SSM_sig_plot(S, sig, sim_function)
