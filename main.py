from tools.annotation_tools_final import *
from tools.load_tools import loadH5

all_signals1 = loadH5("sig_example.h5")

S, nov_ssm, new_fs = EventDetection(all_signals1[:, 8], 2, 2, 1000, 0.95, "features")

plt.plot(nov_ssm)
plt.show()