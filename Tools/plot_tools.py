import matplotlib.pyplot as plt
import numpy as np

def SSM_sig_plot(S, sig, func_i):

    fig1 = plt.figure()

    f_ax1 = plt.subplot2grid((3, 1), (0, 0))
    f_ax2 = plt.subplot2grid((3, 1), (1, 0))
    f_ax3 = plt.subplot2grid((3, 1), (2, 0))


    x1 = np.linspace(0, len(sig) / 1000, len(S))
    f_ax1.pcolormesh(x1, x1, S, cmap="YlGnBu")
    f_ax2.plot(sig)
    f_ax2.set_title("signal")
    f_ax2.set_xlim((0,len(sig)))
    f_ax3.plot(func_i)
    f_ax3.set_title("Extracted function")
    f_ax3.set_xlim((0, len(func_i)))

    plt.show()
