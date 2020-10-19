import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm


with open('experiment.pickle', 'rb') as f:
    stats = pickle.load(f)
    count = stats['count']
    activations = [k for k in stats.keys() if 'activation' in k]
    activations = sorted(activations)
    
    X,Y = np.meshgrid(np.arange(7), np.arange(7))
    
    fig, axs = plt.subplots()
    axs.set_xticks([])
    axs.set_yticks([])
    act_idx = 1
    axs.set_title('activation_%d' % act_idx, fontsize=26)
    val = 100.*stats['activation_%d_readout' % act_idx]['mean_top_k']/count
    cax = axs.imshow(100.*stats['activation_%d_readout' % act_idx]['mean_top_k']/count, vmin=1.0, vmax=10.0)
    tks = [np.min(val), np.min(val) + 0.5*(np.max(val)-np.min(val)),np.max(val)]
    cbar = fig.colorbar(cax, ticks=tks)
    cbar.ax.set_yticklabels(['%.1f %%' % t for t in tks], fontsize=20)
    plt.show()