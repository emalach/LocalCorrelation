import numpy as np
import pickle
import matplotlib.pyplot as plt

classes = ['airplane', 'automobile', 'bird',    'cat',     'deer',    'dog',     'frog',    'horse', 'ship', 'truck']
colors =  ['#1f77b4',  '#e377c2'   , '#ff7f0e', '#d62728', '#9467bd', '#8c564b', '#2ca02c', '#7f7f7f', '#bcbd22', '#17becf']
with open('patches_experiment_cifar_10_linear_epochs.pickle', 'rb') as f:
    stats = pickle.load(f)
    count = stats['count']
    print('minimum mean_acc', np.min(stats['mean_acc'])/stats['count'])
    print('maximum mean_acc', np.max(stats['mean_acc'])/stats['count'])
    print('minimum top_k', np.min(stats['mean_top_k'])/stats['count'])
    print('maximum top_k', np.max(stats['mean_top_k'])/stats['count'])
    
    for i in range(4,16):
        #plt.imshow(stats['sample_x'][i])
        fig, axs = plt.subplots(8,8)
        for x in range(8):
            for y in range(8):
                patch = stats['sample_x'][i,y*4+1:y*4+4, x*4+1:x*4+4]
                axs[y,x].set_aspect('equal', 'box')
                axs[y,x].imshow(patch)
                axs[y,x].set_xticks([])
                axs[y,x].set_yticks([])
                pred = np.argmax(stats['sample_pred'][i,y,x])
                axs[y,x].text(-0.4,0.5,classes[pred][:2].upper(), fontweight='bold', fontsize=14, color=colors[pred], bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 0.5})
        fig.show()
        fg = plt.figure()
        plt.imshow(stats['sample_x'][i])
        plt.axis('off')
        plt.draw()
        while plt.waitforbuttonpress():
            pass
        plt.close(fig)
        plt.close(fg)
