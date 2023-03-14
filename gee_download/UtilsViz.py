import matplotlib.pyplot as plt 
from skimage import io, exposure
import numpy as np 
import os 








def plotS2RGB(rgb_path):
    image = io.imread(rgb_path)
    image = exposure.rescale_intensity(image)
    # Plot the image using matplotlib
    plt.imshow(image)#, cmap='rgb')
    plt.title(f'S2RGB patch {os.path.basename(rgb_path)}')
    plt.axis('off')
   # plt.xlabel('X axis label')
   # plt.ylabel('Y axis label')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()


def plot_hist_and_box(vals, nbin,uquantile = 0.8,lquantile = 0.2):

    lq = np.quantile(vals, lquantile)
    hq = np.quantile(vals, uquantile)
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
    ax1.boxplot(vals)
    ax2.hist(vals, bins=nbin)
    ax2.axvline(lq, color='r', alpha=0.3, label = 'lq')
    ax2.axvline(hq, color='b', alpha=0.3, label = 'hq')
    plt.show()


