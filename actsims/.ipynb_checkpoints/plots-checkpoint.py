import numpy as np
from matplotlib.pyplot import *


def bin1d(data, modlmap, lmax, bin_size):

    # Define the bins and bin centers
    bins = np.arange(0, lmax, bin_size)
    centers = (bins[1:] + bins[:-1])/2.

    # Bin the power spectrum 
    digitized = np.digitize(np.ndarray.flatten(modlmap), bins, right=True)
    binned = np.bincount(digitized, data.reshape(-1))[1:-1]/np.bincount(digitized)[1:-1]
    
    return centers, binned


def plot_1dspec(ps2d,modlmap,pout,lmin=500,lmax=8000,dell=50):  # Plot 1D power spectrum from input 2D spectrum
    fig, axs = subplots(nrows = 3, ncols = 3, figsize = (12, 12), gridspec_kw = {'wspace': 0.5, 'hspace': 0.3})

    for i in range(3):
        for j in range(i, 3):
            # Get 1d binned spectrum
            ell, c_ell = bin1d(ps2d[i, j], modlmap, lmax, dell)
            mask = ell < lmin
            ell_fit = ell[~mask]
            c_ell_fit = c_ell[~mask]
            # Plot
            axs[i, j].plot(ell_fit, c_ell_fit, linestyle = 'none', marker = 'o', markerfacecolor = 'none') # Data
            
            comp1 = {0: 'I', 1: 'Q', 2: 'U'}[i]
            comp2 = {0: 'I', 1: 'Q', 2: 'U'}[j]
            axs[i, j].set_title(f'{comp1}X{comp2}')
        
            axs[i, j].set_xlabel('$\ell$')
            axs[i, j].set_ylabel('$C_\ell$')
            axs[i, j].set_yscale('linear')
            if i is not j:
                axs[j, i].set_axis_off()
    savefig(pout+'_1d_noise-spec.png')


    