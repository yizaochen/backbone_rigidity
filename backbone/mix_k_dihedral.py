import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class MixPlot1:
    resid_lst = list(range(4, 19))
    lbfz = 12
    lgfz = 12
    k_labels = ["C2'(i)-P(i+1)", "C2'(i)-O1P(i+1)", "C4'(i)-O5'(i+1)"]

    def __init__(self, host, strand_id):
        self.host = host
        self.strand_id = strand_id
        self.nrows = 6
        self.n_resid = len(self.resid_lst)
        self.cmap = 'Reds'

    def plot_main(self, figsize, hspace, bottom, top):
        fig = plt.figure(figsize=figsize, facecolor='white')
        d_axes = self.get_d_axes(fig, hspace)
        self.remove_xticks(d_axes)
        self.set_xlabel_xticks(d_axes)
        self.set_ylabels(d_axes)
        for idx in range(3):
            self.plot_errorbar_k(d_axes, idx)
        for idx in range(3, 6):
            self.heatmap(d_axes, idx, bottom, top)
        return fig, d_axes

    def plot_errorbar_k(self, d_axes, idx):
        k_mean_array, k_std_array = self.get_k_mean_std_array()
        d_axes[idx].errorbar(self.resid_lst, k_mean_array, fmt='-o', yerr=k_std_array, label=self.k_labels[idx])
        d_axes[idx].legend(fontsize=self.lgfz, frameon=False)

    def get_k_mean_std_array(self):
        k_mean_array = np.random.rand(self.n_resid)
        k_std_array = np.random.rand(self.n_resid)
        return k_mean_array, k_std_array

    def heatmap(self, d_axes, idx, bottom, top):
        data_mat = np.random.rand(360, self.n_resid)
        d_axes[idx].imshow(data_mat, cmap=self.cmap, origin='lower', extent=self.get_extent(bottom, top))
        d_axes[idx].set_yticks([bottom, top])

    def get_extent(self, bottom, top):
        xoffset = 0.5
        return (4-xoffset, 18+xoffset, bottom, top)

    def get_d_axes(self, fig, hspace):
        d_axes = dict()
        spec = gridspec.GridSpec(ncols=1, nrows=self.nrows, figure=fig, hspace=hspace)
        for idx in range(self.nrows):
            d_axes[idx] = fig.add_subplot(spec[idx])
        return d_axes

    def remove_xticks(self, d_axes):
        for idx in range(self.nrows-1):
            d_axes[idx].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    def set_xlabel_xticks(self, d_axes):
        ax = d_axes[self.nrows-1]
        ax.set_xticks(self.resid_lst)
        ax.set_xlabel('Resid', fontsize=self.lbfz)
        ax.set_xlim(3.5, 18.5)

    def set_ylabels(self, d_axes):
        for idx in range(3):
            d_axes[idx].set_ylabel('k (kcal/mol/Å$^2$)', fontsize=self.lbfz)
        for idx in range(3, 6):
            d_axes[idx].set_ylabel(r'dihedral($\degree$)', fontsize=self.lbfz)
