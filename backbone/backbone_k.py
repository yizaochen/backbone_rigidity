from os import path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from enmspring.na_seq import sequences

class ResidPlotv0:
    resid_lst = list(range(4, 19))
    strand_lst = ['STRAND1', 'STRAND2']
    k_labels = ["C2'(i)-P(i+1)", "C3'(i)-O2P(i+1)", "C2'(i)-O1P(i+1)"]

    d_color = {'STRAND1': 'blue', 'STRAND2': 'red', 'assist_lines': 'grey'}
    xlim = (4, 18)
    d_ylim = {"C2'(i)-P(i+1)": (0.8, 10.3), "C3'(i)-O2P(i+1)": (-0.7, 11.5), "C2'(i)-O1P(i+1)": (-0.7, 9.5)}
    linewidth = 0.5
    markersize = 2

    linewidth_assist_lines = 0.5
    alpha_assist_lines = 0.2
    d_assist_hlines = {"C2'(i)-P(i+1)": np.arange(2, 8.1, 2),  "C3'(i)-O2P(i+1)": np.arange(0, 10.1, 2), 
                       "C2'(i)-O1P(i+1)": np.arange(0, 8.1, 2)}

    tickfz = 4

    def __init__(self, host, dihedral_folder):
        self.host = host
        self.dihedral_folder = dihedral_folder
        self.nrows = self.set_n_rows()

        self.d_df = self.get_d_df()
        self.d_seq = {'STRAND1': sequences[host]['guide'], 'STRAND2': sequences[host]['target']}

    def plot_main(self, figsize, hspace):
        fig = plt.figure(figsize=figsize, facecolor='white')
        d_axes = self.get_d_axes(fig, hspace)
        self.remove_xticks(d_axes)
        self.plot_k(d_axes)
        self.draw_assist_lines(d_axes)
        self.set_xticks(d_axes)
        self.set_xlim(d_axes)
        self.set_yticks_ylim(d_axes)
        self.invert_axis(d_axes)
        self.set_y_tick_params(d_axes)
        return fig, d_axes

    def plot_k(self, d_axes):
        x_array = self.get_k_resid_array()
        for idx in range(self.nrows):
            k_label = self.k_labels[idx]
            for strand_id in self.strand_lst:
                k_mean_array = self.get_k_array(k_label, strand_id)
                d_axes[k_label][strand_id].plot(x_array, k_mean_array, '-o', linewidth=self.linewidth, markersize=self.markersize, color=self.d_color[strand_id])

    def get_k_array(self, k_label, strand_id):
        df_k = self.d_df[strand_id]
        return df_k[k_label]

    def get_k_resid_array(self):
        interval = 0.5
        return np.arange(4+interval, 18, 1)

    def get_d_axes(self, fig, hspace):
        d_axes = dict()
        spec = gridspec.GridSpec(ncols=1, nrows=self.nrows, figure=fig, hspace=hspace)
        for idx in range(self.nrows):
            k_label = self.k_labels[idx]
            d_axes[k_label] = dict()
            d_axes[k_label]['STRAND2'] = fig.add_subplot(spec[idx])
            d_axes[k_label]['STRAND1'] = d_axes[k_label]['STRAND2'].twiny()
        return d_axes
    
    def remove_xticks(self, d_axes):
        for idx in range(self.nrows-1):
            k_label = self.k_labels[idx]
            d_axes[k_label]['STRAND2'].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        for idx in range(1, self.nrows):
            k_label = self.k_labels[idx]
            d_axes[k_label]['STRAND1'].tick_params(axis='x', which='both', bottom=False, top=False, labeltop=False)

    def set_xticks(self, d_axes):
        d_axes_for_ticks = {strand_id: d_axes[self.k_labels[idx]][strand_id] for idx, strand_id in [(0, 'STRAND1'), (2, 'STRAND2')]}
        for strand_id in self.strand_lst:
            ax = d_axes_for_ticks[strand_id]
            ax.set_xticks(self.resid_lst)
            seq = self.d_seq[strand_id]
            xticklabels = [seq[resid-1] for resid in self.resid_lst]
            ax.set_xticklabels(xticklabels)
            ax.tick_params(axis='x', labelsize=self.tickfz, length=1, pad=0.6, color=self.d_color[strand_id], labelcolor=self.d_color[strand_id])

    def set_y_tick_params(self, d_axes):
        for idx in range(self.nrows):
            k_label = self.k_labels[idx]
            ax = d_axes[k_label][self.strand_lst[0]]
            ax.tick_params(axis='y', labelsize=self.tickfz, length=1, pad=1)
            ax = d_axes[k_label][self.strand_lst[1]]
            ax.tick_params(axis='y', labelsize=self.tickfz, length=1, pad=1)


    def set_xlim(self, d_axes):
        for idx in range(self.nrows):
            k_label = self.k_labels[idx]
            for strand_id in self.strand_lst:
                d_axes[k_label][strand_id].set_xlim(self.xlim)

    def set_yticks_ylim(self, d_axes):
        for idx in range(self.nrows):
            k_label = self.k_labels[idx]
            ax = d_axes[k_label][self.strand_lst[0]]
            ax.set_ylim(self.d_ylim[k_label])
            ax.set_yticks(self.d_assist_hlines[k_label])

    def draw_assist_lines(self, d_axes):
        for idx in range(self.nrows):
            k_label = self.k_labels[idx]
            ax = d_axes[k_label][self.strand_lst[0]]
            for resid in self.get_k_resid_array():
                ax.axvline(resid, linestyle='--', linewidth=self.linewidth_assist_lines, color=self.d_color['assist_lines'], alpha=self.alpha_assist_lines)
            for hline in self.d_assist_hlines[k_label]:
                ax.axhline(hline, color=self.d_color['assist_lines'], linewidth=self.linewidth_assist_lines, alpha=self.alpha_assist_lines)

    def invert_axis(self, d_axes):
        for idx in range(self.nrows):
            k_label = self.k_labels[idx]
            d_axes[k_label]['STRAND2'].invert_xaxis()

    def get_d_df(self):
        d_df = dict()
        for strand_id in self.strand_lst:
            f_df = self.set_f_df(strand_id)
            d_df[strand_id] = pd.read_csv(f_df)
        return d_df

    def set_f_df(self, strand_id):
        return path.join(self.dihedral_folder, f'{self.host}_{strand_id}_k_resid_with_next.csv')

    def set_n_rows(self):
        return 3