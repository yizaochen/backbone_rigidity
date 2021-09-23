import matplotlib.pyplot as plt
import numpy as np
from backbone.dihedral import DihedralReader

class FourDihedralHistogram:
    strand_lst = ['STRAND1', 'STRAND2']
    d_resid_lst = {'STRAND1': list(range(4, 19)), 'STRAND2': list(range(25, 40))}
    #dihedral_lst = ["C2prime-P", 'C4prime-P', 'C1prime-N3orO2', 'C1prime-N7orC5']
    dihedral_lst = ["C2prime-P", 'C4prime-P', 'O4prime-C4orC2', 'C2prime-C4orC2']
    d_titles = {"C2prime-P": "C2'-P", 'C4prime-P': "C4'-P", 
                'C1prime-N3orO2': "C1'-N3/C1'-O2", 'C1prime-N7orC5': "C1'-N7/C1'-C5",
                'O4prime-C4orC2': "O4'-C4/O4'-C2", 'C2prime-C4orC2': "C2'-C4/C2'-C2"}
    lbfz = 12
    ttfz = 16
    lgfz = 12
    n_frames = 50001

    def __init__(self, host, big_traj_folder, backbone_data_folder):
        self.host = host
        self.big_traj_folder = big_traj_folder
        self.backbone_data_folder = backbone_data_folder

        self.d_reader = self.get_d_reader()
        self.d_container = self.ini_d_container()

    def histogram(self, figsize, bins):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize, facecolor='white')
        d_axes = self.get_d_axes(axes)
        for dihedral in self.dihedral_lst:
            self.do_histogram_by_dihedral_name(d_axes[dihedral], dihedral, bins)
            self.set_title(d_axes[dihedral], dihedral)
            self.set_xlabel_ylabel(d_axes[dihedral])
            self.set_xticks(d_axes[dihedral])
            self.set_yticks(d_axes[dihedral])
        axes[0,0].legend(fontsize=self.lgfz, frameon=False)
        return fig, d_axes

    def do_histogram_by_dihedral_name(self, ax, dihedral_name, bins):
        for strand_id in self.strand_lst:
            if self.d_container[dihedral_name][strand_id] is None:
                self.get_container(dihedral_name, strand_id)
            ax.hist(self.d_container[dihedral_name][strand_id], bins=bins, density=True, alpha=0.4, label=strand_id)

    def set_title(self, ax, dihedral_name):
        ax.set_title(self.d_titles[dihedral_name], fontsize=self.ttfz)

    def set_xlabel_ylabel(self, ax):
        ax.set_xlabel("Dihedral Angle (degree)", fontsize=self.lbfz)
        ax.set_ylabel("P", fontsize=self.lbfz)

    def set_xticks(self, ax):
        xticks = np.arange(-180, 180.1, 45)
        ax.set_xticks(xticks)
        ax.set_xlim(-180, 180)
        for xtick in xticks:
            ax.axvline(xtick, color='grey', alpha=0.1)

    def set_yticks(self, ax):
        yticks = np.arange(0, 0.031, 0.005)
        ax.set_yticks(yticks)
        ax.set_ylim(0, 0.031)
        for ytick in yticks:
            ax.axhline(ytick, color='grey', alpha=0.1)

    def get_d_axes(self, axes):
        d_axes = dict()
        idx = 0
        for row_id in range(2):
            for col_id in range(2):
                d_axes[self.dihedral_lst[idx]] = axes[row_id, col_id]
                idx += 1
        return d_axes

    def get_d_reader(self):
        d_reader = dict()
        for strand_id in self.strand_lst:
            d_reader[strand_id] = DihedralReader(self.host, strand_id, self.big_traj_folder, self.backbone_data_folder)
        return d_reader

    def ini_d_container(self):
        d_container = dict()
        for dihedral in self.dihedral_lst:
            d_container[dihedral] = {strand_id:None for strand_id in self.strand_lst}
        return d_container

    def get_container(self, dihedral, strand_id):
        resid_lst = self.d_resid_lst[strand_id]
        container = np.zeros((len(resid_lst), self.n_frames))
        for idx, resid in enumerate(resid_lst):
            container[idx,:] = self.d_reader[strand_id].get_d_time_dihedral_by_resid(resid, dihedral)['dihedral']
        self.d_container[dihedral][strand_id] = np.rad2deg(container.flatten())