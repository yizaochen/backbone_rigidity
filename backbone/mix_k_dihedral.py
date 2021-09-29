from os import path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colorbar import ColorbarBase
from matplotlib import cm
from enmspring.graphs_bigtraj import BackboneMeanModeAgent
from enmspring.kappa_mat import KMat
from enmspring.backbone_k import BackboneResidPlot, BackboneResidPlotWithNext
from backbone.dihedral import DihedralReader
from backbone.na_seq import sequences

class DPair:
    d_pairs = {
    "C2'(i)-P(i+1)": {
        'A': {'atomname_i': "C2'", 'atomname_j': "P"},
        'T': {'atomname_i': "C2'", 'atomname_j': "P"},
        'G': {'atomname_i': "C2'", 'atomname_j': "P"},
        'C': {'atomname_i': "C2'", 'atomname_j': "P"}
        },
    "C2'(i)-O1P(i+1)": {
        'A': {'atomname_i': "C2'", 'atomname_j': "O1P"},
        'T': {'atomname_i': "C2'", 'atomname_j': "O1P"},
        'G': {'atomname_i': "C2'", 'atomname_j': "O1P"},
        'C': {'atomname_i': "C2'", 'atomname_j': "O1P"}
        },
    "C3'(i)-O2P(i+1)": {
        'A': {'atomname_i': "C3'", 'atomname_j': "O2P"},
        'T': {'atomname_i': "C3'", 'atomname_j': "O2P"},
        'G': {'atomname_i': "C3'", 'atomname_j': "O2P"},
        'C': {'atomname_i': "C3'", 'atomname_j': "O2P"}
        },
    "C4'(i)-O5'(i+1)": {
        'A': {'atomname_i': "C4'", 'atomname_j': "O5'"},
        'T': {'atomname_i': "C4'", 'atomname_j': "O5'"},
        'G': {'atomname_i': "C4'", 'atomname_j': "O5'"},
        'C': {'atomname_i': "C4'", 'atomname_j': "O5'"}
        },
    "C1'-N3/C1'-O2": {
        'A': {'atomname_i': "C1'", 'atomname_j': "N3"},
        'T': {'atomname_i': "C1'", 'atomname_j': "O2"},
        'G': {'atomname_i': "C1'", 'atomname_j': "N3"},
        'C': {'atomname_i': "C1'", 'atomname_j': "O2"}
    }, 
    "C2'-C8/C2'-C6": {
        'A': {'atomname_i': "C2'", 'atomname_j': "C8"},
        'T': {'atomname_i': "C2'", 'atomname_j': "C6"},
        'G': {'atomname_i': "C2'", 'atomname_j': "C8"},
        'C': {'atomname_i': "C2'", 'atomname_j': "C6"}
    },
    "O4'-O5'": {
        'A': {'atomname_i': "O4'", 'atomname_j': "O5'"},
        'T': {'atomname_i': "O4'", 'atomname_j': "O5'"},
        'G': {'atomname_i': "O4'", 'atomname_j': "O5'"},
        'C': {'atomname_i': "O4'", 'atomname_j': "O5'"}
    }
    }

class MixPlot1:
    interval_time = 500
    resid_lst = list(range(4, 19))
    lbfz = 12
    lgfz = 12
    k_labels = ["C2'(i)-P(i+1)", "C3'(i)-O2P(i+1)", "C2'(i)-O1P(i+1)"]
    dihedral_ylabels = ["C2'-C3'-O3'-P", r"$\epsilon$", r"$\zeta$"]
    dihedral_name_lst = ["C2prime-P", "C4prime-P", "C3prime-O5prime"]

    def __init__(self, host, strand_id, big_traj_folder, dihedral_folder, backbone_data_folder, make_df=False):
        self.host = host
        self.strand_id = strand_id
        self.big_traj_folder = big_traj_folder
        self.dihedral_folder = dihedral_folder
        self.backbone_data_folder = backbone_data_folder
        self.make_df = make_df
        self.nrows = self.set_n_rows()
        self.n_resid = len(self.resid_lst)

        self.s_agent = None
        self.kmat_agent = None
        self.resid_k_agent = None
        
        if self.make_df:
            self.ini_k_agents()

        if self.make_df:
            self.start_mode = 1
            self.end_mode = self.s_agent.n_node
        
        self.big_k_mat = None
        if self.make_df:
            self.ini_big_k_mat()

        self.f_df = self.set_f_df()
        self.df_k = None

        self.d_reader = None
        self.set_d_reader()

        self.d_dihedral_df = None

        self.epsilon_minus_zeta_df = None
        self.d_seq = {'STRAND1': sequences[host]['guide'], 'STRAND2': sequences[host]['target']}

    def plot_main(self, figsize, hspace, bottom, top):
        fig = plt.figure(figsize=figsize, facecolor='white')
        d_axes = self.get_d_axes(fig, hspace)
        self.remove_xticks(d_axes)
        self.set_xlabel_xticks(d_axes)
        self.set_ylabels(d_axes)
        for idx in range(3):
            self.plot_k(d_axes, idx)
            self.set_k_ylim(d_axes, idx)
        for idx in range(3, 6):
            self.heatmap(d_axes, idx, bottom, top)
            self.set_yticks_for_dihedral(d_axes[idx], top, bottom)
        self.add_resid_lines(d_axes)
        self.set_xlim_the_same(d_axes)
        return fig, d_axes

    def plot_k(self, d_axes, idx):
        label = self.k_labels[idx]
        k_mean_array = self.get_k_array(label)
        x_array = self.get_k_resid_array()
        d_axes[idx].plot(x_array, k_mean_array, '-o', label=label)
        d_axes[idx].legend(fontsize=self.lgfz, frameon=False)

    def heatmap(self, d_axes, idx, bottom, top):
        dihedral_name = self.dihedral_name_lst[idx-3]
        data_mat = self.assemble_data_mat(self.d_dihedral_df[dihedral_name])
        norm, cmap = self.get_norm_cmap(dihedral_name)
        d_axes[idx].imshow(data_mat, norm=norm, cmap=cmap, origin='lower', extent=self.get_extent(bottom, top))
        d_axes[idx].set_yticks([bottom, top])

    def get_norm_cmap(self, dihedral_name):
        # "C2prime-P", "C4prime-P", "C3prime-O5prime"
        min_max = {"C2prime-P": (0, 0.035), "C4prime-P": (0, 0.035), "C3prime-O5prime": (0, 0.035),
                   "epsilon-zeta": (0, 0.025), "O4prime-O5prime": (0, 0.042), "C2prime-C8orC6": (0, 0.039), 
                   "O4prime-C8orC6": (0, 0.036)}
        CMAP = LinearSegmentedColormap.from_list('mycmap', ['white','red'])
        norm = Normalize(vmin=min_max[dihedral_name][0], vmax=min_max[dihedral_name][1])
        cmap = cm.get_cmap(CMAP)
        return norm, cmap

    def draw_color_bar(self, figsize, dihedral_name):
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
        norm, cmap = self.get_norm_cmap(dihedral_name)
        cb1 = ColorbarBase(ax1, cmap=cmap, norm=norm, orientation='horizontal')
        return fig, ax1, cb1

    def set_n_rows(self):
        return 6

    def set_f_df(self):
        return path.join(self.dihedral_folder, f'{self.host}_{self.strand_id}_k_resid_with_next.csv')

    def set_k_ylim(self, d_axes, idx):
        d_ylim = {"C2'(i)-P(i+1)": (1.285, 9.921), "C3'(i)-O2P(i+1)": (0, 11.2), "C2'(i)-O1P(i+1)": (0, 9.223), "C4'(i)-O5'(i+1)": (0.634, 6.763)}
        label = self.k_labels[idx]
        d_axes[idx].set_ylim(d_ylim[label])
        d_hlines = {"C2'(i)-P(i+1)": np.arange(2, 8.1, 2),  "C3'(i)-O2P(i+1)": np.arange(2.5, 10.1, 2.5), "C2'(i)-O1P(i+1)": np.arange(2, 8.1, 2), 
                    "C4'(i)-O5'(i+1)": np.arange(2, 6.1, 2)}
        for hline in d_hlines[label]:
            d_axes[idx].axhline(hline, color='grey', alpha=0.1)

    def add_resid_lines(self, d_axes):
        for idx in range(self.nrows):
            for resid in self.resid_lst:
                d_axes[idx].axvline(resid, color='grey', alpha=0.3, linestyle='--')
    
    def set_xlim_the_same(self, d_axes):
        xlim = d_axes[3].get_xlim()
        for idx in range(self.nrows):
            d_axes[idx].set_xlim(xlim)

    def make_k_df(self):
        d_result = dict()
        for label in self.k_labels:
            d_pair = DPair.d_pairs[label]
            d_result[label] = self.resid_k_agent.get_yarray(self.strand_id, self.big_k_mat, d_pair)
        df = pd.DataFrame(d_result)
        df.to_csv(self.f_df, index=False)
        self.df_k = pd.read_csv(self.f_df)

    def read_k_df(self):
        self.df_k = pd.read_csv(self.f_df)

    def make_all_dihedral_df(self):
        self.d_dihedral_df = dict()
        for dihedral_name in self.dihedral_name_lst:
            self.make_dihedral_df(dihedral_name)
            self.d_dihedral_df[dihedral_name] = self.read_dihedral_df(dihedral_name)

    def read_all_diehdral_df(self):
        self.d_dihedral_df = dict()
        for dihedral_name in self.dihedral_name_lst:
            self.d_dihedral_df[dihedral_name] = self.read_dihedral_df(dihedral_name)

    def make_epsilon_minus_zeta_df(self):
        d_result = dict()
        d_resid_lst = {'STRAND1': list(range(4, 19)), 'STRAND2': list(range(25, 40))}
        for resid in d_resid_lst[self.strand_id]:
            epsilon_array = self.d_reader.get_d_time_dihedral_by_resid(resid, 'C4prime-P')['dihedral']
            zeta_array = self.d_reader.get_d_time_dihedral_by_resid(resid, 'C3prime-O5prime')['dihedral']
            dihedral_array = np.rad2deg(epsilon_array - zeta_array)
            dihedral_array = [self.normalize_epsilon_minus_zeta(dihedral) for dihedral in dihedral_array]
            hist_result = np.histogram(dihedral_array, bins=self.get_epsilon_minus_zeta_for_bin(), density=True)
            d_result[resid] = hist_result[0]
        df = pd.DataFrame(d_result)
        f_dihedral = path.join(self.dihedral_folder, f'{self.host}_{self.strand_id}_epsilon_minus_zeta_prob.csv')
        df.to_csv(f_dihedral, index=False)
        self.epsilon_minus_zeta_df = pd.read_csv(f_dihedral)

    def read_epsilon_minus_zeta_df(self):
        f_dihedral = path.join(self.dihedral_folder, f'{self.host}_{self.strand_id}_epsilon_minus_zeta_prob.csv')
        self.epsilon_minus_zeta_df = pd.read_csv(f_dihedral)

    def normalize_epsilon_minus_zeta(self, value):
        if value > 200:
            return value - 360
        elif value < -160:
            return value + 360
        else:
            return value

    def make_dihedral_df(self, dihedral_name):
        d_result = dict()
        d_resid_lst = {'STRAND1': list(range(4, 19)), 'STRAND2': list(range(25, 40))}
        for resid in d_resid_lst[self.strand_id]:
            dihedral_array = self.d_reader.get_d_time_dihedral_by_resid(resid, dihedral_name)['dihedral']
            #dihedral_array = np.rad2deg(dihedral_array) + 180
            dihedral_array = np.rad2deg(dihedral_array)
            hist_result = np.histogram(dihedral_array, bins=self.get_degree_array_for_bin(), density=True)
            d_result[resid] = hist_result[0]
        df = pd.DataFrame(d_result)
        f_dihedral = path.join(self.dihedral_folder, f'{self.host}_{self.strand_id}_{dihedral_name}_prob.csv')
        df.to_csv(f_dihedral, index=False)
    
    def read_dihedral_df(self, dihedral_name):
        f_dihedral = path.join(self.dihedral_folder, f'{self.host}_{self.strand_id}_{dihedral_name}_prob.csv')
        return pd.read_csv(f_dihedral)

    def get_degree_array_for_bin(self):
        return np.arange(-180, 179.1, 1)
    
    def get_degree_array_for_plot(self):
        return np.arange(-179.5, 179, 1)

    def get_epsilon_minus_zeta_for_bin(self):
        return np.arange(-160, 199.1, 1)

    def get_epsilon_minus_zeta_for_plot(self):
        return np.arange(-159.5, 199, 1)

    def get_k_resid_array(self):
        interval = 0.5
        return np.arange(4+interval, 18, 1)

    def get_k_array(self, label):
        return self.df_k[label]

    def assemble_data_mat(self, df):
        d_resid_lst = {'STRAND1': list(range(4, 19)), 'STRAND2': list(range(25, 40))}
        resid_lst = d_resid_lst[self.strand_id]
        degree_array = self.get_degree_array_for_plot()
        data_mat = np.zeros((len(degree_array), len(resid_lst)))
        for resid_idx, resid in enumerate(d_resid_lst[self.strand_id]):
            data_mat[:,resid_idx] = df[f'{resid}']
        return data_mat

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
        seq = self.d_seq[self.strand_id]
        xticklabels = [seq[resid-1] for resid in self.resid_lst]
        ax.set_xticklabels(xticklabels)

    def set_ylabels(self, d_axes):
        for idx in range(3):
            d_axes[idx].set_ylabel('k (kcal/mol/Å$^2$)', fontsize=self.lbfz)
        for idx in range(3, 6):
            ylabel = self.dihedral_ylabels[idx-3] + r' ($\degree$)'
            d_axes[idx].set_ylabel(ylabel, fontsize=self.lbfz)

    def set_yticks_for_dihedral(self, ax, top, bottom):
        yticklabels = np.arange(-180, 179.1, 45)
        yticklabels = [f'{label:.0f}' for label in yticklabels]
        n_yticklabels = len(yticklabels)
        interval = (top - bottom) / n_yticklabels
        yticks = np.arange(bottom, top, interval)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)

    def ini_k_agents(self):
        if self.s_agent is None:
            self.s_agent = BackboneMeanModeAgent(self.host, self.big_traj_folder, self.interval_time)
        if self.kmat_agent is None:
            self.kmat_agent = KMat(self.s_agent)
        if self.resid_k_agent is None:
            self.resid_k_agent = BackboneResidPlotWithNext(self.host, self.s_agent, self.kmat_agent)

    def ini_big_k_mat(self):
        if self.big_k_mat is None:
            self.big_k_mat = self.kmat_agent.get_K_mat(self.start_mode, self.end_mode)

    def set_d_reader(self):
        if self.d_reader is None:
            self.d_reader = DihedralReader(self.host, self.strand_id, self.big_traj_folder, self.backbone_data_folder)


class MixPlot2(MixPlot1):
    dihedral_ylabels = ["C2'-C3'-O3'-P", r"$\epsilon-\zeta$"]

    def plot_main(self, figsize, hspace, bottom, top):
        fig = plt.figure(figsize=figsize, facecolor='white')
        d_axes = self.get_d_axes(fig, hspace)
        self.remove_xticks(d_axes)
        self.set_xlabel_xticks(d_axes)
        self.set_ylabels(d_axes)
        for idx in range(3):
            self.plot_k(d_axes, idx)
            self.set_k_ylim(d_axes, idx)
        idx += 1
        self.heatmap(d_axes, idx, bottom, top) # C2'-C3'-O3'-P
        self.set_yticks_for_dihedral(d_axes[idx], top, bottom)
        idx += 1
        self.heatmap_epsilon_minus_zeta(d_axes, idx, bottom, top)
        self.set_yticks_for_epsilon_minus_zeta(d_axes[idx], top, bottom)
        self.add_resid_lines(d_axes)
        self.set_xlim_the_same(d_axes)
        return fig, d_axes

    def heatmap_epsilon_minus_zeta(self, d_axes, idx, bottom, top):
        data_mat = self.assemble_data_mat(self.epsilon_minus_zeta_df)
        norm, cmap = self.get_norm_cmap("epsilon-zeta")
        d_axes[idx].imshow(data_mat, norm=norm, cmap=cmap, origin='lower', extent=self.get_extent(bottom, top))
        d_axes[idx].set_yticks([bottom, top])
        
    def set_yticks_for_epsilon_minus_zeta(self, ax, top, bottom):
        yticklabels = list(range(-160, 141, 60))
        yticklabels = [f'{label:.0f}' for label in yticklabels]
        n_yticklabels = len(yticklabels)
        interval = (top - bottom) / n_yticklabels
        yticks = np.arange(bottom, top, interval)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.axhline(yticks[3], color='green', alpha=0.3, label="BI: <20, BII: >20")
        ax.legend(fontsize=self.lgfz, frameon=False)

    def set_n_rows(self):
        return 5

    def set_ylabels(self, d_axes):
        for idx in range(3):
            d_axes[idx].set_ylabel('k (kcal/mol/Å$^2$)', fontsize=self.lbfz)
        for idx in range(3, 5):
            ylabel = self.dihedral_ylabels[idx-3] + r' ($\degree$)'
            d_axes[idx].set_ylabel(ylabel, fontsize=self.lbfz)

    def get_data_mat_min_max(self):
        data_mat = self.assemble_data_mat(self.d_dihedral_df["C2prime-P"])
        print("C2'-C3'-O3'-P")
        print(f'Min: {data_mat.min()} Max: {data_mat.max():.3f}')
        data_mat = self.assemble_data_mat(self.epsilon_minus_zeta_df)
        print('epsilon - zeta:')
        print(f'Min: {data_mat.min()} Max: {data_mat.max():.3f}')


class MixPlot3(MixPlot1):
    lbfz = 10
    k_labels = ["C1'-N3/C1'-O2", "C2'-C8/C2'-C6", "O4'-O5'"]
    dihedral_ylabels = ["O4'-C4'-C5'-O5'", "C2'-C1'-N9/N1-C8/C6", "O4'-C1'-N9/N1-C8/C6"]
    dihedral_name_lst = ["O4prime-O5prime", "C2prime-C8orC6", "O4prime-C8orC6"]

    def __init__(self, host, strand_id, big_traj_folder, dihedral_folder, backbone_data_folder, make_df=False):
        super().__init__(host, strand_id, big_traj_folder, dihedral_folder, backbone_data_folder, make_df=make_df)
        self.f_df = path.join(self.dihedral_folder, f'{self.host}_{self.strand_id}_k_resid_with_self.csv')

    def plot_main(self, figsize, hspace, bottom, top):
        fig = plt.figure(figsize=figsize, facecolor='white')
        d_axes = self.get_d_axes(fig, hspace)
        self.remove_xticks(d_axes)
        self.set_xlabel_xticks(d_axes)
        self.set_ylabels(d_axes)
        for idx in range(3):
            self.plot_k(d_axes, idx)
            self.set_k_ylim(d_axes, idx)
        for idx in range(3, 6):
            self.heatmap(d_axes, idx, bottom, top)
            self.set_yticks_for_dihedral(d_axes[idx], top, bottom)
        self.add_resid_lines(d_axes)
        self.set_xlim_the_same(d_axes)
        return fig, d_axes

    def ini_k_agents(self):
        if self.s_agent is None:
            self.s_agent = BackboneMeanModeAgent(self.host, self.big_traj_folder, self.interval_time)
        if self.kmat_agent is None:
            self.kmat_agent = KMat(self.s_agent)
        if self.resid_k_agent is None:
            self.resid_k_agent = BackboneResidPlot(self.host, self.s_agent, self.kmat_agent)

    def set_f_df(self):
        return path.join(self.dihedral_folder, f'{self.host}_{self.strand_id}_k_resid_with_self.csv')

    def get_k_resid_array(self):
        return self.resid_lst

    def set_k_ylim(self, d_axes, idx):
        d_ylim = {"C1'-N3/C1'-O2": (10.947, 30.006), "C2'-C8/C2'-C6": (5.255, 15.881), 
                  "O4'-O5'": (3.188, 11.171)}
        label = self.k_labels[idx]
        d_axes[idx].set_ylim(d_ylim[label])
        d_hlines = {"C1'-N3/C1'-O2": np.arange(15, 30.1, 5), "C2'-C8/C2'-C6": np.arange(6, 14.1, 2), 
                    "O4'-O5'": np.arange(4, 10.1, 2)}
        for hline in d_hlines[label]:
            d_axes[idx].axhline(hline, color='grey', alpha=0.1)

