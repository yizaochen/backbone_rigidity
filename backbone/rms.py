from os import path, system
import numpy as np
import matplotlib.pyplot as plt
from backbone.trajectory import TrajectoryProcess

class RMSFProcess(TrajectoryProcess):

    def __init__(self, host, strand_id, big_traj_folder, backbone_data_folder):
        super().__init__(host, strand_id, big_traj_folder, backbone_data_folder)
        self.rmsf_res_xvg = path.join(self.strand_folder, f'{self.strand_id}.res.xvg')
        self.rmsf_aa_xvg = path.join(self.strand_folder, f'{self.strand_id}.aa.xvg')

    def calc_rmsf_res(self):
        cmd = f'echo 0 |{self.gmx_path} rmsf -s {self.fit_pdb} -f {self.fit_xtc} -o {self.rmsf_res_xvg} -res'
        self.execute_cmd(cmd)

    def calc_rmsf_aa(self):
        cmd = f'echo 0 |{self.gmx_path} rmsf -s {self.fit_pdb} -f {self.fit_xtc} -o {self.rmsf_aa_xvg}'
        self.execute_cmd(cmd)

    def check_rmsf_res_xvg(self):
        cmd = f'vim {self.rmsf_res_xvg}'
        print(cmd)

    def check_rmsf_aa_xvg(self):
        cmd = f'vim {self.rmsf_aa_xvg}'
        print(cmd)

class RMSFReader(RMSFProcess):
    def __init__(self, host, strand_id, big_traj_folder, backbone_data_folder):
        super().__init__(host, strand_id, big_traj_folder, backbone_data_folder)
        self.d_rmsf_res = self.read_rmsf_res()
        self.d_rmsf_aa = self.read_rmsf_aa()

    def read_rmsf_res(self):
        data = np.genfromtxt(self.rmsf_res_xvg, skip_header=17)
        d_result = dict()
        d_result['resid'] = data[:,0]
        d_result['rmsf'] = data[:,1] * 10
        return d_result

    def read_rmsf_aa(self):
        data = np.genfromtxt(self.rmsf_res_xvg, skip_header=17)
        d_result = dict()
        d_result['atomid'] = data[:,0]
        d_result['rmsf'] = data[:,1] * 10
        return d_result

class RMSFResPlot:
    strand_lst = ['STRAND1', 'STRAND2']
    lbfz = 12

    def __init__(self, host, big_traj_folder, backbone_data_folder):
        self.host = host
        self.big_traj_folder = big_traj_folder
        self.backbone_data_folder = backbone_data_folder

        self.d_reader = self.get_d_reader()

    def plot_main(self, figsize, inverse, ylim=None, assisthlines=None):
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = ax1.twiny()
        self.plot_strand1(ax2)
        self.plot_strand2(ax1, inverse)
        self.set_xlabel_ylabel(ax1, ax2)
        if ylim is not None:
            ax1.set_ylim(ylim)
        if assisthlines is not None:
            for hline in assisthlines:
                ax1.axhline(hline, color='grey', alpha=0.2)
        return fig, ax1, ax2

    def plot_strand1(self, ax):
        d_rmsf_res = self.d_reader['STRAND1'].d_rmsf_res
        ax.plot(d_rmsf_res['resid'], d_rmsf_res['rmsf'], '-o', color='blue')
        ax.set_xticks(d_rmsf_res['resid'])

    def plot_strand2(self, ax, inverse):
        d_rmsf_res = self.d_reader['STRAND2'].d_rmsf_res
        d_rmsf_res['resid'] = d_rmsf_res['resid'] - 21 # get compensate resid
        ax.plot(d_rmsf_res['resid'], d_rmsf_res['rmsf'], '-o', color='red')
        ax.set_xticks(d_rmsf_res['resid'])
        if inverse:
            ax.invert_xaxis()

    def set_xlabel_ylabel(self, ax1, ax2):
        ax1.set_ylabel('RMSF(Å)', fontsize=self.lbfz)
        ax1.set_xlabel('STRAND2 Resid', fontsize=self.lbfz, color='red')
        ax2.set_xlabel('STRAND1 Resid', fontsize=self.lbfz, color='blue')
        ax1.tick_params(axis='x', color='red', labelcolor='red')
        ax2.tick_params(axis='x', color='blue', labelcolor='blue')

    def get_d_reader(self):
        d_reader = dict()
        for strand_id in self.strand_lst:
            d_reader[strand_id] = RMSFReader(self.host, strand_id, self.big_traj_folder, self.backbone_data_folder)
        return d_reader

