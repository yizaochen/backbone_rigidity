from os import path, system
import numpy as np
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

class RMSFPlot(RMSFProcess):
    def __init__(self, host, strand_id, big_traj_folder, backbone_data_folder):
        super().__init__(host, strand_id, big_traj_folder, backbone_data_folder)
