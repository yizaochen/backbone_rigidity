from os import path, system
from backbone.strand import Strand
from backbone.miscell import check_dir_exist_and_make

class TrajectoryProcess:
    type_na = 'bdna+bdna'
    gmx_path = '/usr/bin/gmx'
    ref_pdb = '/home/yizaochen/codes/backbone_rigidity/ref_structures/backbone.pdb'

    def __init__(self, host, strand_id, big_traj_folder, backbone_data_folder):
        self.host = host
        self.strand_id = strand_id
        self.big_traj_folder = big_traj_folder
        self.backbone_data_folder = backbone_data_folder

        self.all_folder = path.join('/home/yizaochen/codes/dna_rna/all_systems', self.host, self.type_na, 'input', 'allatoms')
        self.all_gro = path.join(self.all_folder, f'{self.type_na}.npt4.all.gro')
        self.all_pdb = path.join(self.all_folder, f'{self.type_na}.npt4.all.pdb')
        self.all_xtc = path.join(self.all_folder, f'{self.type_na}.all.xtc')

        self.host_folder = path.join(self.backbone_data_folder, self.host)
        self.strand_folder = path.join(self.host_folder, self.strand_id)
        self.ndx_folder = path.join(self.strand_folder, 'ndx')

        self.f_strand_ndx = path.join(self.ndx_folder, f'{self.strand_id}.ndx')
        self.pdb = path.join(self.strand_folder, f'{self.strand_id}.pdb')
        self.xtc = path.join(self.strand_folder, f'{self.strand_id}.xtc')

        self.fit_pdb = path.join(self.strand_folder, f'{self.strand_id}.fit.pdb')
        self.fit_xtc = path.join(self.strand_folder, f'{self.strand_id}.fit.xtc')

        self.check_folder()

    def check_folder(self):
        for folder in [self.host_folder, self.strand_folder, self.ndx_folder]:
            check_dir_exist_and_make(folder)

    def make_ndx_file(self, ref_gro, ndx_txt, f_ndx):
        temp_txt = path.join(self.ndx_folder, 'temp.txt')
        f = open(temp_txt, 'w')
        f.write(ndx_txt)
        f.write('\n')
        f.write('q\n')
        f.close()

        cmd = f'{self.gmx_path} make_ndx -f {ref_gro} -o {f_ndx} < {temp_txt}'
        self.execute_cmd(cmd)

    def make_strand_central_pdb(self):
        strand_agent = Strand(self.host, self.strand_id)
        ndx_txt = strand_agent.get_gmx_makendx_text()
        self.make_ndx_file(self.all_gro, ndx_txt, self.f_strand_ndx)
        cmd = f'echo 2 | {self.gmx_path} editconf -f {self.all_gro} -o {self.pdb} -n {self.f_strand_ndx}'
        self.execute_cmd(cmd)

    def make_strand_central_xtc(self):
        cmd = f'echo 2 | {self.gmx_path} trjconv -s {self.pdb} -f {self.all_xtc} -o {self.xtc} -n {self.f_strand_ndx}'
        self.execute_cmd(cmd)

    def make_fit_pdb(self):
        cmd = f'echo 0 0 | {self.gmx_path} confrms -f1 {self.ref_pdb} -f2 {self.pdb} -o {self.fit_pdb} -one'
        self.execute_cmd(cmd)

    def make_fit_xtc(self):
        cmd = f'echo 0 0 | {self.gmx_path} trjconv -s {self.ref_pdb} -f {self.xtc} -o {self.fit_xtc} -fit rot+trans'
        self.execute_cmd(cmd)

    def check_pdb(self):
        cmd = f'vmd -pdb {self.pdb}'
        print(cmd)

    def check_xtc(self):
        cmd = f'vmd -pdb {self.pdb} {self.xtc}'
        print(cmd)

    def check_fit_pdb(self):
        cmd = f'vmd -pdb {self.fit_pdb}'
        print(cmd)
        cmd = f'mol new {self.ref_pdb} type pdb'
        print(cmd)

    def check_fit_xtc(self):
        cmd = f'vmd -pdb {self.fit_pdb} {self.fit_xtc}'
        print(cmd)

    def check_traj(self):
        cmd = f'vmd -gro {self.all_gro} {self.all_xtc}'
        print(cmd)

    def execute_cmd(self, cmd):
        print(cmd)
        system(cmd)