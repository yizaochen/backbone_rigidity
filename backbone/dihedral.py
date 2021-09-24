from os import path, system
import numpy as np
import MDAnalysis as mda
from backbone.trajectory import TrajectoryProcess
from backbone.na_seq import sequences
from backbone.miscell import check_dir_exist_and_make

class DihedralPair:
    d_pair = {
        "C2prime-P": [('i', "C2'"), ('i', "C3'"), ('i', "O3'"), ('i+1', 'P')],
        'C4prime-P': [('i', "C4'"), ('i', "C3'"), ('i', "O3'"), ('i+1', 'P')],
        'C1prime-N3orO2': {
            'A': [('i', "C1'"), ('i', "N9"), ('i', "C4"), ('i', 'N3')],
            'T': [('i', "C1'"), ('i', "N1"), ('i', "C2"), ('i', 'O2')],
            'G': [('i', "C1'"), ('i', "N9"), ('i', "C4"), ('i', 'N3')],
            'C': [('i', "C1'"), ('i', "N1"), ('i', "C2"), ('i', 'O2')]
        },
        'C1prime-N7orC5': {
            'A': [('i', "C1'"), ('i', "N9"), ('i', "C8"), ('i', 'N7')],
            'T': [('i', "C1'"), ('i', "N1"), ('i', "C6"), ('i', 'C5')],
            'G': [('i', "C1'"), ('i', "N9"), ('i', "C8"), ('i', 'N7')],
            'C': [('i', "C1'"), ('i', "N1"), ('i', "C6"), ('i', 'C5')]
        },
        'O4prime-C4orC2': {
            'A': [('i', "O4'"), ('i', "C1'"), ('i', "N9"), ('i', "C4")],
            'T': [('i', "O4'"), ('i', "C1'"), ('i', "N1"), ('i', "C2")],
            'G': [('i', "O4'"), ('i', "C1'"), ('i', "N9"), ('i', "C4")],
            'C': [('i', "O4'"), ('i', "C1'"), ('i', "N1"), ('i', "C2")]
        },
        'C2prime-C4orC2': {
            'A': [('i', "C2'"), ('i', "C1'"), ('i', "N9"), ('i', "C4")],
            'T': [('i', "C2'"), ('i', "C1'"), ('i', "N1"), ('i', "C2")],
            'G': [('i', "C2'"), ('i', "C1'"), ('i', "N9"), ('i', "C4")],
            'C': [('i', "C2'"), ('i', "C1'"), ('i', "N1"), ('i', "C2")]
        },
        'O4prime-C8orC6': {
            'A': [('i', "O4'"), ('i', "C1'"), ('i', "N9"), ('i', "C8")],
            'T': [('i', "O4'"), ('i', "C1'"), ('i', "N1"), ('i', "C6")],
            'G': [('i', "O4'"), ('i', "C1'"), ('i', "N9"), ('i', "C8")],
            'C': [('i', "O4'"), ('i', "C1'"), ('i', "N1"), ('i', "C6")]
        },
        'C2prime-C8orC6': {
            'A': [('i', "C2'"), ('i', "C1'"), ('i', "N9"), ('i', "C8")],
            'T': [('i', "C2'"), ('i', "C1'"), ('i', "N1"), ('i', "C6")],
            'G': [('i', "C2'"), ('i', "C1'"), ('i', "N9"), ('i', "C8")],
            'C': [('i', "C2'"), ('i', "C1'"), ('i', "N1"), ('i', "C6")]
        }
    }

class DihedralMaker(TrajectoryProcess):
    d_resid_lst = {'STRAND1': list(range(4, 19)), 'STRAND2': list(range(25, 40))}

    def __init__(self, host, strand_id, big_traj_folder, backbone_data_folder):
        super().__init__(host, strand_id, big_traj_folder, backbone_data_folder)

        self.plumed_in_folder = path.join(self.strand_folder, 'plumed_input')
        self.plumed_out_folder = path.join(self.strand_folder, 'plumed_out')
        self.d_atom_container = self.get_d_atom_container()
        self.d_seq = {'STRAND1': sequences[self.host]['guide'], 'STRAND2': sequences[self.host]['target']}

        self.check_folder_in_dihedralmaker()

    def check_folder_in_dihedralmaker(self):
        for folder in [self.plumed_in_folder, self.plumed_out_folder]:
            check_dir_exist_and_make(folder)

    def get_d_atom_container(self):
        d_atom_container = dict()
        u = mda.Universe(self.all_gro)
        indices_array = u.atoms.indices+1
        for resid, atomname, idx in zip(u.atoms.resids, u.atoms.names, indices_array):
            key = (resid, atomname)
            d_atom_container[key] = Atom(resid, atomname, idx)
        return d_atom_container

    def make_all_out(self, dihedral_name):
        resid_lst = self.d_resid_lst[self.strand_id]
        for resid in resid_lst:
            f_input = self.make_plumed_input(resid, dihedral_name)
            cmd = f'plumed driver --plumed {f_input} --mf_xtc {self.all_xtc}'
            print(cmd)
            system(cmd)

    def make_plumed_input(self, resid, dihedral_name):
        f_name = path.join(self.plumed_in_folder, f'{dihedral_name}.{resid}.dat')
        f_out = path.join(self.plumed_out_folder, f'{dihedral_name}.{resid}.out')
        indices_text = self.get_indices_by_resid_dihedral_name(resid, dihedral_name)
        f = open(f_name, 'w')
        f.write(f'phi1: TORSION ATOMS={indices_text}\n')
        f.write(f'PRINT ARG=phi1 FILE={f_out} STRIDE=1\n')
        f.close()
        print(f'Write PLUMED Input: {f_name}')
        return f_name

    def get_indices_by_resid_dihedral_name(self, resid, dihedral_name):
        atom_lst = list()
        for resid_symbol, atomname in DihedralPair.d_pair[dihedral_name]:
            if resid_symbol == 'i':
                atom_lst.append(self.d_atom_container[(resid, atomname)])
            else:
                atom_lst.append(self.d_atom_container[(resid+1, atomname)])
        dihedral_atoms = FourAtoms(atom_lst[0], atom_lst[1], atom_lst[2], atom_lst[3])
        return dihedral_atoms.get_dihedral_indices_text()

    def make_all_out_with_resname(self, dihedral_name):
        resid_lst = self.d_resid_lst[self.strand_id]
        for resid in resid_lst:
            resname = self.get_resname_by_resid(resid)
            f_input = self.make_plumed_input_with_resname(resid, dihedral_name, resname)
            cmd = f'plumed driver --plumed {f_input} --mf_xtc {self.all_xtc}'
            print(cmd)
            system(cmd)

    def make_plumed_input_with_resname(self, resid, dihedral_name, resname):
        f_name = path.join(self.plumed_in_folder, f'{dihedral_name}.{resid}.dat')
        f_out = path.join(self.plumed_out_folder, f'{dihedral_name}.{resid}.out')
        indices_text = self.get_indices_by_resid_dihedral_name_with_resname(resid, dihedral_name, resname)
        f = open(f_name, 'w')
        f.write(f'phi1: TORSION ATOMS={indices_text}\n')
        f.write(f'PRINT ARG=phi1 FILE={f_out} STRIDE=1\n')
        f.close()
        print(f'Write PLUMED Input: {f_name}')
        return f_name

    def get_indices_by_resid_dihedral_name_with_resname(self, resid, dihedral_name, resname):
        atom_lst = list()
        for resid_symbol, atomname in DihedralPair.d_pair[dihedral_name][resname]:
            if resid_symbol == 'i':
                atom_lst.append(self.d_atom_container[(resid, atomname)])
            else:
                atom_lst.append(self.d_atom_container[(resid+1, atomname)])
        dihedral_atoms = FourAtoms(atom_lst[0], atom_lst[1], atom_lst[2], atom_lst[3])
        return dihedral_atoms.get_dihedral_indices_text()

    def get_resname_by_resid(self, resid):
        if self.strand_id == 'STRAND1':
            return self.d_seq[self.strand_id][resid-1]
        else:
            resid_modify = resid - 21
            return self.d_seq[self.strand_id][resid_modify-1]

class DihedralReader(DihedralMaker):
    #dihedral_lst = ["C2prime-P", 'C4prime-P', 'C1prime-N3orO2', 'C1prime-N7orC5']
    dihedral_lst = ["C2prime-P", 'C4prime-P', 'O4prime-C4orC2', 'C2prime-C4orC2']

    def __init__(self, host, strand_id, big_traj_folder, backbone_data_folder):
        super().__init__(host, strand_id, big_traj_folder, backbone_data_folder)

        self.d_result = self.ini_d_result()

    def ini_d_result(self):
        resid_lst = self.d_resid_lst[self.strand_id]
        d_result = dict()
        for dihedral_name in self.dihedral_lst:
            d_result[dihedral_name] = dict()
            for resid in resid_lst:
                d_result[dihedral_name][resid] = {'time': None, 'dihedral': None}
        return d_result

    def get_d_time_dihedral_by_resid(self, resid, dihedral_name):
        if self.d_result[dihedral_name][resid]['time'] is None:
            self.read_dihedral_by_resid(resid, dihedral_name)
        return self.d_result[dihedral_name][resid]

    def read_dihedral_by_resid(self, resid, dihedral_name):
        f_in = path.join(self.plumed_out_folder, f'{dihedral_name}.{resid}.out')
        data = np.genfromtxt(f_in, skip_header=3)
        self.d_result[dihedral_name][resid]['time'] = data[:,0]
        self.d_result[dihedral_name][resid]['dihedral'] = data[:,1]

class Atom:
    def __init__(self, resid, name, atom_id):
        self.resid = resid
        self.name = name
        self.atom_id = atom_id

    def get_atom_id(self):
        return self.atom_id

    def __repr__(self):
        return f'{self.atom_id}-{self.resid}-{self.name}'

class FourAtoms:
    def __init__(self, atom1, atom2, atom3, atom4):
        self.atom1 = atom1
        self.atom2 = atom2
        self.atom3 = atom3
        self.atom4 = atom4

    def get_dihedral_indices_text(self):
        return f'{self.atom1.get_atom_id()},{self.atom2.get_atom_id()},{self.atom3.get_atom_id()},{self.atom4.get_atom_id()}'

    