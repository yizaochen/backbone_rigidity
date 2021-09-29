from os import path
import MDAnalysis as mda
from backbone.na_seq import sequences
from backbone.dihedral import DihedralPair

class DihedralDrawer:
    type_na = 'bdna+bdna'

    def __init__(self, host, strand_id, all_folder, tcl_folder):
        self.host = host
        self.strand_id = strand_id
        self.all_folder = all_folder
        self.tcl_folder = tcl_folder

        self.input_folder = path.join(self.all_folder, host, self.type_na, 'input', 'allatoms')
        self.perfect_pdb = path.join(self.input_folder, f'{self.type_na}.perfect.pdb')
        self.u = mda.Universe(self.perfect_pdb)

        self.resid_i = self.get_resid_i()
        self.resid_j = self.resid_i + 1
        self.d_resid_ij = {'i': self.resid_i, 'i+1': self.resid_j}

        self.d_seq = {'STRAND1': sequences[self.host]['guide'], 'STRAND2': sequences[self.host]['target']}

        self.resname_i = self.get_resname_by_resid(self.resid_i)

    def show_vmd(self):
        print(f'vmd -pdb {self.perfect_pdb}')

    def show_system(self, dihedral_name):
        txt_lst_1 = ['mol color Name', 'mol representation Licorice 0.100000 12.000000 12.000000', 
                     'mol selection resid 5 6 and not hydrogen', 'mol material Transparent', 'mol addrep 0']
        txt_lst_2 = ['mol representation CPK 1.000000 0.300000 12.000000 12.000000', 
                     self.get_selection_txt_by_dihedral_name(dihedral_name), 
                     'mol material AOChalky', 'mol addrep 0']
        f_tcl = path.join(self.tcl_folder, 'show_system.tcl')
        self.write_tcl(f_tcl, txt_lst_1+txt_lst_2)

    def draw_two_triangles(self, dihedral_name, material):
        txt_lst = list()
        if dihedral_name in ["C2prime-P", 'C4prime-P', "O4prime-O5prime"]:
            txt_lst += self.get_triangle_txt_lst('magenta', DihedralPair.d_pair[dihedral_name][:3], material)
            txt_lst += self.get_triangle_txt_lst('orange', DihedralPair.d_pair[dihedral_name][1:], material)
        else:
            txt_lst += self.get_triangle_txt_lst('magenta', DihedralPair.d_pair[dihedral_name][self.resname_i][:3], material)
            txt_lst += self.get_triangle_txt_lst('orange', DihedralPair.d_pair[dihedral_name][self.resname_i][1:], material)
        f_tcl = path.join(self.tcl_folder, 'draw_triangle.tcl')
        self.write_tcl(f_tcl, txt_lst)

    def get_triangle_txt_lst(self, color, pair_lst, material):
        return [f'draw color {color}',  f'draw material {material}',
                 'graphics 0 triangle ' + ' '.join([self.get_txt_triangle(pair) for pair in pair_lst])]

    def get_txt_triangle(self, pair):
        resid = self.d_resid_ij[pair[0]]
        name = pair[1]
        temp = self.u.select_atoms(f'resid {resid} and name {name}')
        coor = temp.positions[0]
        return '{' + f'{coor[0]:.2f} {coor[1]:.2f} {coor[2]:.2f}' + '}'

    def get_selection_txt_by_dihedral_name(self, dihedral_name):
        txt_1 = 'mol selection '
        if dihedral_name in ["C2prime-P", 'C4prime-P', "O4prime-O5prime"]:
            txt_2 = ' or '.join([self.get_txt(pair) for pair in DihedralPair.d_pair[dihedral_name]])
        else:
            txt_2 = ' or '.join([self.get_txt(pair) for pair in DihedralPair.d_pair[dihedral_name][self.resname_i]])
        return txt_1 + txt_2

    def get_txt(self, pair):
        resid = self.d_resid_ij[pair[0]]
        name = pair[1]
        return f'(resid {resid} and name {name})'

    def write_tcl(self, f_tcl, txt_lst):
        f = open(f_tcl, 'w')
        for txt in txt_lst:
            f.write(txt)
            f.write('\n')
        f.close()
        print(f'source {f_tcl}')

    def get_resid_i(self):
        if self.strand_id == 'STRAND1':
            return 5
        else:
            return 30

    def tachyon_take_photo_cmd(self, drawzone_folder, tga_name):
        output = path.join(drawzone_folder, tga_name)
        str_1 = f'render Tachyon {output} '
        str_2 = '"/usr/local/lib/vmd/tachyon_LINUXAMD64" '
        str_3 = '-aasamples 12 %s -format TARGA -o %s.tga'
        print(str_1+str_2+str_3)

    def get_resname_by_resid(self, resid):
        if self.strand_id == 'STRAND1':
            return self.d_seq[self.strand_id][resid-1]
        else:
            resid_modify = resid - 21
            return self.d_seq[self.strand_id][resid_modify-1]