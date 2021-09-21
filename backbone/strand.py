class Strand:
    d_resid_lst = {'STRAND1': list(range(4, 19)), 'STRAND2': list(range(25, 40))}
    backbone_atomlst = ['P', 'O1P', 'O2P', "O5'", "C5'"]
    ribose_atomlst = ["C4'", "O4'", "C1'", "C2'", "C3'", "O3'"]

    def __init__(self, host, strand_id):
        self.host = host
        self.strand_id = strand_id
        self.resid_lst = self.d_resid_lst[self.strand_id]

    def get_gmx_makendx_text(self):
        resid_lst = [f'{resid}' for resid in self.resid_lst]
        resid_text = ' '.join(resid_lst)
        txt_1 = f'r {resid_text}'
        backbone_ribose_text = ' '.join(self.backbone_atomlst + self.ribose_atomlst)
        txt_2 = f'a {backbone_ribose_text}'
        return f'{txt_1} & {txt_2}'