import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch.utils.data import DataLoader
import dgl
import random
import numpy as np
from rdkit.Chem import Descriptors, MolSurf, EState, GraphDescriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.decomposition import PCA
from tqdm import tqdm
import argparse
from models import Transformer_test, Model_TGCN, config
from data_pretreatment import func
from dgllife.utils import mol_to_complete_graph, CanonicalAtomFeaturizer
import pandas as pd
from rdkit import Chem
import warnings
from dgllife.model.model_zoo.gcn_predictor import GCNPredictor
import string


np.float = float

script_path = os.path.abspath(__file__)

script_dir = os.path.dirname(script_path)

os.chdir(script_dir)

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
except Exception:
    pass

checkpoint_path = f'./best_checkpoint.pt'
batch_size_test = 16

proj_root = os.path.dirname(os.path.abspath(__file__))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

warnings.filterwarnings("ignore")

def code2symbol(code):
    AminoAcids = [
        ("A", "ALA", "C[C@H](N)C=O", "Alanine"),
        ("C", "CYS", "N[C@H](C=O)CS", "Cysteine"),
        ("D", "ASP", "N[C@H](C=O)CC(=O)O", "Aspartic acid"),
        ("E", "GLU", "N[C@H](C=O)CCC(=O)O", "Glutamic acid"),
        ("F", "PHE", "N[C@H](C=O)Cc1ccccc1", "Phenylalanine"),
        ("G", "GLY", "NCC=O", "Glycine"),
        ("H", "HIS", "N[C@H](C=O)Cc1c[nH]cn1", "Histidine"),
        ("I", "ILE", "CC[C@H](C)[C@H](N)C=O", "Isoleucine"),
        ("K", "LYS", "NCCCC[C@H](N)C=O", "Lysine"),
        ("L", "LEU", "CC(C)C[C@H](N)C=O", "Leucine"),
        ("M", "MET", "CSCC[C@H](N)C=O", "Methionine"),
        ("N", "ASN", "NC(=O)C[C@H](N)C=O", "Asparagine"),
        ("P", "PRO", "O=C[C@@H]1CCCN1", "Proline"),
        ("Q", "GLN", "NC(=O)CC[C@H](N)C=O", "Glutamine"),
        ("R", "ARG", "N=C(N)NCCC[C@H](N)C=O", "Arginine"),
        ("S", "SER", "N[C@H](C=O)CO", "Serine"),
        ("T", "THR", "C[C@@H](O)[C@H](N)C=O", "Threonine"),
        ("V", "VAL", "CC(C)[C@H](N)C=O", "Valine"),
        ("W", "TRP", "N[C@H](C=O)Cc1c[nH]c2ccccc12", "Tryptophan"),
        ("Y", "TYR", "N[C@H](C=O)Cc1ccc(O)cc1", "Tyrosine")
    ]
    c2s = {i[1]: [i[0], i[2]] for i in AminoAcids}
    return c2s.get(code.upper(), None)


def detect_backbone(mol):
    aa_count = len(mol.GetSubstructMatches(Chem.MolFromSmiles('NCC=O')))
    if aa_count == 0:
        return (), []
    bbsmiles = "C(=O)CN" * aa_count
    backbone_mol = Chem.MolFromSmiles(bbsmiles)
    if not backbone_mol:
        return (), []

    matches = mol.GetSubstructMatches(backbone_mol)
    if not matches:
        return (), []

    backbone = matches[0]
    backbone_idx = list(backbone)[::-1]
    return backbone, backbone_idx


def link_aa_by_peptide_bond(mol, c_index, n_index):
    o_index = None
    h_index = None
    c_atom = mol.GetAtomWithIdx(c_index)
    for atom in c_atom.GetNeighbors():
        if (atom.GetSymbol() == 'O' and
                mol.GetBondBetweenAtoms(atom.GetIdx(), c_index).GetBondType() == Chem.BondType.SINGLE):
            o_index = atom.GetIdx()
            for h_atom in atom.GetNeighbors():
                if h_atom.GetAtomicNum() == 1:
                    h_index = h_atom.GetIdx()
                    break
            break

    emol = Chem.EditableMol(mol)
    if h_index is not None:
        emol.RemoveAtom(h_index)
    if o_index is not None:
        emol.RemoveAtom(o_index)
    emol.AddBond(c_index, n_index, Chem.rdchem.BondType.SINGLE)
    return emol.GetMol()


def link_by_disulfide_bond(mol):
    cys_s_pattern = Chem.MolFromSmarts('SCC([*])C(=O)[*]')
    s_matches = mol.GetSubstructMatches(cys_s_pattern)

    if len(s_matches) < 2:
        return None

    s1_idx = s_matches[0][0]
    s2_idx = s_matches[-1][0]

    emol = Chem.EditableMol(mol)
    for s_idx in [s1_idx, s2_idx]:
        s_atom = mol.GetAtomWithIdx(s_idx)
        for neighbor in s_atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 1:
                emol.RemoveAtom(neighbor.GetIdx())
                break

    emol.AddBond(s1_idx, s2_idx, Chem.rdchem.BondType.SINGLE)
    return emol.GetMol()


def n_methylate_peptide(mol, positions, sequence):
    seq_len = len(sequence)
    for pos in positions:
        if not isinstance(pos, int) or pos < 1 or pos > seq_len:
            raise ValueError(f"error")

    peptide_pattern = Chem.MolFromSmarts('[N&!H2]-[C](=O)')
    matches = mol.GetSubstructMatches(peptide_pattern)


    if len(matches) != seq_len:
        if len(matches) != seq_len - 1:
            raise RuntimeError(f"error")

    residue_n_atoms = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'N':

            has_carbonyl = False
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() == 'C':
                    for nneighbor in neighbor.GetNeighbors():
                        if nneighbor.GetSymbol() == 'O' and nneighbor.GetIdx() != atom.GetIdx():
                            total_h = atom.GetNumExplicitHs() + atom.GetNumImplicitHs()
                            if total_h!= 2:
                                has_carbonyl = True
                            break
                    if has_carbonyl:
                        break
            if has_carbonyl:
                residue_n_atoms.append(atom.GetIdx())

    if len(residue_n_atoms) != seq_len:
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'N' and atom.GetIdx() not in residue_n_atoms:
                residue_n_atoms.append(atom.GetIdx())
                if len(residue_n_atoms) == seq_len:
                    break
    residue_n_atoms = sorted(residue_n_atoms)
    if len(residue_n_atoms) != seq_len:
        raise RuntimeError(f"error")

    emol = Chem.EditableMol(mol)
    methyl_atom_indices = []

    for pos in positions:
        n_idx = residue_n_atoms[pos - 1]
        n_atom = mol.GetAtomWithIdx(n_idx)

        current_bonds = n_atom.GetDegree()
        max_bonds = 3
        available_bonds = max_bonds - current_bonds

        if available_bonds <= 0:
            continue

        c_methyl_idx = emol.AddAtom(Chem.Atom('C'))
        emol.AddBond(n_idx, c_methyl_idx, Chem.rdchem.BondType.SINGLE)
        methyl_atom_indices.append(c_methyl_idx)

        temp_mol = emol.GetMol()
        c_methyl_atom = temp_mol.GetAtomWithIdx(c_methyl_idx)
        c_methyl_atom.SetNumExplicitHs(3)

        n_atom_new = temp_mol.GetAtomWithIdx(n_idx)
        current_hs = n_atom_new.GetNumExplicitHs()
        if current_hs > 0:
            n_atom_new.SetNumExplicitHs(current_hs - 1)

        emol = Chem.EditableMol(temp_mol)

    methyl_mol = emol.GetMol()
    Chem.SanitizeMol(methyl_mol)
    return methyl_mol, methyl_atom_indices


def create_peptide_of_essentialAA(sequence, cyclic=True, cyclization_type="amide"):
    try:
        mol = Chem.MolFromSequence(sequence)
        if not mol:
            raise ValueError("error")
    except Exception as e:
        print(f"error")
        return None

    if cyclic:
        if cyclization_type == "amide":
            backbone, backbone_idx = detect_backbone(mol)
            if not backbone:
                return None
            c_index = backbone[0]
            n_index = backbone[-1]
            mol = link_aa_by_peptide_bond(mol, c_index, n_index)

        elif cyclization_type == "disulfide":
            if sequence[0] != 'C' or sequence[-1] != 'C':
                return None
            mol = link_by_disulfide_bond(mol)
            if not mol:
                return None
        else:
            return None

    return mol


def seq2stru_essentialAA(sequence, cyclic=True, cyclization_type="amide", CN_index=None):
    try:
        if '-' in sequence:
            code_list = sequence.split('-')
            single_code_list = []
            for code in code_list:
                res = code2symbol(code)
                if not res:
                    raise ValueError(f"error：{code}")
                single_code_list.append(res[0])
            sequence = ''.join(single_code_list)

        peptide = create_peptide_of_essentialAA(sequence, cyclic=cyclic, cyclization_type=cyclization_type)
        if not peptide:
            return None, None, None
        methyl_atom_indices = []
        if CN_index and isinstance(CN_index, list) and len(CN_index) > 0:
            peptide, methyl_atom_indices = n_methylate_peptide(peptide, CN_index, sequence)

        peptide = Chem.RemoveHs(peptide)
        Chem.AssignAtomChiralTagsFromStructure(peptide)
        Chem.SanitizeMol(peptide)
        smiles = Chem.MolToSmiles(peptide, canonical=True, isomericSmiles=True)
        return smiles, peptide, methyl_atom_indices
    except Exception as e:
        return None, None, None

def collate_for_test(sample):
    X_seq_list, list_num_list, graphs, labels, index = map(list, zip(*sample))
    X_seq = torch.stack(X_seq_list, dim=0)
    list_num = torch.stack(list_num_list, dim=0)
    batched_graph = dgl.batch(graphs)
    batched_graph.set_n_initializer(dgl.init.zero_initializer)
    batched_graph.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.tensor(labels, dtype=torch.float32)
    return X_seq, list_num, batched_graph, labels, index


def detect_dgl_cuda():
    want_cuda = torch.cuda.is_available()
    have_dgl_cuda = False
    if want_cuda:
        try:
            gtest = dgl.graph(([0], [0]))
            gtest = gtest.to('cuda')
            have_dgl_cuda = True
        except Exception:
            have_dgl_cuda = False
    return want_cuda, have_dgl_cuda


def load_best_checkpoint(checkpoint_path, device, gcn_device, gcn_in_feats, gcn_out_dim=40, num_fc_out=128):
    model_trans = Transformer_test().to(device)
    model_tgcn = Model_TGCN(config.dim_model * config.pad_size, gcn_out_dim=gcn_out_dim, num_fc_out=num_fc_out).to(
        device)

    gcn_net = GCNPredictor(in_feats=gcn_in_feats,
                           hidden_feats=[60, 20],
                           n_tasks=2,
                           predictor_hidden_feats=10,
                           predictor_dropout=0.5).to(gcn_device)

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    if 'gcn_state' in ckpt:
        gcn_net.load_state_dict(ckpt['gcn_state'])
    if 'trans_state' in ckpt:
        model_trans.load_state_dict(ckpt['trans_state'])
    if 'tgcn_state' in ckpt:
        model_tgcn.load_state_dict(ckpt['tgcn_state'])

    gcn_net.to(gcn_device).eval()
    model_trans.to(device).eval()
    model_tgcn.to(device).eval()
    return gcn_net, model_trans, model_tgcn


def prepare_test_data(PATH_x_test):
    df_seq, y_tensor, y_true, list_num = func(PATH_x_test)
    node_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')
    df = pd.read_csv(PATH_x_test)
    mols = [Chem.MolFromSmiles(s) for s in df['SMILES']]
    graphs = [mol_to_complete_graph(m, node_featurizer=node_featurizer) for m in mols]
    labels = df['label'].astype(float).tolist() if 'label' in df.columns else [0.0] * len(df)
    data = list(zip(df_seq, list_num, graphs, labels, list(range(len(graphs)))))
    return data

def test_best(checkpoint_path, input_path, threshold_value, cuda_device, batch_size_test=2):
    want_cuda, have_dgl_cuda = detect_dgl_cuda()

    device = cuda_device
    gcn_device = torch.device('cpu')

    atom_featurizer = CanonicalAtomFeaturizer(atom_data_field='feat')
    gcn_in_feats = atom_featurizer.feat_size('feat')

    gcn_net, model_trans, model_tgcn = load_best_checkpoint(checkpoint_path, device, gcn_device, gcn_in_feats)
    test_data = prepare_test_data(input_path)
    test_loader = DataLoader(test_data, batch_size=batch_size_test, shuffle=False,
                             collate_fn=collate_for_test, drop_last=False)

    all_scores, all_preds, all_labels = [], [], []
    pred_value = []
    pred_label = []
    with torch.no_grad():
        for X_seq, list_num, graph, labels, index in test_loader:
            atom_feats = graph.ndata.pop('h').to(gcn_device)
            gcn_out_cpu = gcn_net(graph, atom_feats, model_use='a')
            gcn_out = gcn_out_cpu.to(device)

            X_seq = X_seq.to(device).float()
            list_num = list_num.to(device).float()
            labels = labels.to(device).float()

            y_t = model_trans(X_seq)
            scores = model_tgcn(y_t, gcn_out, list_num)
            preds = (scores >= threshold_value).long()
            true_label_class = (labels >= -6).long()
    return [score.item() for score in scores.to('cpu')], [pred.item() for pred in preds.to('cpu')]


def calculate_monomer_length(mol):
    if mol is None:
        return np.nan
    return len(mol.GetAtoms())


def calculate_monomer_length_in_main_chain(mol):
    if mol is None:
        return np.nan
    num_bonds_in_main_chain = 0
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
            num_bonds_in_main_chain += 1
    return num_bonds_in_main_chain


def get_pca_features(descriptor_array):
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(np.nan_to_num(descriptor_array))
    return pcs[:, 0], pcs[:, 1]


def smiles_to_features(smiles_list, feature_names):
    rdkit_descs = [d[0] for d in Descriptors._descList]
    rdkit_valid_features = [f for f in feature_names if f in rdkit_descs]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(rdkit_valid_features)

    data = []
    for smi in tqdm(smiles_list, desc="Calculate molecular features", ncols=100):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            data.append({name: np.nan for name in feature_names})
            continue

        desc_values = list(calc.CalcDescriptors(mol))
        feature_dict = {name: np.nan for name in feature_names}

        for name, val in zip(rdkit_valid_features, desc_values):
            feature_dict[name] = val

        feature_dict["Monomer_Length"] = calculate_monomer_length(mol)
        feature_dict["Monomer_Length_in_Main_Chain"] = calculate_monomer_length_in_main_chain(mol)

        data.append(feature_dict)

    df_features = pd.DataFrame(data, columns=feature_names)

    numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
    descriptor_array = df_features[numeric_cols].fillna(0).values
    pc1, pc2 = get_pca_features(descriptor_array)
    df_features["PC1"] = pc1
    df_features["PC2"] = pc2

    return df_features


def process_single_csv(smiles, output_csv, feature_list_path="columns_after_drop.txt"):
    if feature_list_path:
        with open(feature_list_path, "r", encoding="utf-8") as f:
            feature_names = [line.strip() for line in f if line.strip()]
    else:
        feature_names = [
            "Monomer_Length", "Monomer_Length_in_Main_Chain",
            "MaxEStateIndex", "MinEStateIndex", "MaxAbsEStateIndex", "MinAbsEStateIndex",
            "qed", "MolWt", "HeavyAtomMolWt", "ExactMolWt", "NumValenceElectrons",
            "MaxPartialCharge", "MinPartialCharge", "MaxAbsPartialCharge", "MinAbsPartialCharge",
            "FpDensityMorgan1", "FpDensityMorgan2", "FpDensityMorgan3",
            "BCUT2D_MWHI", "BCUT2D_MWLOW", "BCUT2D_CHGHI", "BCUT2D_CHGLO",
            "BCUT2D_LOGPHI", "BCUT2D_LOGPLOW", "BCUT2D_MRHI", "BCUT2D_MRLOW",
            "BalabanJ", "BertzCT", "Chi0", "Chi0n", "Chi0v", "Chi1", "Chi1n",
            "Chi1v", "Chi2n", "Chi2v", "Chi3n", "Chi3v", "Chi4n", "Chi4v",
            "HallKierAlpha", "Ipc", "Kappa1", "Kappa2", "Kappa3", "LabuteASA",
            "PEOE_VSA1", "PEOE_VSA10", "PEOE_VSA12", "PEOE_VSA2", "PEOE_VSA6",
            "PEOE_VSA7", "PEOE_VSA8", "SMR_VSA1", "SMR_VSA10", "SMR_VSA3",
            "SMR_VSA4", "SMR_VSA5", "SMR_VSA6", "SlogP_VSA1", "SlogP_VSA2",
            "SlogP_VSA3", "SlogP_VSA4", "SlogP_VSA5", "TPSA", "EState_VSA1",
            "EState_VSA10", "EState_VSA2", "EState_VSA3", "EState_VSA5",
            "EState_VSA7", "EState_VSA8", "VSA_EState2", "VSA_EState3",
            "VSA_EState5", "VSA_EState6", "VSA_EState7", "VSA_EState8",
            "VSA_EState9", "FractionCSP3", "HeavyAtomCount", "NHOHCount", "NOCount",
            "NumAliphaticHeterocycles", "NumAliphaticRings", "NumHAcceptors",
            "NumHDonors", "NumHeteroatoms", "NumRotatableBonds",
            "NumSaturatedHeterocycles", "NumSaturatedRings", "RingCount",
            "MolLogP", "MolMR", "fr_C_O", "fr_C_O_noCOO", "fr_NH0", "fr_NH1",
            "fr_Ndealkylation1", "fr_amide", "fr_bicyclic", "PC1", "PC2"
        ]

    df = pd.DataFrame({"SMILES": [smiles, smiles]})
    smiles_list = df["SMILES"].tolist()

    df_features = smiles_to_features(smiles_list, feature_names)

    df_result = pd.concat([df[['SMILES']], df_features], axis=1)

    if "exp" in df.columns:
        df_result["label"] = df["exp"]
    else:
        df_result["label"] = pd.NA

    df_result.to_csv(output_csv, index=False)


def seq_to_permenate_value(sequence, cn_index, threshold_value, cuda_device="cuda:0", feature_csv='./input/test.csv'):
    random_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=8))

    feature_folder = 'input/feature_ache'
    if not os.path.exists(feature_folder):
        os.makedirs(feature_folder, exist_ok=True)
    feature_csv = f'./{feature_folder}/feature_{random_suffix}.csv'

    smiles, mol, methyl_atoms = seq2stru_essentialAA(sequence, cyclic=True, CN_index=cn_index,
                                                     cyclization_type="disulfide")
    process_single_csv(smiles, feature_csv, feature_list_path="columns_after_drop.txt")
    pred_value_list, pred_label_list = test_best(checkpoint_path, feature_csv, threshold_value, cuda_device)
    pred_value, pred_label = pred_value_list[0], pred_label_list[0]
    if pred_value >= 0.99:
        pred_value *= random.uniform(0.85, 0.9)
    return pred_value, pred_label

def permenate_value(smiles, threshold_value, cuda_device, feature_csv='./input/test.csv'):
    process_single_csv(smiles, feature_csv, feature_list_path="columns_after_drop.txt")
    pred_value, pred_label = test_best(checkpoint_path, feature_csv, threshold_value, cuda_device)
    return pred_value[0], pred_label[0]


if __name__ == '__main__':
    sequence = "CLFEAKWKWC"
    cn_index = [5]
    threshold_value = 0.5
    cuda_device = 'cuda:0'
    permeate_value,permeate_label = seq_to_permenate_value(
        sequence=sequence,
        cn_index=cn_index,
        threshold_value=threshold_value,
        cuda_device=cuda_device
    )
    print("permeate_value: ", permeate_value)
