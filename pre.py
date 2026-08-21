import copy
import json
import os
import pickle
import shutil
import subprocess
import tarfile
from pathlib import Path
from time import time
from typing import Dict, Final, List, Optional, Tuple

import numpy as np
import yaml
from boltz_utils import BoltzRunner
from boltz_utils import read_json_file
from boltz_utils import Input
from Bio.PDB import PDBIO, Model, PDBParser, Selection
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure
from loguru import logger
from numpy.typing import NDArray

from boltz_utils import get_botlz_prediction
from design_loss import (
    binder_helicity_loss,
    get_con_loss,
    get_pae_loss,
    get_plddt_loss,
    rg_loss,
    termini_distance_loss,
)

# from Bio.PDB.MMCIFParser import Structure as CIFStructure
from force_distance_constraint import DistanceConstraint
from ptm_utils import (
    get_fixed_ptm_list,
    ptm_list_to_extend_sequence,
    ptm_list_to_origin_sequence,
    ptm_list_to_sequence,
    ptm_list_to_sequence_list,
)
import warnings
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from Multi_CycGT.permenate import seq_to_permenate_value

warnings.filterwarnings("ignore", category=PossibleUserWarning)

RESTYPES: Final[List[str]] = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]
EXTENDED_RESTYPES: Final[List[str]] = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
    "_",
]


ACID2RES_DICT = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}
CYCLIC_PTMS = []
RES2ACID_DICT = dict([val, key] for key, val in ACID2RES_DICT.items())


def read_config_from_yaml(yaml_file):
    with open(yaml_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_config_to_yaml(config, yaml_file):
    with open(yaml_file, "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False)


def get_cc_groups(cc_indexes, use_fixed_group=False, fixed_group=None):
    if use_fixed_group:
        fix_cc_indexes = []
        for i in fixed_group:
            fix_cc_indexes.append(cc_indexes[i])
        return [fix_cc_indexes]

    if len(cc_indexes) == 3:
        return [
            [cc_indexes[0], cc_indexes[1]],
            [cc_indexes[0], cc_indexes[2]],
            [cc_indexes[1], cc_indexes[2]],
        ]
    if len(cc_indexes) == 4:
        return [
            [cc_indexes[0], cc_indexes[1], cc_indexes[2], cc_indexes[3]],
            [cc_indexes[0], cc_indexes[2], cc_indexes[1], cc_indexes[3]],
            [cc_indexes[0], cc_indexes[3], cc_indexes[1], cc_indexes[2]],
        ]
    return [[cc_indexes[0], cc_indexes[1]]]


def CC_index(peptide, get_all: bool = False):
    indexes_of_c = []
    index = -1
    while True:
        index = peptide.find("C", index + 1)
        if index == -1:
            break
        indexes_of_c.append(index)
    if get_all:
        return indexes_of_c
    C1 = indexes_of_c[0]
    C2 = indexes_of_c[-1]
    return C2, C1


def parse_pdb_file(path: str) -> str:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", path)
    res_list: List[Residue] = list(structure.get_chains())[0].child_list
    raw_res_dict = {}
    for i, res in enumerate(res_list):
        name = res.get_resname()
        index = res.get_id()[1]
        raw_res_dict[index] = RES2ACID_DICT[name]

    length = max(raw_res_dict.keys())
    seq = []
    for i in range(1, length + 1):
        if i in raw_res_dict.keys():
            seq.append(raw_res_dict[i])
        else:
            seq.append("X")
    return "".join(seq)


def CC_distance(peptide):
    C2, C1 = CC_index(peptide)
    return C2 - C1


def get_predicted_hotspot_pair_dist(
    hotspot_pairs: List[Tuple[Tuple[str, int], Tuple[str, int]]],
    chain_coord_dict: Dict[str, Dict[int, NDArray]],
) -> float:
    """Calculate the distance between hotspot pairs."""
    total_distance = 0
    for (chain1, res1), (chain2, res2) in hotspot_pairs:
        coord1 = chain_coord_dict[chain1][res1][:, np.newaxis, :]
        coord2 = chain_coord_dict[chain2][res2][np.newaxis, :, :]
        delta = coord1 - coord2  # 形状 (2, 2, 3)
        distances = np.linalg.norm(delta, axis=-1)  # 形状 (2, 2)
        min_distance = np.min(distances)
        total_distance += min_distance

    if len(hotspot_pairs) > 0:
        return total_distance / len(hotspot_pairs)

    return 0.0


def get_pocket_dist(
    pocket: Dict[str, List[int]],
    chain_coord_dict: Dict[str, Dict[int, NDArray]],
    ligand_chain_id: str,
) -> float:
    """Calculate the distance between pocket and ligand."""

    pocket_coords = []
    for chain, pocket_indexes in pocket.items():
        for res_index in pocket_indexes:
            pocket_coords.append(chain_coord_dict[chain][res_index])
    # N * 3
    pocket_coords = np.concatenate(pocket_coords)
    # M * 3
    ligand_coords = np.concatenate(list(chain_coord_dict[ligand_chain_id].values()))

    coord1 = pocket_coords[:, np.newaxis, :]
    coord2 = ligand_coords[np.newaxis, :, :]
    delta = coord1 - coord2
    # N * M
    distances = np.linalg.norm(delta, axis=-1)
    cloest_dist = np.min(distances, axis=1)  # N

    return np.mean(cloest_dist)


def get_hotspot_precision(
    predict_hotspots: Dict[str, List[int]],
    gt_hotspots: Dict[str, List[int]],
) -> float:
    """Calculate the precision of hotspot pairs."""
    TP = 0
    FP = 0
    for chain, gt_hotspot in gt_hotspots.items():
        predict_hotspot = predict_hotspots.get(chain, [])
        TP += len(set(gt_hotspot) & set(predict_hotspot))
        FP += len(predict_hotspot) - TP
    if TP + FP == 0:
        return 0
    return TP / (TP + FP)


def read_structure_coordinates(structure: Structure):
    chain_coord_dict = {}
    for chain in structure:
        coord_dict = {}

        for residue in chain:
            atom_coords = []

            for atom in residue:
                atom_coords.append(atom.coord)
            coord_dict[residue.id[1]] = np.array(atom_coords)

        chain_coord_dict[chain.id] = coord_dict

    return chain_coord_dict


def read_pdb_coordinates(file_path):
    protein_resno = []
    protein_atoms = []
    protein_atom_coords = []
    protein_res_name = []
    with open(file_path, "r") as pdb_file:
        for line in pdb_file:
            if line.startswith("ATOM"):
                resname = line[16:20].strip()
                resno = int(line[23:30])
                protein_resno.append(resno)
                atoms = line[12:16].strip()
                protein_atoms.append(atoms)
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                protein_atom_coords.append([x, y, z])
                protein_res_name.append(resname)

    return (
        np.array(protein_resno),
        np.array(protein_atoms),
        np.array(protein_atom_coords),
        np.array(protein_res_name),
    )


def find_peptide_index(arr):
    for i in range(1, len(arr)):
        if arr[i] < arr[i - 1]:
            return i


def is_peptide_sequence_valid(
    peptide_sequence: str, cc_num: int = 2, is_nc_cyclic: bool = False
) -> bool:
    """Check if a peptide sequence is valid about CC.

    Args:
        peptide_sequence (str): input str of peptide sequence
        strict (bool, optional): if limit CC number ==2 or >=2.

    Returns:
        bool: peptide sequence is valid or not
    """
    if not peptide_sequence:
        return False
    length = len(peptide_sequence)

    if is_nc_cyclic:
        return peptide_sequence.count("C") < 2
    else:
        return (
            peptide_sequence.count("C") == cc_num
            and CC_distance(peptide_sequence) / length > 2 / 3
        )


def copy_str_by_index(target: str, src: str, mask: NDArray) -> str:
    """Copy src into target by mask == 1.

    Args:
        target (str): target str
        src (str): src str
        mask (NDArray): mask array

    Returns:
        str: changed target str
    """
    target = list(target)
    for i, flag in enumerate(mask):
        if flag:
            target[i] = src[i]
    return "".join(target)


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


def matrix_softmax(input: NDArray) -> NDArray:
    exp = np.exp(input)
    return exp / (np.sum(exp, axis=1, keepdims=True) + 1e-6)


def is_peptide_with_mask(peptide_sequence: str) -> bool:
    """Check input peptide is incluede mask char 'X' or not.

    Args:
        peptide_sequence (str): Like 'ACXXXA' or 'ACADAAC'

    Returns:
        bool: is_peptide_with_mask '_'
    """
    return peptide_sequence.count("_") > 0


def is_init_peptide_sequence_valid(
    peptide_mask_indexes, initial_peptide_seq, peptide_length, cc_num, is_nc_cyclic
):
    is_valid = True
    if not is_nc_cyclic:
        if initial_peptide_seq and np.any(peptide_mask_indexes > 0):
            # Fill 'C' in the unmasked index position of initial_peptide_seq to check initial_peptide_seq can get valid random sequence.
            free_sequence = "C" * peptide_length
            free_sequence = copy_str_by_index(
                free_sequence, initial_peptide_seq, peptide_mask_indexes
            )
            is_valid = (
                free_sequence.count("C") >= cc_num
                and CC_distance(free_sequence) / peptide_length > 2 / 3
            )
    else:
        is_valid = initial_peptide_seq.count("C") < 2

    return is_valid


def random_init_cys_ptm(initial_peptide_seq, random_ptms):
    if len(CYCLIC_PTMS) < 1:
        return random_ptms
    cc_index = CC_index(initial_peptide_seq, get_all=True)
    cc_ptms = np.array(CYCLIC_PTMS + ["CYS"] * len(CYCLIC_PTMS) * 2)
    # cc_ptms = np.array(CYCLIC_PTMS)
    weights = np.random.gumbel(0, 1, (len(cc_index), len(cc_ptms)))
    weights = matrix_softmax(weights)
    random_cc_ptm = cc_ptms[np.argmax(weights, axis=1)]

    new_ptms = []
    for ptm in random_ptms:
        if ptm[2] == "C":
            continue
        new_ptms.append(ptm)

    for index, ptm in zip(cc_index, random_cc_ptm):
        if ptm == "CYS":
            continue
        new_ptm = (index + 1, ptm, "C")
        new_ptms.append(new_ptm)

    return new_ptms


def random_initialize_weights(
    peptide_length: int,
    peptide_mask_indexes: NDArray,
    cc_num: int = 2,
    initial_peptide_seq: Optional[str] = None,
    is_nc_cyclic: bool = False,
    use_ptms: bool = False,
    ptms: Optional[List[str]] = None,
) -> Tuple[NDArray, str]:
    """Initialize sequence probabilities"""

    max_sample_times = 10000
    sample_times = 0

    restypes = np.array(RESTYPES + ptms) if use_ptms else np.array(RESTYPES)
    random_ptms = []
    res_index = np.arange(len(restypes))
    p = [15.0] * len(RESTYPES) + [1.0] * (len(restypes) - len(RESTYPES))
    p = np.array(p) / np.sum(p)
    while True:

        weights = np.random.choice(res_index, peptide_length, p=p)

        # Get the peptide sequence
        # Residue types
        random_peptide_sequence = ""
        random_restypes = restypes[weights]
        if use_ptms:
            random_peptide_sequence, random_ptms = ptm_list_to_origin_sequence(
                random_restypes, peptide_mask_indexes
            )
        else:
            random_peptide_sequence = "".join(random_restypes)

        # only random the unlocked residues
        anti_mask = 1 - peptide_mask_indexes
        initial_peptide_seq = copy_str_by_index(
            initial_peptide_seq, random_peptide_sequence, anti_mask
        )

        # Prevent endless loop
        sample_times += 1
        if sample_times > max_sample_times:
            raise ValueError(
                "Randomly generation out of limit times: 10000, please check your input."
            )

        if is_peptide_sequence_valid(initial_peptide_seq, cc_num, is_nc_cyclic):
            if not is_nc_cyclic and use_ptms and len(CYCLIC_PTMS) > 0:
                random_ptms = random_init_cys_ptm(initial_peptide_seq, random_ptms)

            return initial_peptide_seq, random_ptms


def mock_loss_input(target_sequence, binder_sequence, res_index, hotspot):
    inputs = {"opt": {}}
    inputs["opt"]["con"] = {
        "num": 2,
        "cutoff": 14.0,
        "binary": False,
        "seqsep": 9,
        "num_pos": float("inf"),
    }
    inputs["opt"]["i_con"] = {
        "num": 1,
        "cutoff": 21.6875,
        "binary": False,
        "num_pos": float("inf"),
    }
    inputs["opt"]["weights"] = {
        "pae": 0.1,
        "plddt": 0.1,
        "ipae": 0.1,
        "con": 0.1,
        "i_con": 0.1,
    }
    if len(hotspot) > 0:
        inputs["opt"]["hotspot"] = hotspot
    target_len = len(target_sequence)
    binder_len = len(binder_sequence)
    inputs["seq_mask"] = np.ones((target_len + binder_len))

    inputs["residue_index"] = np.concatenate(
        [res_index, np.arange(target_len + 50, target_len + 50 + binder_len)]
    )
    return inputs


def loss_binder(inputs, outputs, target_len, binder_len):
    """get losses"""
    opt = inputs["opt"]
    mask = inputs["seq_mask"]

    zeros = np.zeros_like(mask)
    tL, bL = target_len, binder_len

    binder_id = np.zeros_like(mask)
    binder_id[-bL:] = mask[-bL:]

    target_id = np.zeros_like(mask)
    if "hotspot" in opt:
        target_id[opt["hotspot"]] = mask[opt["hotspot"]]
        i_con_loss = get_con_loss(
            inputs, outputs, opt["i_con"], mask_1d=target_id, mask_1b=binder_id
        )
    else:
        target_id[:tL] = mask[:tL]
        i_con_loss = get_con_loss(
            inputs, outputs, opt["i_con"], mask_1d=binder_id, mask_1b=target_id
        )

    # unsupervised losses
    loss = {
        # "plddt": get_plddt_loss(outputs, mask_1d=binder_id),  # plddt over binder
        "pae": get_pae_loss(outputs, mask_1d=binder_id),  # pae over binder + interface
        "con": get_con_loss(
            inputs, outputs, opt["con"], mask_1d=binder_id, mask_1b=binder_id
        ),
        # interface
        "i_con": i_con_loss,
        "ipae": get_pae_loss(outputs, mask_1d=binder_id, mask_1b=target_id),
        # "termini_distance_loss": termini_distance_loss(inputs, outputs, binder_len),
        "helix_loss": binder_helicity_loss(inputs, outputs, target_len, binder_len),
        "rg_loss": rg_loss(outputs, binder_len),
    }
    weight_loss = 1e-5
    weight_loss += loss["pae"] * 0.1
    weight_loss += loss["con"] * 1.0
    weight_loss += loss["i_con"] * 1.0
    weight_loss += loss["ipae"] * 0.4
    weight_loss += loss["helix_loss"] * 0.3
    weight_loss += loss["rg_loss"] * 0.3
    return weight_loss


def is_cyclic_valid(
    cyclic_group: List[Tuple[int, int]],
    peptide_res_index: NDArray,
    peptide_atoms: NDArray,
    peptide_coords: NDArray,
    bond_limit: Tuple[float, float],
) -> float:
    cyclic_norm = 0.0
    valid_num = 0
    for head, tail in cyclic_group:

        head_index = np.argwhere(peptide_res_index == head + 1)[:, 0]
        head_sg = np.argwhere(peptide_atoms[head_index] == "SG")[:, 0]
        tail_index = np.argwhere(peptide_res_index == tail + 1)[:, 0]
        tail_sg = np.argwhere(peptide_atoms[tail_index] == "SG")[:, 0]
        head_sg_coord = peptide_coords[head_index][head_sg]
        tail_sg_coord = peptide_coords[tail_index][tail_sg]
        act_dist = np.sqrt(np.square(head_sg_coord - tail_sg_coord).sum())
        if act_dist >= bond_limit[0] and act_dist <= bond_limit[1]:
            bond_limit = 0
            valid_num += 1
            continue
        cyclic_norm += min(
            np.abs(act_dist - bond_limit[0]), np.abs(act_dist - bond_limit[1])
        )

    if valid_num == len(cyclic_group):
        return 0

    return cyclic_norm / (len(cyclic_group) - valid_num)


def convert_cif_to_pdb(cif_file: str) -> Structure | None:
    """
    Convert a CIF file to PDB format using Biopython.
    """
    parser = MMCIFParser(QUIET=True)
    s = parser.get_structure("structure_cif", cif_file)
    io = PDBIO()
    io.set_structure(s)
    pdb_path = Path(cif_file).with_suffix(".pdb")
    io.save(str(pdb_path))
    if pdb_path.exists():
        return s[0]
    return None


def get_ipae(predicted_aligned_error, mask_1d=None, mask_1b=None, mask_2d=None):
    p = predicted_aligned_error / 31.0
    p = (p + p.T) / 2
    L = p.shape[0]
    if mask_1d is None:
        mask_1d = np.ones(L)
    if mask_1b is None:
        mask_1b = np.ones(L)
    if mask_2d is None:
        mask_2d = np.ones((L, L))
    mask_2d = mask_2d * mask_1d[:, None] * mask_1b[None, :]
    x_masked = (p * mask_2d).sum() / (1e-8 + mask_2d.sum())
    return x_masked


def get_confidences(
    plddt_path: Path,
    pae_path: Path,
    confidences_json_path: Path,
    target_chain_length: int,
) -> tuple[float, float, float, float, List[Tuple[Tuple[str, int], Tuple[str, int]]]]:

    confidences_dict = read_json_file(confidences_json_path)
    plddt = np.load(plddt_path)["plddt"][-target_chain_length:].mean() * 100.0
    iplddt = confidences_dict["complex_iplddt"] * 100.0
    ptm = confidences_dict["ptm"]
    iptm = list(confidences_dict["pair_chains_iptm"]["0"].values())[-1]
    ipae = get_ipae(np.load(pae_path)["pae"])
    return plddt, 0, iptm, ipae, iplddt


def groups_predict_cycle(
    receptor_input: Input,
    model_runner: BoltzRunner,
    peptide_sequence: str,
    output_dir_base,
    pocket,
    num_iter,
    distance_constraints: List[DistanceConstraint],
    cc_groups: List[int] = None,
    ptms: List[Tuple[int, str, str]] = [],
    ligand_bonds=None,
):

    plddt = 0
    ipae = 0
    iptm = 0
    hotspot_distance = 0
    score = 0
    best_index = 0
    stas_distance_constraints = []
    for i, cc_group in enumerate(cc_groups):
        cur_plddt, curr_i_pae, curr_iptm, curr_hotspot_distance, curr_score = (
            predict_cycle(
                receptor_input,
                model_runner,
                peptide_sequence,
                output_dir_base,
                pocket,
                num_iter,
                distance_constraints,
                cc_list=cc_group,
                group_index=i,
                ptms=ptms,
                ligand_bonds=None,
            )
        )
        # stas_distance_constraints.append(copy.deepcopy(distance_constraints))
        if cur_plddt > plddt:
            plddt = cur_plddt
            ipae = curr_i_pae
            iptm = curr_iptm
            hotspot_distance = curr_hotspot_distance
            score = curr_score
            best_index = i

    group_str = "_".join([str(i) for i in cc_groups[best_index]]) + f"_{best_index}"
    # distance_constraints = stas_distance_constraints[best_index]
    return plddt, ipae, iptm, hotspot_distance, score, "_" + group_str

def get_permeability(peptide_sequence: str, ptms: List[Tuple[int, str, str]] = []):

    cn_index = [ptm[0] for ptm in ptms]
    threshold_value = 0.7
    print(peptide_sequence,cn_index)
    pred_value,pred_label = seq_to_permenate_value(peptide_sequence,cn_index,threshold_value)
    print(pred_value,pred_label)
    return pred_value


def predict_cycle(
    receptor_input: Input,
    model_runner: BoltzRunner,
    peptide_sequence,
    output_dir_base,
    pocket: Dict[str, List[int]],
    num_iter,
    distance_constraints: List[DistanceConstraint],
    pep_head: int = 0,
    pep_tail: int = 0,
    cc_list: List[int] = None,
    group_index: int = -1,
    is_nc_cyclic: bool = False,
    ptms: List[Tuple[int, str, str]] = [],
    ligand_bonds=None,
):
    logger.debug(
        f"Start eval {peptide_sequence},{ptm_list_to_sequence(peptide_sequence, ptms)}, {pep_head}-{pep_tail}"
    )

    bonds = []
    if ligand_bonds is None or len(ligand_bonds) == 0:
        ligand_bonds = []

    if cc_list is not None and len(cc_list) > 0:
        for head, tail in zip(cc_list[::2], cc_list[1::2]):
            bonds.append(((head, "SG"), (tail, "SG")))
    else:
        if is_nc_cyclic:
            bonds.append(((0, "N"), (len(peptide_sequence) - 1, "C")))
        else:
            bonds.append(((pep_head, "SG"), (pep_tail, "SG")))

    temp_out_dir = Path(output_dir_base).parent / "temp"
    if not temp_out_dir.exists():
        temp_out_dir.mkdir(parents=True)
    start_time = time()
    logger.disable("af3_utils")

    target_chain_id = get_botlz_prediction(
        receptor_input,
        peptide_sequence,
        model_runner,
        temp_out_dir,
        [],
        ptms,
        ligand_bonds=ligand_bonds,
    )

    logger.enable("af3_utils")

    task_id = receptor_input.record.id + "_complex"
    # cif_file = temp_out_dir / task_id / f"{task_id}_model_0.cif"
    # predict_structure = convert_cif_to_pdb(cif_file)
    pdb_file = temp_out_dir / task_id / f"{task_id}_model_0.pdb"
    parser = PDBParser(QUIET=True)
    predict_structure = parser.get_structure("structure_cif", pdb_file)[0]

    confidences_json_path = (
        temp_out_dir / task_id / f"confidence_{task_id}_model_0.json"
    )

    plddt_path = temp_out_dir / task_id / f"plddt_{task_id}_model_0.npz"
    pae_path = temp_out_dir / task_id / f"pae_{task_id}_model_0.npz"

    plddt, ptm, iptm, ipae, iplddt = get_confidences(
        plddt_path, pae_path, confidences_json_path, len(peptide_sequence)
    )

    all_coords = read_structure_coordinates(predict_structure)
    hotspot_distance = 0.0
    hotspot_precision = 0.0
    if pocket is not None and len(pocket) > 0:
        hotspot_distance = get_pocket_dist(pocket, all_coords, target_chain_id)

    score_plddt = (
        plddt * 0.02
        + 1 / (abs(hotspot_distance - 3.5) + 2.0) * 2
        + hotspot_precision
    )
    permeability = get_permeability(peptide_sequence, ptms)

    score_plddt = score_plddt / 3.0
    score = np.array([score_plddt, permeability], dtype=np.float32)
    score_str = str(score)

    logger.info(
        f"Iter {num_iter} Eval {peptide_sequence} cost time: {time() - start_time}s"
    )
    logger.info(
        f"Score: {score_str} pLDDT: {plddt:.2f}, iPAE: {ipae:.2f}, IPTM: {iptm:.2f} "
    )
    new_file_path = Path(output_dir_base) / f"af3_{num_iter}.pdb"
    if cc_list is not None:
        ss_indexs = "_".join([str(i) for i in cc_list])
        new_file_path = (
            Path(output_dir_base) / f"af3_{num_iter}_{ss_indexs}_{group_index}.pdb",
        )

    shutil.move(pdb_file, new_file_path)
    shutil.rmtree(temp_out_dir)

    return plddt, ipae, iptm, hotspot_distance, score, permeability


def sequence_to_onehot(
    sequence,
    aatypes=RESTYPES,
    max_length: int = 0,
    use_ptms: bool = False,
    init_ptms: List[tuple[int, str, str]] = [],
):

    length = max(len(sequence), max_length)
    one_hot_arr = np.zeros((length, len(aatypes)), dtype=np.int32)
    sequence_list = list(sequence)
    if use_ptms:
        sequence_list = ptm_list_to_sequence_list(sequence_list, init_ptms)
    for aa_index, aa_type in enumerate(sequence_list):
        aa_id = aatypes.index(aa_type)
        one_hot_arr[aa_index, aa_id] = 1
    return one_hot_arr


def onehot_to_sequence(onehot, aatypes: List[str] = RESTYPES):

    aatypes = np.array(aatypes)
    res_list = aatypes[np.argmax(onehot, axis=1)]

    return ptm_list_to_origin_sequence(res_list)


def onehot_to_sequence_with_extend(onehot, aatypes: List[str] = RESTYPES):

    aatypes = np.array(aatypes)
    res_list = aatypes[np.argmax(onehot, axis=1)]

    return ptm_list_to_extend_sequence(res_list)


def get_availables(
    state: NDArray,
    locked_mask: NDArray,
    enable_extend: bool = False,
    peptide_len: int = 0,
    c_index_horizontal: List[int] = [],
    c_index_vertical=[],
    allow_extra_C: bool = False,
    allow_c_mutate_interior: bool = True,
    is_nc_cyclic: bool = False,
) -> NDArray:
    """Get avaliables positions to mutate. unavailable positions are locked by locked_mask and current residues index in aatypes.

    Args:
        state (NDArray): input one-hot aatypes for residues.
        locked_mask (NDArray): locked_mask, 0 for available, 1 for locked.
        enable_extend (bool, optional): whether to enable extended peptide sequence. Defaults to False.

    Returns:
        NDArray: availables positions of state to mutate.
    """

    row, col = state.shape
    availables = np.arange(row * col)
    availables = availables.reshape(row, col)
    state_tamp = state.copy()
    if not allow_extra_C and not is_nc_cyclic:
        # not allow to mutate into 'C'
        for i in c_index_horizontal:
            state_tamp[:, i] = 1

    if enable_extend:
        # non-extended residues are not available to mutate into 'X'
        # extended residues are available to delete by mutating into 'X'
        delta_len = row - peptide_len
        split_index = np.flatnonzero(locked_mask == 0)[-1] + 1
        state_tamp[:, -1] = 1
        state_tamp[split_index - delta_len : split_index, -1] = 0

    # delete masked rows

    if allow_c_mutate_interior and not is_nc_cyclic:
        # allow to mutate into 'C' in ptm
        for i in c_index_vertical:
            for j in c_index_horizontal:
                state_tamp[i, j] = 0

    if is_nc_cyclic:
        # assert len(c_index_vertical) < 2
        if len(c_index_vertical) == 1:
            only_one_C = c_index_vertical[0]
            state_tamp[only_one_C, :] = 0

            for i in c_index_horizontal:
                state_tamp[:, i] = 1

    state_tamp[locked_mask == 1] = 1
    # select all availables residues
    availables: NDArray = availables[np.nonzero(state_tamp == 0)]

    return availables


def mutate_seq(
    peptide_sequence: str,
    ex_list: List[str],
    locked_mask: NDArray,
    cc_num: int = 2,
    is_nc_cyclic=False,
    all_restypes=RESTYPES,
    use_ptms: bool = False,
    init_ptms: List[tuple[int, str, str]] = [],
) -> str:
    """Random mutate a peptide sequence except locked residues and peptide sequence not excuted.

    Args:
        peptide_sequence (peptide_sequence): input peptide sequence
        ex_list (List[str]): excuted peptides.
        locked_mask (NDArray): indexes of locked residues

    Returns:
        str: mutated peptide sequence.
    """
    restypes = np.array(all_restypes)

    initial_peptide_seq = peptide_sequence
    seq_length = len(peptide_sequence)
    anti_mask = 1 - locked_mask
    random_ptms = []

    res_index = np.arange(len(restypes))
    p = [15.0] * len(RESTYPES) + [1.0] * (len(restypes) - len(RESTYPES))
    p = np.array(p) / np.sum(p)
    while True:

        weights = np.random.choice(res_index, seq_length, p=p)
        # Get the peptide sequence
        # Residue types
        random_peptide_sequence = ""
        random_restypes = restypes[weights]

        random_peptide_sequence, random_ptms = ptm_list_to_origin_sequence(
            random_restypes, locked_mask
        )

        initial_peptide_seq = copy_str_by_index(
            initial_peptide_seq, random_peptide_sequence, anti_mask
        )
        # limit mutate seq is valid about CC

        if is_peptide_sequence_valid(
            initial_peptide_seq, cc_num, is_nc_cyclic=is_nc_cyclic
        ):
            if not is_nc_cyclic and use_ptms and len(CYCLIC_PTMS) > 0:
                random_ptms = random_init_cys_ptm(initial_peptide_seq, random_ptms)
            random_ptms.extend(get_fixed_ptm_list(init_ptms, locked_mask))
            ptm_initial_peptide_seq = ptm_list_to_sequence(
                initial_peptide_seq, random_ptms
            )
            if ptm_initial_peptide_seq not in ex_list:
                break
    return initial_peptide_seq, random_ptms


def mutate_extend_seq(
    peptide_sequence: str,
    ex_list: List[str],
    locked_mask: NDArray,
    init_len: int = 0,
    max_extend_length: int = 0,
    cc_num: int = 2,
    is_nc_cyclic=False,
    all_restypes=RESTYPES,
    use_ptms: bool = False,
    init_ptms: List[tuple[int, str, str]] = [],
) -> str:

    ex_restypes = np.array(all_restypes)
    restypes = np.array(all_restypes[:-1])

    delta_l = max_extend_length - init_len
    split_index = np.flatnonzero(locked_mask == 0)[-1]
    ex_start = split_index - delta_l + 1
    ex_end = split_index + 1

    unlocked_mask = 1 - locked_mask

    initial_peptide_seq = peptide_sequence
    random_ptms = []
    seq_length = len(peptide_sequence)

    res_index = np.arange(len(restypes))
    p = [15.0] * len(RESTYPES) + [1.0] * (len(restypes) - len(RESTYPES))
    p = np.array(p) / np.sum(p)

    ex_res_index = np.arange(len(ex_restypes))
    ex_p = (
        [10.0] * len(RESTYPES) + [1.0] * (len(ex_restypes) - len(RESTYPES) - 1) + [5.0]
    )
    ex_p = np.array(ex_p) / np.sum(ex_p)

    while True:

        weights = np.random.choice(res_index, max_extend_length, p=p)

        ex_weights = np.random.choice(ex_res_index, delta_l, p=ex_p)
        # Get the peptide sequence
        # Residue types

        random_restypes = restypes[weights]
        random_restypes[ex_start:ex_end] = ex_restypes[ex_weights]
        random_peptide_sequence, random_ptms = ptm_list_to_extend_sequence(
            random_restypes, locked_mask
        )

        initial_peptide_seq = copy_str_by_index(
            initial_peptide_seq, random_peptide_sequence, unlocked_mask
        )

        # limit mutate seq is valid about CC
        if is_peptide_sequence_valid(
            initial_peptide_seq, cc_num, is_nc_cyclic=is_nc_cyclic
        ):
            if not is_nc_cyclic and use_ptms and len(CYCLIC_PTMS) > 0:
                random_ptms = random_init_cys_ptm(initial_peptide_seq, random_ptms)
            random_ptms.extend(get_fixed_ptm_list(init_ptms, locked_mask))
            initial_peptide_seq_ptm = ptm_list_to_sequence(
                initial_peptide_seq, random_ptms
            )
            if initial_peptide_seq_ptm not in ex_list:
                break
    return initial_peptide_seq, random_ptms


def get_locked_mask_from_seq(peptide_length: int, peptide_sequence: str) -> NDArray:
    """get locked mask from mutilated peptide sequence.

    Args:
        peptide_length (int): peptide length
        peptide_sequence (str): mutilated peptide sequence. e.g.  '_A___C__C__'

    Returns:
        NDArray: locked mask array. e.g. [0, 1, 1, 0, 0, 0, 1, 1, 1, 1] 1 mean locked , 0 mean not locked.
    """
    ligand_seq_locked_mask = np.ones(peptide_length, dtype=np.int64)
    if peptide_sequence:
        peptide_sequence_array = np.array(list(peptide_sequence))
        ligand_seq_locked_mask[peptide_sequence_array == "_"] = 0

    return ligand_seq_locked_mask


def get_locked_mask_from_flag(
    peptide_length: int, ligand_seq_locked_mask_index: List[int]
) -> NDArray:
    """Convert str mask list into np.array mask

    Args:
        peptide_length (int): input peptide length
        ligand_seq_locked_mask_index (List[int]): input peptide locked mask index str, like [0,1,2,3]

    Raises:
        ValueError: Any of input mask index out of peptide length.

    Returns:
        NDArray: locked mask array. 1 mean locked and 0 mean unlocked.
    """
    ligand_seq_locked_mask = np.zeros(peptide_length, dtype=np.int64)
    if len(ligand_seq_locked_mask_index) > 0:
        mask_indexes = np.array(ligand_seq_locked_mask_index, dtype=np.int32)

        # check input seq mask valid.
        if np.any(mask_indexes > peptide_length - 1):
            raise ValueError(
                f"Invalid residue index in args_ligand_seq_locked_mask: {ligand_seq_locked_mask_index}"
            )
        ligand_seq_locked_mask[mask_indexes] = 1

    return ligand_seq_locked_mask


def get_emphasize_locked_sequence_str(
    peptide_sequence: str,
    locked_mask: NDArray,
    init_ptms: List[Tuple[int, str, str]] = [],
) -> str:
    """get the colorful str in order to emphasize the locked residues.

    Args:
        peptide_sequence (str): peptide sequence
        locked_mask (NDArray): locked mask array

    Returns:
        str: emphasize locked sequence str. Residue C will be Red. Other Locked residues will be Blue.
    """
    peptide_sequence_char_list = (
        list(peptide_sequence)
        if len(init_ptms) == 0
        else ptm_list_to_sequence_list(peptide_sequence, init_ptms)
    )
    if len(locked_mask) != len(peptide_sequence_char_list):
        delta_l = len(locked_mask) - len(peptide_sequence_char_list)
        split_index = np.flatnonzero(locked_mask == 0)[-1]
        locked_mask = list(locked_mask[: split_index - delta_l]) + list(
            locked_mask[split_index:]
        )
    cyc_res = CYCLIC_PTMS + ["C"]
    show_char_list = []
    for i, nce in enumerate(locked_mask):
        res_char = peptide_sequence_char_list[i]
        is_ptm = len(peptide_sequence_char_list[i]) > 1
        if nce == 1:
            if res_char in cyc_res:
                if is_ptm:
                    show_char_list.append(f"<red>({res_char})</red>")
                else:
                    show_char_list.append(f"<red>{res_char}</red>")
            else:
                if is_ptm:
                    show_char_list.append(f"<blue>({res_char})</blue>")
                else:
                    show_char_list.append(f"<blue>{res_char}</blue>")
        else:
            if is_ptm:
                show_char_list.append(f"({res_char})")
            else:
                show_char_list.append(res_char)
    return "".join(show_char_list)
