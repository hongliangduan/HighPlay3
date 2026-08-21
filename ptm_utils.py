import re
from typing import Dict, List, Tuple
from loguru import logger
from boltz_finetune.utils.input_utils import (
    get_modified_residue,
    parse_modified_sequence,
    ModifiedResidueId,
)

from numpy.typing import NDArray
from boltz_finetune.utils.temp_one import CCD_NAME_TO_ONE_LETTER

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
ACID_NAME_2RES_DICT = {
    "ALANINE": "ALA",
    "ARGININE": "ARG",
    "ASPARAGINE": "ASN",
    "ASPARTIC": "ASP",
    "CYSTEINE": "CYS",
    "GLUTAMINE": "GLN",
    "GLUTAMIC": "GLU",
    "GLYCINE": "GLY",
    "HISTIDINE": "HIS",
    "ISOLEUCINE": "ILE",
    "LEUCINE": "LEU",
    "LYSINE": "LYS",
    "METHIONINE": "MET",
    "PHENYLALANINE": "PHE",
    "PROLINE": "PRO",
    "SERINE": "SER",
    "THREONINE": "THR",
    "TRYPTOPHAN": "TRP",
    "TYROSINE": "TYR",
    "VALINE": "VAL",
}
RES2ACID_DICT = dict([val, key] for key, val in ACID2RES_DICT.items())


def parse_modified_sequence(squence_str: str) -> tuple[str, List[ModifiedResidueId]]:
    pattern = r"\((\w+)\)"
    modified_residues: List[ModifiedResidueId] = []
    matches = re.finditer(pattern, squence_str)
    parts = []
    ori_index = 0
    index = 0
    for match_patt in matches:

        parts.append(squence_str[ori_index : match_patt.start()])

        modified_residue = match_patt.group(1)
        ccd_result = CCD_NAME_TO_ONE_LETTER.get(modified_residue, None)
        if ccd_result:
            is_modified = True
            ori_residue = ccd_result

            if is_modified:
                one_letter_code = ori_residue
        else:
            one_letter_code = modified_residue

        ori_index = match_patt.end()

        index += len(parts[-1]) + 1
        parts.append(one_letter_code)
        modified_residues.append((index, modified_residue, one_letter_code))
    parts.append(squence_str[ori_index:])
    return "".join(parts), modified_residues


def sequence_to_ptm_list(sequence: str):
    """
    Convert a sequence of PTMs to a list of PTMs.
    """
    return parse_modified_sequence(sequence)


def ptm_list_to_origin_sequence(
    ptm_list: List[str], peptide_mask_indexes: NDArray = None
):
    """
    Convert a list of PTMs to a sequence of PTMs.
    """
    ptms = []
    one_letter_names = []
    ptm_index = 0
    for i, ptm in enumerate(ptm_list):
        if not ptm == "_":
            ptm_index += 1
        else:
            continue

        if len(ptm) > 1:
            if ptm not in CCD_NAME_TO_ONE_LETTER:
                raise ValueError(f"Unknown PTM: {ptm}")
            one_letter_name = CCD_NAME_TO_ONE_LETTER[ptm]
            one_letter_names.append(one_letter_name)
            if peptide_mask_indexes is None or peptide_mask_indexes[i] == 0:
                ptms.append((ptm_index, str(ptm), one_letter_name))

        else:
            one_letter_names.append(str(ptm))

    return "".join(one_letter_names), ptms


def ptm_list_to_extend_sequence(
    ptm_list: List[str], peptide_mask_indexes: NDArray = None
):
    """
    Convert a list of PTMs to a sequence of PTMs.
    """
    ptms = []
    one_letter_names = []
    ptm_index = 0
    for i, ptm in enumerate(ptm_list):

        ptm_index += 1

        if len(ptm) > 1:
            if ptm not in CCD_NAME_TO_ONE_LETTER:
                raise ValueError(f"Unknown PTM: {ptm}")
            one_letter_name = CCD_NAME_TO_ONE_LETTER[ptm]
            one_letter_names.append(one_letter_name)
            if peptide_mask_indexes is None or peptide_mask_indexes[i] == 0:
                ptms.append((ptm_index, str(ptm), one_letter_name))

        else:
            one_letter_names.append(str(ptm))

    return "".join(one_letter_names), ptms


def ptm_list_to_sequence_list(sequence: str, ptm_list: List[Tuple[int, str, str]]):
    """
    Convert a list of PTMs to a sequence of PTMs.
    """
    res_list = list(sequence)
    for ptm in ptm_list:
        res_list[ptm[0] - 1] = f"{ptm[1]}"

    return res_list


def ptm_list_to_sequence(sequence: str, ptm_list: List[Tuple[int, str, str]]):
    """
    Convert a list of PTMs to a sequence of PTMs.
    """
    res_list = list(sequence)
    for ptm in ptm_list:
        res_list[ptm[0] - 1] = f"({ptm[1]})"

    return "".join(res_list)


def get_fixed_ptm_list(
    ptm_list: List[Tuple[int, str, str]], peptide_mask_indexes: NDArray
):
    fixed_ptm_list = []
    for ptm in ptm_list:
        if peptide_mask_indexes[ptm[0] - 1] == 1:
            fixed_ptm_list.append(ptm)
    return fixed_ptm_list


def list_to_sequence(list):
    """
    Convert a list to a sequence.
    """
    return "".join(list)
