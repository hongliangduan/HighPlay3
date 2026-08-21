import copy
import os
import numpy as np
import torch
from boltz.data import const
import json
import pickle as pkl
import tempfile

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import typing
from json import JSONDecodeError
from pathlib import Path
from time import time
from typing import Final, List, cast, Literal, Optional
from boltz.data.pad import pad_to_max
from boltz_finetune.utils.infer_utils import (
    get_pred_writer,
    get_model_module,
)
from Bio.PDB import PDBIO, Model, PDBParser, Selection
import pytorch_lightning as pl
from loguru import logger
from boltz.data.types import (
    Manifest,
    Input,
    StructureV2,
    BondV2,
    Residue,
    Chain,
    Ensemble,
    Record,
    StructureInfo,
    ChainInfo,
)
from move_file import relax_files
from pytorch_lightning import Trainer, seed_everything
from boltz_finetune.utils.input_utils import (
    mock_peptiede_boltz_data,
    handle_template,
    get_boltz_input,
    DEFAULT_API_SERVER,
    ModifiedResidueId,
    BondAtomId,
)
from boltz.data.parse.schema import parse_boltz_schema
from boltz.data.mol import load_canonicals, load_molecules
from boltz_finetune.utils.data_utils import mock_sequence_id, ChainType
from pytorch_lightning import LightningModule, Trainer
from torch import Tensor
from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
from boltz.data.feature.featurizerv2 import Boltz2Featurizer
from torch.utils.data import DataLoader

BUCKETS: Final[List[int]] = [
    256,
]


def collate(data: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate the data.

    Parameters
    ----------
    data : List[Dict[str, Tensor]]
        The data to collate.

    Returns
    -------
    Dict[str, Tensor]
        The collated data.

    """
    # Get the keys
    keys = data[0].keys()

    # Collate the data
    collated = {}
    for key in keys:
        values = [d[key] for d in data]

        if key not in [
            "all_coords",
            "all_resolved_mask",
            "crop_to_all_atom_map",
            "chain_symmetries",
            "amino_acids_symmetries",
            "ligand_symmetries",
            "record",
            "affinity_mw",
        ]:
            # Check if all have the same shape
            shape = values[0].shape
            if not all(v.shape == shape for v in values):
                values, _ = pad_to_max(values, 0)
            else:
                values = torch.stack(values, dim=0)

        # Stack the values
        collated[key] = values

    return collated


class PlayDataset(torch.utils.data.Dataset):
    """Base iterable dataset."""

    def __init__(
        self,
        input: Input,
        mol_dir: Path,
    ) -> None:

        super().__init__()
        self.input = input

        self.mol_dir = mol_dir

        self.tokenizer = Boltz2Tokenizer()
        self.featurizer = Boltz2Featurizer()
        self.canonicals = load_canonicals(self.mol_dir)

    def __getitem__(self, idx: int) -> dict:
        """Get an item from the dataset.

        Returns
        -------
        Dict[str, Tensor]
            The sampled data features.

        """
        # Get record
        record = self.input.record

        # Tokenize structure
        try:
            tokenized = self.tokenizer.tokenize(self.input)
        except Exception as e:  # noqa: BLE001
            print(
                f"Tokenizer failed on {record.id} with error {e}. Skipping."
            )  # noqa: T201
            return self.__getitem__(0)

        # Load conformers
        try:
            molecules = {}
            molecules.update(self.canonicals)
            molecules.update(self.input.extra_mols)
            mol_names = set(tokenized.tokens["res_name"].tolist())
            mol_names = mol_names - set(molecules.keys())
            molecules.update(load_molecules(self.mol_dir, mol_names))
        except Exception as e:  # noqa: BLE001
            print(f"Molecule loading failed for {record.id} with error {e}. Skipping.")
            return self.__getitem__(0)

        # Inference specific options
        options = record.inference_options
        if options is None:
            pocket_constraints = None, None
        else:
            pocket_constraints = options.pocket_constraints

        # Get random seed
        seed = 42
        random = np.random.default_rng(seed)

        # Compute features
        try:
            features = self.featurizer.process(
                tokenized,
                molecules=molecules,
                random=random,
                training=False,
                max_atoms=None,
                max_tokens=None,
                max_seqs=const.max_msa_seqs,
                pad_to_max_seqs=False,
                single_sequence_prop=0.0,
                compute_frames=True,
                inference_pocket_constraints=pocket_constraints,
                compute_constraint_features=True,
                override_method=None,
                compute_affinity=False,
            )
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print(
                f"Featurizer failed on {record.id} with error {e}. Skipping."
            )  # noqa: T201
            return self.__getitem__(0)

        # Add record
        features["record"] = record
        return features

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.

        """
        return 1


class Boltz2PlayDataModule(pl.LightningDataModule):
    """DataModule for Boltz2 inference."""

    def __init__(
        self,
        input: Input,
        mol_dir: Path,
    ) -> None:

        super().__init__()
        self.input = input

        self.mol_dir = mol_dir
        self.dataset = PlayDataset(
            self.input,
            mol_dir=self.mol_dir,
        )

    def upate_input(self, input):
        self.input = input
        self.dataset = PlayDataset(
            self.input,
            mol_dir=self.mol_dir,
        )

    def predict_dataloader(self) -> DataLoader:
        """Get the training dataloader.

        Returns
        -------
        DataLoader
            The training dataloader.

        """

        return DataLoader(
            self.dataset,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            shuffle=False,
            collate_fn=collate,
            drop_last=False,
        )

    def transfer_batch_to_device(
        self,
        batch: dict,
        device: torch.device,
        dataloader_idx: int,  # noqa: ARG002
    ) -> dict:
        """Transfer a batch to the given device.

        Parameters
        ----------
        batch : Dict
            The batch to transfer.
        device : torch.device
            The device to transfer to.
        dataloader_idx : int
            The dataloader index.

        Returns
        -------
        np.Any
            The transferred batch.

        """
        for key in batch:
            if key not in [
                "all_coords",
                "all_resolved_mask",
                "crop_to_all_atom_map",
                "chain_symmetries",
                "amino_acids_symmetries",
                "ligand_symmetries",
                "record",
                "affinity_mw",
            ]:
                batch[key] = batch[key].to(device)
        return batch


class BoltzRunner:
    def __init__(self, model_dir: Path, output_dir: Path, data_dir: Path):
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.pred_writer = get_pred_writer(data_dir, output_dir, "pdb")

        self.trainer = Trainer(
            default_root_dir=output_dir,
            callbacks=[self.pred_writer],
            accelerator="gpu",
            devices=1,
            precision="bf16-mixed",
            strategy="auto",
            enable_progress_bar=False,
            logger=False,
            enable_model_summary=False,
        )
        self.model_module = get_model_module(self.model_dir)

    def run_predict(self, data_module: Boltz2PlayDataModule):
        self.trainer.predict(
            self.model_module,
            datamodule=data_module,
            return_predictions=False,
        )


def make_model_runner(model_dir: str, output_dir: Path, data_dir: Path):
    return BoltzRunner(Path(model_dir), output_dir, data_dir)


def make_base_input(
    task_id: str,
    sequences: list[str],
    save_data: bool,
    out_dir: Path,
    sequences_type: list[int] | None = None,
    mol_dir: Path = Path("/home/fuxin/.boltz/mols"),
    bonds: List[tuple[BondAtomId, BondAtomId]] = [],
):
    if sequences_type is None:
        sequences_type = [1] * len(sequences)

    ccd = load_canonicals(mol_dir)
    chain_data = []
    for i, sequence in enumerate(sequences):
        chain_data.append((mock_sequence_id(i), ChainType(sequences_type[i]), sequence))

    boltz_data = mock_peptiede_boltz_data(chain_data, {}, bonds)

    boltz_target = parse_boltz_schema(task_id, boltz_data, ccd, mol_dir, boltz_2=True)

    tempfile_obj = tempfile.TemporaryDirectory()
    tempfile_path = Path(tempfile_obj.name)
    env_dir = tempfile_path / "envs"

    handle_template(
        boltz_target,
        template_dict=None,
        use_template=True,
        env_dir=env_dir,
        ccd=ccd,
        mol_dir=mol_dir,
    )

    msa_dir = tempfile_path / "msa"

    input = get_boltz_input(
        boltz_target,
        msa_dir=msa_dir,
        use_msa_server=True,
        msa_server_url=DEFAULT_API_SERVER,
        msa_pairing_strategy="greedy",
        max_msa_seqs=8192,
        use_mock_msa=False,  # use_mock_msa=True,
    )
    if save_data:
        with open(out_dir / f"{task_id}_feature.pkl", "wb") as f:
            pkl.dump(input, f)
    tempfile_obj.cleanup()

    return input


def load_base_input(data_path: Path) -> Input:
    with open(data_path, "rb") as f:
        return pkl.load(f)


def merge_structure(
    receptor_structure: StructureV2,
    binder_structure: StructureV2,
    ligand_bonds: List[tuple[BondAtomId, BondAtomId]],
):
    merge_atoms = np.concatenate(
        [receptor_structure.atoms, binder_structure.atoms], axis=0
    )
    binder_chain_index = len(receptor_structure.chains)
    modified_bonds = []
    binder_atom_start = len(receptor_structure.atoms)
    binder_res_start = len(receptor_structure.residues)

    for bond in binder_structure.bonds:
        modified_bonds.append(
            (
                binder_chain_index,  # start chain index
                binder_chain_index,  # end chain index
                bond[2] + binder_res_start,  # start res index
                bond[3] + binder_res_start,  # end res index
                bond[4] + binder_atom_start,  # start atom index
                bond[5] + binder_atom_start,  # end atom index
                bond[6],  # bond type
            )
        )

    modified_residues = []
    for res in binder_structure.residues:
        modified_residues.append(
            (
                res[0],
                res[1],
                res[2],
                res[3] + binder_atom_start,
                res[4],
                res[5] + binder_atom_start,
                res[6] + binder_atom_start,
                res[7],
                res[8],
            )
        )
    modified_residues = np.array(modified_residues, dtype=Residue)
    merge_residues = np.concatenate(
        [receptor_structure.residues, modified_residues], axis=0
    )

    modified_chain = np.array(
        [
            (
                binder_structure.chains[0][0],
                binder_structure.chains[0][1],
                binder_chain_index,
                binder_structure.chains[0][3],
                binder_chain_index,
                binder_atom_start,
                len(binder_structure.atoms),
                binder_res_start,
                binder_structure.chains[0][8],
                binder_structure.chains[0][9],
            )
        ],
        Chain,
    )

    merge_chains = np.concatenate([receptor_structure.chains, modified_chain], axis=0)

    chain_ids = [chain[0] for chain in merge_chains]

    for ligand_bond in ligand_bonds:
        start_chain_index = chain_ids.index(ligand_bond[0][0])
        end_chain_index = chain_ids.index(ligand_bond[1][0])
        start_res_index = merge_chains[start_chain_index][7] + ligand_bond[0][1] - 1
        end_res_index = merge_chains[end_chain_index][7] + ligand_bond[1][1] - 1
        start_atoms = list(
            merge_atoms[
                merge_residues[start_res_index][3] : merge_residues[start_res_index][3]
                + merge_residues[start_res_index][4]
            ]
        )
        start_atoms_names = [atom[0] for atom in start_atoms]
        start_atom_index = (
            start_atoms_names.index(ligand_bond[0][2])
            + merge_residues[start_res_index][3]
        )
        end_atoms = list(
            merge_atoms[
                merge_residues[end_res_index][3] : merge_residues[end_res_index][3]
                + merge_residues[end_res_index][4]
            ]
        )
        end_atoms_names = [atom[0] for atom in end_atoms]
        end_atom_index = (
            end_atoms_names.index(ligand_bond[1][2]) + merge_residues[end_res_index][3]
        )
        modified_bonds.append(
            (
                start_chain_index,
                end_chain_index,
                start_res_index,
                end_res_index,
                start_atom_index,
                end_atom_index,
                5,
            )
        )
    modified_bonds = np.array(modified_bonds, dtype=BondV2)
    merge_bonds = np.concatenate([receptor_structure.bonds, modified_bonds], axis=0)
    merge_coords = np.concatenate(
        [receptor_structure.coords, binder_structure.coords], axis=0
    )
    merge_mask = np.concatenate(
        [receptor_structure.mask, binder_structure.mask], axis=0
    )
    merge_ensemble = np.array([(0, len(merge_coords))], Ensemble)
    return StructureV2(
        atoms=merge_atoms,
        bonds=merge_bonds,
        residues=merge_residues,
        chains=merge_chains,
        interfaces=receptor_structure.interfaces,
        coords=merge_coords,
        mask=merge_mask,
        ensemble=merge_ensemble,
    )


def merge_recored(receptor_record: Record, binder_record: Record):
    merge_id = receptor_record.id + "_complex"
    merge_structure_info = StructureInfo(
        resolution=receptor_record.structure.resolution,
        method=receptor_record.structure.method,
        deposited=receptor_record.structure.deposited,
        released=receptor_record.structure.released,
        num_chains=receptor_record.structure.num_chains + 1,
        num_interfaces=receptor_record.structure.num_interfaces,
        pH=receptor_record.structure.pH,
        temperature=receptor_record.structure.temperature,
    )
    merge_chains = [*receptor_record.chains]
    last_chain_id = receptor_record.chains[-1].entity_id + 1
    for chain in binder_record.chains:
        merge_chains.append(
            ChainInfo(
                chain.chain_id + last_chain_id,
                chain.chain_name,
                chain.mol_type,
                chain.cluster_id,
                chain.msa_id,
                chain.num_residues,
                chain.valid,
                chain.entity_id + last_chain_id,
            )
        )

    return Record(
        id=merge_id,
        structure=merge_structure_info,
        chains=merge_chains,
        interfaces=receptor_record.interfaces,
        inference_options=receptor_record.inference_options,
        templates=receptor_record.templates,
    )


def merge_input(
    receptor_input: Input,
    binder_input: Input,
    ligand_bonds: List[tuple[BondAtomId, BondAtomId]],
):

    # merge structure
    merged_structure = merge_structure(
        receptor_input.structure, binder_input.structure, ligand_bonds
    )

    # merge record
    merged_record = merge_recored(receptor_input.record, binder_input.record)

    # merge msa
    merge_msa = {**(receptor_input.msa)}
    last_chain_id = receptor_input.record.chains[-1].entity_id + 1
    for chain in binder_input.record.chains:
        merge_msa[chain.entity_id + last_chain_id] = binder_input.msa[chain.entity_id]

    return Input(
        structure=merged_structure,
        msa=merge_msa,
        record=merged_record,
        residue_constraints=receptor_input.residue_constraints,
        templates=receptor_input.templates,
        extra_mols={**receptor_input.extra_mols, **binder_input.extra_mols},
    )


def add_binder_input(
    task_id: str,
    receptor_input: Input,
    binder_sequence: str,
    modified_info: List[ModifiedResidueId],
    transed_bond_pairs: List[tuple[BondAtomId, BondAtomId]],
    ligand_bonds: List[tuple[BondAtomId, BondAtomId]],
    out_dir: Path,
    mol_dir: Path = Path("/home/fuxin/.boltz/mols"),
):
    receptor_chain_num = len(receptor_input.record.chains)
    ccd = load_canonicals(mol_dir)
    binder_id = mock_sequence_id(receptor_chain_num)
    chain_data = [(binder_id, ChainType(1), binder_sequence)]
    bonds = []
    for transed_bond_pair in transed_bond_pairs:
        bonds.append(
            (
                (
                    binder_id,
                    transed_bond_pair[0][0] + 1,
                    transed_bond_pair[0][1],
                ),
                (
                    binder_id,
                    transed_bond_pair[1][0] + 1,
                    transed_bond_pair[1][1],
                ),
            )
        )
    boltz_data = mock_peptiede_boltz_data(
        chain_data, {receptor_chain_num: modified_info}, bonds, is_cyclic=False
    )

    boltz_target = parse_boltz_schema(task_id, boltz_data, ccd, mol_dir, boltz_2=True)
    temp_dir_obj = tempfile.TemporaryDirectory()
    msa_dir = Path(temp_dir_obj.name)
    if not msa_dir.exists():
        msa_dir.mkdir(parents=True, exist_ok=True)

    input = get_boltz_input(
        boltz_target,
        msa_dir=msa_dir,
        use_msa_server=True,
        msa_server_url=DEFAULT_API_SERVER,
        msa_pairing_strategy="greedy",
        max_msa_seqs=8192,
        use_mock_msa=True,  # use_mock_msa=True,
    )
    merged_input = merge_input(
        receptor_input=receptor_input, binder_input=input, ligand_bonds=ligand_bonds
    )
    structure_dir = (
        out_dir
        / merged_input.record.id
        / "boltz_data"
        / "processed"
        / "structures"
        / f"{merged_input.record.id}.npz"
    )
    structure_dir.parent.mkdir(parents=True, exist_ok=True)
    structure = merged_input.structure
    structure.dump(structure_dir)
    temp_dir_obj.cleanup()
    return merged_input


def get_data_moudle(input: Input, mol_dir: Path = Path("/home/fuxin/.boltz/mols")):
    return Boltz2PlayDataModule(
        input,
        mol_dir,
    )


def read_json_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print("错误: 文件未找到!")
    except json.JSONDecodeError:
        print("错误: 无法解析 JSON 数据!")
    except Exception as e:
        print(f"错误: 发生了一个未知错误: {e}")
    return None


def get_botlz_prediction(
    receptor_input,
    peptide_sequence,
    model_runner: BoltzRunner,
    temp_out_dir,
    bonds,
    ptms,
    ligand_bonds=[],
):
    merge_input = add_binder_input(
        "t1",
        receptor_input,
        peptide_sequence,
        ptms,
        bonds,
        ligand_bonds,
        temp_out_dir,
    )
    data_moudle = get_data_moudle(merge_input)
    model_runner.run_predict(data_moudle)
    return merge_input.record.chains[-1].chain_name


if __name__ == "__main__":
    sequences = [
        "MYDFTNCDFEKIKAAYLSTISKDLITYMSGTKSTEFNNTVSCSNRPHCLTEIQSLTFNPTAGCASLAKEMFAMKTKAALAIWCPGYSETQINATQAMKKRRKRKVTTNKCLEQVSQLQGLWRRFNRPLLKQQ"
    ]
    global complex_input
    complex_input = make_base_input(
        "tlsp_complex",
        sequences + ["CKLMNPQRSTVWC"],
        mol_dir=Path("/home/fuxin/.boltz/mols"),
        out_dir=Path("/home/fuxin/HL/wwt/boltz_play/output/TLSP_complex"),
        bonds=[(("B", 1, "SG"), ("B", 13, "SG"))],
    )

    inputs = make_base_input(
        "tlsp",
        sequences,
        mol_dir=Path("/home/fuxin/.boltz/mols"),
        out_dir=Path("/home/fuxin/HL/wwt/boltz_play/output/TLSP"),
    )
