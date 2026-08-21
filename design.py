import os
from pathlib import Path
from time import time

import numpy as np
from absl import app, flags
from boltz_utils import read_json_file
from loguru import logger

from boltz_utils import BUCKETS, load_base_input, make_base_input, make_model_runner
from pre import (
    CC_index,
    dump_config_to_yaml,
    get_emphasize_locked_sequence_str,
    get_locked_mask_from_flag,
    get_locked_mask_from_seq,
    is_init_peptide_sequence_valid,
    is_peptide_sequence_valid,
    is_peptide_with_mask,
    parse_pdb_file,
    random_initialize_weights,
    read_config_from_yaml,
)
from ptm_utils import sequence_to_ptm_list
from train import TrainPipeline

flags.DEFINE_string("config_path", "evo_conf.yaml", "config file path")

FLAGS = flags.FLAGS


def main(argv):

    if not os.path.exists(FLAGS.config_path):
        raise ValueError("config yaml file not exist")
    config = read_config_from_yaml(FLAGS.config_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{config['gpu_index']}"
    out_dir = config["output_dir"] + config["receptor_name"]
    logger.info(f"Config is : {FLAGS.config_path}")
    logger.info(f"Output dir is : {out_dir}")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    yaml_file = Path(out_dir) / f"{config['receptor_name']}_config.yaml"
    dump_config_to_yaml(config, yaml_file)

    if config["use_fixed_group"]:
        assert (
            len(config["fixed_group"]) == config["cys_num"]
        ), "fixed_group length not equal to cys_num"

    peptide_sequence = config["initial_ligand_seq"]
    peptide_length = config["peptide_length"]
    peptide_mask_indexes = get_locked_mask_from_flag(
        peptide_length, config["ligand_seq_locked_mask"]
    )
    peptide_sequence_has_mask = is_peptide_with_mask(peptide_sequence)
    init_peptide_seq, init_ptms = sequence_to_ptm_list(peptide_sequence)

    # get flag for if use ptm
    ptms_path = Path(config["user_restypes"])
    ptms = [] if not config["use_ptms"] else read_json_file(ptms_path)["ptms"]
    use_ptms = len(ptms) > 0 and config["use_ptms"]
    if not use_ptms:
        ptms = []
    # get init locked mask
    if peptide_sequence_has_mask:
        peptide_mask_indexes = peptide_mask_indexes | get_locked_mask_from_seq(
            peptide_length, init_peptide_seq
        )
    is_nc_cyclic = config["is_nc_cyclic"]
    assert is_init_peptide_sequence_valid(
        peptide_mask_indexes,
        init_peptide_seq,
        peptide_length,
        config["cys_num"],
        is_nc_cyclic,
    ), "Input initial peptide sequence can not get valid sequence under input peptide_mask_indexes."

    if config["random_init_ligand_seq"]:
        init_peptide_seq, random_ptms = random_initialize_weights(
            peptide_length,
            peptide_mask_indexes,
            config["cys_num"],
            init_peptide_seq,
            is_nc_cyclic,
            use_ptms=use_ptms,
            ptms=ptms,
        )
        init_ptms.extend(random_ptms)
    else:
        if peptide_sequence_has_mask:
            raise ValueError(
                "Fixed peptide_sequence with mask 'X' but not allow random init."
            )

        if not is_peptide_sequence_valid(
            init_peptide_seq, config["cys_num"], is_nc_cyclic=is_nc_cyclic
        ):
            raise ValueError(
                "Fixed peptide_sequence is not valid but not allow random init."
            )

    receptor_name = config["receptor_name"]

    log_dir = Path(out_dir + "/log/")
    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    logger.add(
        f"{out_dir}/log/{str(time())}_{receptor_name}_{peptide_length}.log",
        format="{time} | {file} | {line} | {level} | {message}",
        level="INFO",
        colorize=False,
    )

    # merget CC index and locked mask
    if not is_nc_cyclic:
        indexes_of_c = CC_index(init_peptide_seq, get_all=True)
        logger.opt(colors=True).info(f"CC index: <r>{indexes_of_c}</r>")
        peptide_mask_indexes[indexes_of_c] = 1

    logger.opt(colors=True).info(
        f"Init seq: {get_emphasize_locked_sequence_str(init_peptide_seq, peptide_mask_indexes,init_ptms)}",
    )
    start_time = time()

    model_runner = make_model_runner(
        config["model_path"], Path(out_dir) / "temp", Path(out_dir) / "temp"
    )
    # model_runner = None
    padding_length = max(int(config["padding_length"]), BUCKETS[0])
    BUCKETS[0] = padding_length

    if config["receptor_data"] is None or not Path(config["receptor_data"]).exists():
        logger.warning(
            f"Receptor data path {config['receptor_data']} does not exist or null , will create new input."
        )
        receptor_seq = config["receptor_seq"]
        if isinstance(receptor_seq, str):
            receptor_seq = [receptor_seq]
        elif isinstance(receptor_seq, list):
            receptor_seq = [seq for seq in receptor_seq if seq.strip()]
        receptor_input = make_base_input(
            config["receptor_name"],
            receptor_seq,
            True,
            Path(out_dir),
            config["receptor_type"],
        )

    else:
        receptor_data_path = Path(config["receptor_data"])
        receptor_input = load_base_input(receptor_data_path)
        if receptor_input is None:
            logger.error(f"Failed to load receptor input from {receptor_data_path}")
            return
    ligand_bonds = None
    if "ligand_bonds" in config:
        ligand_bonds = [
            tuple(tuple(inner_list) for inner_list in item)
            for item in config["ligand_bonds"]
        ]

    training_pipeline = TrainPipeline(
        init_seq=init_peptide_seq,
        receptor_input=receptor_input,
        model_runner=model_runner,
        peptide_locked_mask=peptide_mask_indexes,
        pocket=config["pocket"],
        output_dir=out_dir,
        num_iterations=config["num_iterations"],
        plDDT_only=config["plDDT_only"],
        only_loss=config["only_loss"],
        jumpout_num=config["jumpout_num"],
        distance_constraints=config["distance_constraints"],
        max_extend_length=config["max_ligand_extend_length"],
        init_model=config["init_model"],
        cc_num=config["cys_num"],
        use_fixed_group=config["use_fixed_group"],
        fixed_group=config["fixed_group"],
        is_nc_cyclic=is_nc_cyclic,
        use_ptms=use_ptms,
        ptms=ptms,
        init_ptms=init_ptms,
        max_run_time=config["max_run_time"],
        ligand_bonds=ligand_bonds,
    )
    training_pipeline.run()
    logger.info(f"Total time: {time() - start_time}s")


if __name__ == "__main__":
    app.run(main)
