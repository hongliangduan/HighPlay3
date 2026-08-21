import shutil
import subprocess
from pathlib import Path
import csv
import pandas as pd


def read_csv_top(csv_path: Path, top: int):
    with open(csv_path, "r") as f:
        lines = f.readlines()
    result_list = [line.strip().split(",") for line in lines[1:]]
    # sort by the second column
    result_list.sort(key=lambda x: float(x[6]), reverse=True)

    return result_list[0:top]


def move_from_csv(task_path: Path, top: int):
    task_tag = task_path.stem
    csv_path = task_path / f"{task_tag}_stats.csv"
    result_path = task_path / "result"
    target_path = task_path / f"top_{top}"
    if not target_path.exists():
        target_path.mkdir(parents=True)

    result_list = read_csv_top(csv_path, top)
    target_csv = target_path / f"stats_top_{top}.csv"
    with open(target_csv, "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "i_plddt",
                "peptide_sequence",
                "peptide_sequence_ptm",
                "ipae",
                "iptm",
                "hotspot_distance",
                "reward",
                "interface_sc",
                "interface_dG",
                "interface_dSASA",
                "interface_dG_SASA_ratio",
                "interface_nres",
                "interface_interface_hbonds",
                "cyclic_norm",
                "ptm_num",
                "pdb_file",
                "group",
            ]
        )
        for result in result_list:
            writer.writerow(result)
            pdb_id = result[15]
            pdb_path = result_path / f"{pdb_id}.pdb"
            if pdb_path.exists():
                shutil.copy(pdb_path, target_path)
            else:
                print(f"{pdb_id} not found")


def move_from_hf(source_dir: Path, task_tag: str):

    source_pbds = source_dir.rglob("*.pdb")
    pdb_tag = "_relaxed_rank_001_"
    target_path = Path(f"/data/wwt/highplay3/output/{task_tag}/hf_top/")
    if not target_path.exists():
        target_path.mkdir(parents=True)
    for source_pdb_flie in source_pbds:
        if pdb_tag not in source_pdb_flie.stem:
            continue
        new_name = "_".join(source_pdb_flie.stem.split("_")[:2])
        shutil.copy(source_pdb_flie, target_path / f"{new_name}.pdb")


def relax_files(target_path: Path):
    commd = f"conda run -n af2 colabfold_relax -d {str(target_path)} -o {str(target_path)}_relaxed"
    subprocess.run(commd, shell=True)
    pass


def move_from_names(file_names: list[str], task_tag: Path):
    pdb_path = task_tag / "result"
    file_paths = [pdb_path / f"{name}.pdb" for name in file_names]
    target_path = task_tag / "top_select"
    if not target_path.exists():
        target_path.mkdir(parents=True)
    for pdb_file in file_paths:
        shutil.copy(pdb_file, target_path / f"{pdb_file.stem}.pdb")


if __name__ == "__main__":
    # move_from_csv(Path("/home/fuxin/HL/wwt/highplay3/output/TLSP/TLSP_PTM_730"), 30)

    # target_files = ['af3_10048', 'af3_20441', 'af3_20427', 'af3_20440', 'af3_10047', 'af3_20420', 'af3_737', 'af3_20439', 'af3_20416', 'af3_20436', 'af3_20415', 'af3_20418', 'af3_20419', 'af3_730', 'af3_704', 'af3_10051', 'af3_742', 'af3_20450', 'af3_20413', 'af3_20437', 'af3_805', 'af3_705', 'af3_36', 'af3_10046', 'af3_809', 'af3_793', 'af3_787', 'af3_20491', 'af3_20013', 'af3_723', 'af3_20410', 'af3_10050', 'af3_20501', 'af3_849', 'af3_20498', 'af3_20513', 'af3_20399', 'af3_20516', 'af3_20506', 'af3_20432', 'af3_20011']

    df = pd.read_csv('select.csv')
    target_files = [i.split('.')[0] for i in df['file'].tolist()]
    move_from_names(
        target_files, Path("/home/fuxin/HL/wwt/highplay3/output/TL1A/TL1A_H3")
    )
    relax_files(Path("/home/fuxin/HL/wwt/highplay3/output/TL1A/TL1A_H3/top_select"))
