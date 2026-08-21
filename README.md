# HighPlay3

## Getting started / installation

To get started using HighPlay3, clone the repo:

   git clone [https://github.com/hongliangduan/HighPlay3](https://github.com/hongliangduan/HighPlay3)

## Environment Setup

Create and activate a new conda environment:

   conda create -n highplay3 python=3.11.13

   conda activate highplay3

Install the required Python packages:

   pip install -r requirements.txt

Because this algorithm uses **Boltz2** to evaluate peptide–target complex structures and uses **Multi_CycGT** to predict cyclic peptide permeability, both packages must also be installed in the same environment.

Please refer to the following repositories for installation instructions:

* Multi_CycGT: [https://github.com/hongliangduan/Multi_CycGT](https://github.com/hongliangduan/Multi_CycGT)

* Boltz2: [https://github.com/jwohlwend/boltz](https://github.com/jwohlwend/boltz)

## Usage

### Design Configuration Parameters

Before running the design pipeline, prepare a YAML configuration file. You may refer to `config.yaml` for a complete example.

Important parameters include:

#### Name of the target receptor

receptor_name

#### Amino acid sequence of the target receptor

receptor_seq

#### Binding pocket information of the target receptor (optional)

pocket

#### Peptide design range, peptide_length-max_ligand_extend_length

peptide_length

max_ligand_extend_length

#### Output directory for design results

output_dir

### Start Peptide Design

Run the design pipeline using:

   python design.py --config_path your_conf.yaml

The generated results will be saved in the output directory specified by `output_dir`.


### Screening

After the design process is completed, candidate sequences that satisfy the initial screening criteria will be collected in:

   high.csv

By default, the initial screening criteria are:

Affinity Score > 0.7

Permeability Score > 0.7

These candidates are considered preliminary hits and should be further evaluated using additional structure- and physics-based screening methods.


After the initial screening, additional filtering is recommended using tools such as **PLIP**, **Rosetta**, and **molecular dynamics simulations**.

Recommended post-screening steps include:

1. **Interaction analysis using PLIP**

   Identify key interactions between the designed cyclic peptide and the target protein, including hydrogen bonds, hydrophobic contacts, salt bridges, and π-related interactions.

2. **Interface energy evaluation using Rosetta**

   Evaluate the peptide–protein binding interface and calculate interface energy-related metrics.

3. **Molecular dynamics simulation**

   Assess the structural stability of the peptide–target complex under dynamic conditions.

Please refer to the official documentation of each tool for installation and usage:

* PLIP: [https://github.com/pharmai/plip](https://github.com/pharmai/plip)

* Rosetta: [https://docs.rosettacommons.org](https://docs.rosettacommons.org)

* AMBER: [https://ambermd.org/doc12/Amber25.pdf](https://ambermd.org/doc12/Amber25.pdf)
