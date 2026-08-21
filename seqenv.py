import copy
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from boltz_utils import BoltzRunner, Input
from loguru import logger
from numpy.typing import NDArray

from force_distance_constraint import DistanceConstraint
from pre import (
    CYCLIC_PTMS,
    EXTENDED_RESTYPES,
    RESTYPES,
    CC_index,
    get_availables,
    get_cc_groups,
    get_emphasize_locked_sequence_str,
    groups_predict_cycle,
    onehot_to_sequence,
    predict_cycle,
    sequence_to_onehot,
    onehot_to_sequence_with_extend,
)
from ptm_utils import ptm_list_to_sequence

playout_dict = {}
init_dict = {}
move_dict = {}
count = 0
count2 = 20000


class Seqenv:
    """board for the game"""

    def __init__(
        self,
        init_seq: str,
        receptor_input: Input,
        peptide_locked_mask: NDArray,
        pocket: Dict[str, List[int]],
        output_dir: str,
        onlyplddt: bool,
        only_loss: bool,
        enable_extend: bool,
        max_extend_length: int,
        distance_constraints: List[List[Any]],
        cc_num: int = 2,
        use_fixed_group: bool = False,
        fixed_group: List[int] = None,
        gpu_id: int = 0,
        is_nc_cyclic: bool = False,
        model_runner: BoltzRunner = None,
        use_ptms: bool = False,
        ptms: List[str] = [],
        init_ptms: List[tuple[int, str, str]] = [],
        ligand_bonds=None,
    ):

        self.peptide_locked_mask = peptide_locked_mask
        self.receptor_input = receptor_input
        self.model_runner = model_runner
        self.use_ptms = use_ptms
        self.ptms = ptms
        self.init_ptms = init_ptms
        restypes = [] + RESTYPES
        if self.use_ptms:
            restypes += self.ptms
        # self.states = []
        self.peptide_len = len(init_seq)

        self.output_dir = output_dir
        output_dir: Path = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        self.onlyplddt = onlyplddt
        self.only_loss = only_loss
        self.repeated = False

        self.init_state_count = 0

        self.pocket = pocket

        self.index = 10000
        #self.reward = 0.0
        self.reward_dim = 2
        self.reward = np.zeros(self.reward_dim, dtype=np.float32)
        self.seqs = []

        self.enable_extend = enable_extend
        self.max_extend_length = max_extend_length

        # distance_constraints
        self.distance_constraints = [
            DistanceConstraint(*distance_constaint)
            for distance_constaint in distance_constraints
        ]

        self.start_seq = init_seq
        # extend with '_' from last unlocked char.

        if enable_extend:
            restypes += ["_"]
            split_index = np.flatnonzero(self.peptide_locked_mask == 0)[-1]

            last_unlocked_char = self.start_seq[: split_index + 1]
            # extend start seq  EL*21 or L*20
            self.start_seq = self.start_seq.replace(
                last_unlocked_char,
                last_unlocked_char + (self.max_extend_length - self.peptide_len) * "_",
            )

            # extended mask EL*1
            new_locked_mask = np.zeros(max_extend_length, dtype=np.int32)
            new_locked_mask[: split_index + 1] = self.peptide_locked_mask[
                : split_index + 1
            ]
            new_locked_mask[(split_index - self.peptide_len) :] = (
                self.peptide_locked_mask[split_index:]
            )
            self.peptide_locked_mask = new_locked_mask
            extend_init_ptms = []
            for ptm in self.init_ptms:
                ptm_index = ptm[0] - 1
                ptm_res = ptm[1]
                ptm_res_one = ptm[2]
                if ptm_index <= split_index:
                    extend_init_ptms.append(ptm)
                elif ptm_index > split_index:
                    extend_init_ptms.append(
                        (
                            ptm_index + self.max_extend_length - self.peptide_len + 1,
                            ptm_res,
                            ptm_res_one,
                        )
                    )
            self.init_ptms = extend_init_ptms
        self.restypes = restypes
        self.c_index_horizontal = [self.restypes.index("C")]
        if use_ptms:
            for sg_res in CYCLIC_PTMS:
                self.c_index_horizontal.append(self.restypes.index(sg_res))

        self.init_state = sequence_to_onehot(
            self.start_seq,
            self.restypes,
            use_ptms=self.use_ptms,
            init_ptms=self.init_ptms,
        )

        self.previous_init_state = self.init_state

        self.cc_num = cc_num
        self.use_fixed_group = use_fixed_group
        self.fixed_group = fixed_group
        if not is_nc_cyclic:
            C2, C1 = CC_index(init_seq)
            # C1 C2 index in the init sequence. after extend C1 C2 will be changed
            self.cyclic_head = C1
            self.cyclic_tail = C2
        else:
            self.c_index_vertical = []
            self.cyclic_head = 0
            self.cyclic_tail = len(self.start_seq) - 1

        self.c_index_vertical = CC_index(self.start_seq, get_all=True)

        self.gpu_id = gpu_id
        self.is_nc_cyclic = is_nc_cyclic
        self.ligand_bonds = ligand_bonds

        ##zrj,20260601
        self.global_best_reward = np.zeros(self.reward_dim, dtype=np.float32)
        self.global_best_state = copy.deepcopy(self.init_state)

    @classmethod
    def copy(cls, seq_env: "Seqenv"):
        """Create a copy of the Seqenv instance."""
        new_env = cls(
            init_seq=seq_env.start_seq,
            receptor_input=seq_env.receptor_input,
            peptide_locked_mask=seq_env.peptide_locked_mask,
            pocket=seq_env.pocket,
            output_dir=seq_env.output_dir,
            onlyplddt=seq_env.onlyplddt,
            only_loss=seq_env.only_loss,
            enable_extend=seq_env.enable_extend,
            max_extend_length=seq_env.max_extend_length,
            distance_constraints=[list(dc) for dc in seq_env.distance_constraints],
            cc_num=seq_env.cc_num,
            use_fixed_group=seq_env.use_fixed_group,
            fixed_group=seq_env.fixed_group,
            gpu_id=seq_env.gpu_id,
            is_nc_cyclic=seq_env.is_nc_cyclic,
            use_ptms=seq_env.use_ptms,
            ptms=seq_env.ptms,
            init_ptms=seq_env.init_ptms,
            ligand_bonds=seq_env.ligand_bonds,
        )
        # mutable need copy
        new_env._state = copy.deepcopy(seq_env._state)
        new_env.init_state = copy.deepcopy(seq_env.init_state)
        new_env.previous_init_state = copy.deepcopy(seq_env.previous_init_state)
        new_env.seqs = copy.deepcopy(seq_env.seqs)
        new_env.availables = copy.deepcopy(seq_env.availables)

        # global stable state not need copy
        new_env.index = seq_env.index
        new_env.reward = seq_env.reward

        new_env.model_runner = seq_env.model_runner
        new_env.previous_reward = seq_env.previous_reward
        new_env.unuseful_move = seq_env.unuseful_move
        new_env.restypes = seq_env.restypes
        new_env.c_index_horizontal = seq_env.c_index_horizontal
        new_env.c_index_vertical = seq_env.c_index_vertical

        return new_env

    def init_seq_state(self):
        self.repeated = False
        self.previous_reward = -float("inf")

        self.unuseful_move = 0
        self._state = copy.deepcopy(self.init_state)

        self.c_index_vertical = CC_index(self.start_seq, get_all=True)
        self.availables = get_availables(
            self._state,
            self.peptide_locked_mask,
            self.enable_extend,
            self.peptide_len,
            self.c_index_horizontal,
            self.c_index_vertical,
            is_nc_cyclic=self.is_nc_cyclic,
        )

        combo, ptms = onehot_to_sequence(self._state, self.restypes)
        combo_ptm = ptm_list_to_sequence(combo, ptms)
        self.seqs.append(combo_ptm)

        # get new C1 C2 index in the init sequence
        if not self.is_nc_cyclic:
            cc_indexes = CC_index(combo, get_all=True)
            self.cyclic_tail, self.cyclic_head = cc_indexes[-1], cc_indexes[0]
            groups = get_cc_groups(cc_indexes, self.use_fixed_group, self.fixed_group)
        pdb_str = ""
        if combo_ptm not in init_dict.keys():
            if self.cc_num >= 3:
                plddt, ipae, iptm, hotspot_distance, score, permeability, group_str = (
                    groups_predict_cycle(
                        self.receptor_input,
                        self.model_runner,
                        combo,
                        self.output_dir,
                        self.pocket,
                        self.index,
                        self.distance_constraints,
                        cc_groups=groups,
                        ptms=ptms,
                        ligand_bonds=self.ligand_bonds,
                    )
                )
                pdb_str = group_str
            else:
                plddt, ipae, iptm, hotspot_distance, score, permeability = predict_cycle(
                    self.receptor_input,
                    self.model_runner,
                    combo,
                    self.output_dir,
                    self.pocket,
                    self.index,
                    self.distance_constraints,
                    self.cyclic_head,
                    self.cyclic_tail,
                    is_nc_cyclic=self.is_nc_cyclic,
                    ptms=ptms,
                    ligand_bonds=self.ligand_bonds,
                )

            if self.onlyplddt:
                reward = plddt / 100

            elif self.only_loss:
                reward = score
            else:
                reward = plddt * 0.02 + 1 / (abs(hotspot_distance - 3.5) + 2.0)
                reward = reward / 3.0

            self.reward = reward

            init_dict[combo_ptm] = [
                plddt,
                ipae,
                iptm,
                hotspot_distance,
                permeability,
                reward,
                f"{self.index}{pdb_str}",
            ]
            self.index = self.index + 1

        else:
            plddt = init_dict[combo_ptm][0]
            ipae = init_dict[combo_ptm][1]
            iptm = init_dict[combo_ptm][2]
            hotspot_distance = init_dict[combo_ptm][3]
            permeability = init_dict[combo_ptm][4]
            reward = init_dict[combo_ptm][5]
            self.reward = reward

        logger.opt(colors=True).info(
            "<green>_____________________________init_seq_state_____________________________</green>"
        )
        extend = ""
        extend_num = extend_num = len(combo) - self.peptide_len
        if extend_num > 0:
            extend = f" <green>Extended:</green> <red>{extend_num}</red>"
        logger.opt(colors=True).info(
            f"Start Sequence: {get_emphasize_locked_sequence_str(combo, self.peptide_locked_mask,ptms)} {extend}",
        )
        self.previous_init_state = copy.deepcopy(self._state)

        ##zrj,20260601
        self.global_best_reward = copy.deepcopy(self.reward)
        self.global_best_state = copy.deepcopy(self.init_state)

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height
        """
        # square_state = self.states[-1]
        square_state = self._state
        return square_state

    def do_move(self, move, playout=0):
        """make the move for current player"""
        self.previous_reward = self.reward

        width = len(self.restypes)

        one_dim = move // width
        two_dim = move % width
        # new_state[one_dim,:] = 0
        # new_state[one_dim,two_dim] = 1

        if self._state[one_dim, two_dim] == 1:
            self.unuseful_move = 1
            self.reward = np.zeros(self.reward_dim, dtype=np.float32)
        else:
            self._state[one_dim, :] = 0
            self._state[one_dim, two_dim] = 1

            # self.states.append(self._state)
            # self.states.append(new_state)
            # self.availables = np.flatnonzero(self._state==0)

            peptide_sequence, ptms = onehot_to_sequence(self._state, self.restypes)
            peptide_sequence_ptm = ptm_list_to_sequence(peptide_sequence, ptms)
            moved_sequence, _ = onehot_to_sequence_with_extend(
                self._state, self.restypes
            )
            self.c_index_vertical = CC_index(
                moved_sequence,
                get_all=True,
            )

            self.availables = get_availables(
                self._state,
                self.peptide_locked_mask,
                self.enable_extend,
                self.peptide_len,
                self.c_index_horizontal,
                self.c_index_vertical,
                is_nc_cyclic=self.is_nc_cyclic,
            )

            if not self.is_nc_cyclic:
                cc_indexes = CC_index(peptide_sequence, get_all=True)
                groups = get_cc_groups(
                    cc_indexes, self.use_fixed_group, self.fixed_group
                )
                self.cyclic_tail, self.cyclic_head = cc_indexes[-1], cc_indexes[0]
            pdb_str = ""
            if playout == 0:
                if peptide_sequence_ptm not in move_dict.keys():
                    global count2
                    if self.cc_num >= 3:
                        plddt, ipae, iptm, hotspot_distance, score, permeability, group_str = (
                            groups_predict_cycle(
                                self.receptor_input,
                                self.model_runner,
                                peptide_sequence,
                                self.output_dir,
                                self.pocket,
                                count2,
                                self.distance_constraints,
                                cc_groups=groups,
                                ptms=ptms,
                                ligand_bonds=self.ligand_bonds,
                            )
                        )
                        pdb_str = group_str
                    else:
                        plddt, ipae, iptm, hotspot_distance, score, permeability = predict_cycle(
                            self.receptor_input,
                            self.model_runner,
                            peptide_sequence,
                            self.output_dir,
                            self.pocket,
                            count2,
                            self.distance_constraints,
                            self.cyclic_head,
                            self.cyclic_tail,
                            is_nc_cyclic=self.is_nc_cyclic,
                            ptms=ptms,
                            ligand_bonds=self.ligand_bonds,
                        )

                    if self.onlyplddt:
                        reward = plddt / 100

                    elif self.only_loss:
                        reward = score
                    else:
                        reward = plddt * 0.02 + 1 / (abs(hotspot_distance - 3.5) + 2.0)
                        reward = reward / 3.0

                    self.reward = reward

                    move_dict[peptide_sequence_ptm] = [
                        plddt,
                        ipae,
                        iptm,
                        hotspot_distance,
                        permeability,
                        reward,
                        f"{count2}{pdb_str}",
                    ]
                    count2 = count2 + 1

                else:
                    plddt = move_dict[peptide_sequence_ptm][0]
                    ipae = move_dict[peptide_sequence_ptm][1]
                    iptm = move_dict[peptide_sequence_ptm][2]
                    hotspot_distance = move_dict[peptide_sequence_ptm][3]
                    permeability =  move_dict[peptide_sequence_ptm][4]
                    reward = move_dict[peptide_sequence_ptm][5]
                    self.reward = reward

            else:
                if peptide_sequence_ptm not in playout_dict.keys():
                    global count
                    if self.cc_num >= 3:
                        plddt, ipae, iptm, hotspot_distance, score, permeability, group_str = (
                            groups_predict_cycle(
                                self.receptor_input,
                                self.model_runner,
                                peptide_sequence,
                                self.output_dir,
                                self.pocket,
                                count,
                                self.distance_constraints,
                                cc_groups=groups,
                                ptms=ptms,
                                ligand_bonds=self.ligand_bonds,
                            )
                        )
                        pdb_str = group_str
                    else:
                        plddt, ipae, iptm, hotspot_distance, score, permeability = predict_cycle(
                            self.receptor_input,
                            self.model_runner,
                            peptide_sequence,
                            self.output_dir,
                            self.pocket,
                            count,
                            self.distance_constraints,
                            self.cyclic_head,
                            self.cyclic_tail,
                            is_nc_cyclic=self.is_nc_cyclic,
                            ptms=ptms,
                            ligand_bonds=self.ligand_bonds,
                        )

                    if self.onlyplddt:
                        reward = plddt / 100

                    elif self.only_loss:
                        reward = score
                    else:
                        reward = plddt * 0.02 + 1 / (abs(hotspot_distance - 3.5) + 2.0)
                        reward = reward / 3.0

                    self.reward = reward

                    playout_dict[peptide_sequence_ptm] = [
                        plddt,
                        ipae,
                        iptm,
                        hotspot_distance,
                        permeability,
                        reward,
                        f"{count}{pdb_str}",
                    ]

                    count = count + 1

                else:
                    plddt = playout_dict[peptide_sequence_ptm][0]
                    ipae = playout_dict[peptide_sequence_ptm][1]
                    iptm = playout_dict[peptide_sequence_ptm][2]
                    hotspot_distance = playout_dict[peptide_sequence_ptm][3]
                    permeability = playout_dict[peptide_sequence_ptm][4]
                    reward = playout_dict[peptide_sequence_ptm][5]
                    self.reward = reward

        peptide_sequence, ptms = onehot_to_sequence(self._state, self.restypes)
        peptide_sequence_ptm = ptm_list_to_sequence(peptide_sequence, ptms)
        if peptide_sequence_ptm in self.seqs:
            self.repeated = True
            self._state_fitness = 0.0
        else:
            self.seqs.append(peptide_sequence_ptm)

        """if self.reward > self.previous_reward:
            self.init_state = copy.deepcopy(self._state)
            self.init_state_count = 0"""

        all_not_worse = np.all(self.reward >= self.global_best_reward)
        at_least_one_better = np.any(self.reward > self.global_best_reward)

        all_above_threshold = np.all(self.reward > 0.7)

        if all_not_worse and at_least_one_better and all_above_threshold:
            self.global_best_reward = copy.deepcopy(self.reward)
            self.global_best_state = copy.deepcopy(self._state)
            self.init_state = copy.deepcopy(self._state)
            self.init_state_count = 0

    def game_end(self):
        """Check whether the game is ended or not"""
        # check hot-pot distance constraints
        for distance_constraint in self.distance_constraints:
            if not distance_constraint.is_available:
                #self.reward = 0.0
                self.reward = np.zeros(self.reward_dim, dtype=np.float32)
                return True

        all_not_worse2 = np.all(self.reward <= self.previous_reward)
        at_least_one_better2 = np.any(self.reward < self.previous_reward)
        if all_not_worse2 and at_least_one_better2:
            return True
        
        if self.unuseful_move == 1:
            return True
        if self.repeated:
            return True
        return False
