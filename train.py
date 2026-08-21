import csv
import functools
import os
import random
from collections import deque
from time import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from loguru import logger

from mcts import MCTSPlayer, action_scale
from mutate import Mutate
from policyvaluenet import PolicyValueNet
from pre import EXTENDED_RESTYPES, RESTYPES, onehot_to_sequence
from seqenv import Seqenv, init_dict, move_dict, playout_dict

aatypes = [
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


def filter_dict(before_dict, filtered_dict):
    for peptide_sequence, values in before_dict.items():
        print(float(values[5][0]),float(values[5][1]))
        if float(values[5][0]) > 0.7 and float(values[5][1]) > 0.7:
            if peptide_sequence not in filtered_dict:
                filtered_dict[peptide_sequence] = values


class TrainPipeline:
    def __init__(
        self,
        init_seq,
        receptor_input,
        model_runner,
        peptide_locked_mask,
        pocket: Dict[str, List[int]],
        output_dir,
        num_iterations,
        plDDT_only,
        only_loss,
        jumpout_num,
        distance_constraints: List[List[Any]],
        max_extend_length=None,
        init_model=None,
        cc_num: int = 2,
        use_fixed_group: bool = False,
        fixed_group: List[int] = None,
        is_nc_cyclic: bool = False,
        use_ptms: bool = False,
        ptms: List[str] = [],
        init_ptms: List[tuple[int, str, str]] = [],
        max_run_time: float = 12.0,
        ligand_bonds=None,
    ):
        # peptide and pocket params
        self.init_seq = init_seq
        self.peptide_locked_mask = peptide_locked_mask
        self.pocket = pocket
        self.peptide_length = len(init_seq)
        self.use_ptms = use_ptms
        self.ptms = ptms
        self.init_ptms = init_ptms
        self.max_run_time = max_run_time
        # default not enable extend length
        # if inputed valid max_extend_length(>peptide_length), enable extend length
        self.enbale_extend_length: bool = False
        if max_extend_length and max_extend_length > self.peptide_length:
            self.max_extend_length = max_extend_length
            self.enbale_extend_length = True
        else:
            self.max_extend_length = self.peptide_length

        self.output_dir = output_dir
        self.onlyplddt = plDDT_only
        self.only_loss = only_loss
        self.jumpout_num = jumpout_num

        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        # self.n_playout = 16  # num of simulations for each move
        self.n_playout = len(init_seq)
        self.c_puct = 0.5
        self.batch_size = 8  # mini-batch size for training
        self.data_buffer = deque(maxlen=1000)
        self.play_batch_size = 1
        self.game_batch_num = num_iterations
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02

        #
        self.seq_env = Seqenv(
            self.init_seq,
            receptor_input,
            peptide_locked_mask,
            self.pocket,
            self.output_dir + "/result",
            self.onlyplddt,
            self.only_loss,
            self.enbale_extend_length,
            self.max_extend_length,
            distance_constraints,
            cc_num,
            use_fixed_group,
            fixed_group,
            0,
            is_nc_cyclic,
            model_runner=model_runner,
            use_ptms=use_ptms,
            ptms=ptms,
            init_ptms=init_ptms,
            ligand_bonds=ligand_bonds,
        )
        self.mutate = Mutate(self.seq_env)
        self.playout_dict = {}

        model_width = len(RESTYPES)
        if self.enbale_extend_length:
            model_width = len(EXTENDED_RESTYPES)

        if use_ptms:
            model_width += len(ptms)

        if init_model:  # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(
                model_width,
                self.max_extend_length,
                model_file=init_model,
                use_gpu=True,
            )
        else:  # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(
                model_width, self.max_extend_length, use_gpu=True
            )

        self.mcts_player = MCTSPlayer(
            self.policy_value_net.policy_value_fn,
            functools.partial(action_scale, width=model_width),
            c_puct=self.c_puct,
            n_playout=self.n_playout,
            is_selfplay=True,
        )

    def collect_selfplay_data(self, n_games, n_iter):
        """collect self-play data for training"""
        for _ in range(n_games):
            play_data = self.mutate.start_mutating(
                self.mcts_player, n_iter, temp=self.temp, jumpout=self.jumpout_num
            )
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            self.data_buffer.extend(play_data)
            np.save(
                self.output_dir + "/data_buffer.npy",
                np.array(self.data_buffer, dtype=object),
            )

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]

        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                self.learn_rate * self.lr_multiplier,
            )
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            # todo: add mask to aviod padding influence on kl
            kl = np.mean(
                np.sum(
                    old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1,
                )
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = 1 - np.var(
            np.array(winner_batch) - old_v.flatten()
        ) / np.var(np.array(winner_batch))
        explained_var_new = 1 - np.var(
            np.array(winner_batch) - new_v.flatten()
        ) / np.var(np.array(winner_batch))

        print(
            (
                "kl:{:.5f},"
                "lr_multiplier:{:.3f},"
                "loss:{},"
                "entropy:{},"
                "explained_var_old:{:.3f},"
                "explained_var_new:{:.3f}"
            ).format(
                kl,
                self.lr_multiplier,
                loss,
                entropy,
                explained_var_old,
                explained_var_new,
            )
        )
        return loss, entropy

    def run(self):
        start_t = time()
        max_run_t = self.max_run_time * 3600
        assert self.game_batch_num * self.play_batch_size < (
            pow(len(aatypes) - 1, self.peptide_length - self.peptide_locked_mask.sum())
            * self.jumpout_num
        ), "Mutable resduidues are too less, maybe exhaustion all sequences, please reduce iteration number."
        prefix = "af3_"
        iter_num = -1
        high_num = 0
        while (time() - start_t) < max_run_t:
            if iter_num >= self.game_batch_num and high_num >= 20:
                break
            iter_num += 1
            logger.info("start i:{}".format(iter_num + 1))
            batch_start = time()
            self.collect_selfplay_data(self.play_batch_size, iter_num)
            logger.info(
                f"Batch i:{iter_num + 1}, episode_len:{self.episode_len}  Cost time:{time() - batch_start:.3f}s"
            )

            with open(f"{self.output_dir}/playout_dict.csv", "w") as f:
                f.write("sequence, plddt, ipae, iptm, hotspot_distance, permeability, reward, file\n")
                writer = csv.writer(f)
                for key, value in playout_dict.items():
                    writer.writerow(
                        [
                            key,
                            value[0],
                            value[1],
                            value[2],
                            value[3],
                            value[4],
                            ','.join([f"{s:.3f}" for s in value[5]]),
                            f"{prefix}{value[6]}.pdb",
                        ]
                    )

            with open(f"{self.output_dir}/init_dict.csv", "w") as f:
                f.write("sequence, plddt, ipae, iptm, hotspot_distance, permeability, reward, file\n")
                writer = csv.writer(f)
                for key, value in init_dict.items():
                    writer.writerow(
                        [
                            key,
                            value[0],
                            value[1],
                            value[2],
                            value[3],
                            value[4],
                            ','.join([f"{s:.3f}" for s in value[5]]),
                            f"{prefix}{value[6]}.pdb",
                        ]
                    )

            with open(f"{self.output_dir}/move_dict.csv", "w") as f:
                f.write("sequence, plddt, ipae, iptm, hotspot_distance, permeability, reward, file\n")
                writer = csv.writer(f)
                for key, value in move_dict.items():
                    writer.writerow(
                        [
                            key,
                            value[0],
                            value[1],
                            value[2],
                            value[3],
                            value[4],
                            ','.join([f"{s:.3f}" for s in value[5]]),
                            f"{prefix}{value[6]}.pdb",
                        ]
                    )

            high_dict = {}
            filter_dict(init_dict, high_dict)
            filter_dict(move_dict, high_dict)
            filter_dict(playout_dict, high_dict)
            high_num = len(high_dict)
            with open(f"{self.output_dir}/high.csv", "w") as f:
                f.write("sequence, plddt, ipae, iptm, hotspot_distance, permeability, reward, file\n")
                writer = csv.writer(f)
                for key, value in high_dict.items():
                    writer.writerow(
                        [
                            key,
                            value[0],
                            value[1],
                            value[2],
                            value[3],
                            value[4],
                            #value[5],
                            ','.join([f"{s:.3f}" for s in value[5]]),
                            f"{prefix}{value[6]}.pdb",
                        ]
                    )

            if len(self.data_buffer) > self.batch_size:
                print("start training...")
                loss, entropy = self.policy_update()
                self.policy_value_net.save_model(self.output_dir + "/current_policy.pt")
