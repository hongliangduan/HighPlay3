import copy

import numpy as np
from loguru import logger

from mcts import MCTSPlayer
from pre import (
    mutate_extend_seq,
    mutate_seq,
    onehot_to_sequence,
    onehot_to_sequence_with_extend,
    sequence_to_onehot,
)
from ptm_utils import ptm_list_to_sequence
from seqenv import Seqenv, init_dict


class Mutate:
    """game server"""

    def __init__(self, seqenv: Seqenv):
        self.seqenv = seqenv

    def start_mutating(self, player: MCTSPlayer, n_iter, temp=1e-3, jumpout=50):
        """start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        if (self.seqenv.previous_init_state == self.seqenv.init_state).all():
            self.seqenv.init_state_count += 1
        if self.seqenv.init_state_count >= jumpout:
            print("****Random start replacement****")

            current_start_seq, ptms = onehot_to_sequence_with_extend(
                self.seqenv.init_state, self.seqenv.restypes
            )
            # occurred_seqs = copy.deepcopy(self.seqenv.seqs)
            if self.seqenv.enable_extend:
                new_start_seq, mew_init_ptms = mutate_extend_seq(
                    current_start_seq,
                    init_dict.keys(),
                    self.seqenv.peptide_locked_mask,
                    self.seqenv.peptide_len,
                    self.seqenv.max_extend_length,
                    self.seqenv.cc_num,
                    is_nc_cyclic=self.seqenv.is_nc_cyclic,
                    all_restypes=self.seqenv.restypes,
                    init_ptms=self.seqenv.init_ptms,
                )
            else:
                new_start_seq, mew_init_ptms = mutate_seq(
                    current_start_seq,
                    init_dict.keys(),
                    self.seqenv.peptide_locked_mask,
                    self.seqenv.cc_num,
                    is_nc_cyclic=self.seqenv.is_nc_cyclic,
                    all_restypes=self.seqenv.restypes,
                    init_ptms=self.seqenv.init_ptms,
                )
            self.seqenv.start_seq = new_start_seq
            self.seqenv.init_ptms = mew_init_ptms
            self.seqenv.init_state = sequence_to_onehot(
                new_start_seq,
                self.seqenv.restypes,
                use_ptms=self.seqenv.use_ptms,
                init_ptms=self.seqenv.init_ptms,
            )
            self.seqenv.init_state_count = 0

        states, mcts_probs, rewards = [], [], []
        logger.info(f"Batch: {n_iter}, Init Num: {self.seqenv.init_state_count}")
        self.seqenv.init_seq_state()

        while True:

            move, move_probs = player.get_action(
                self.seqenv, temp=temp, return_prob=True
            )

            states.append(self.seqenv.current_state())

            mcts_probs.append(move_probs)
            rewards.append(self.seqenv.reward)
            # perform a move
            self.seqenv.do_move(move)

            current_start_seq, ptms = onehot_to_sequence(
                self.seqenv.init_state, self.seqenv.restypes
            )
            current_start_seq_ptm = ptm_list_to_sequence(current_start_seq, ptms)

            player.mcts._n_playout = len(current_start_seq)

            end = self.seqenv.game_end()
            if end:
                logger.opt(colors=True).info(
                    f"<green>Batch: {n_iter} game end at init num: {self.seqenv.init_state_count}</green>"
                )
                player.reset_player()
                return zip(states, mcts_probs, rewards)
