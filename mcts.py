import copy
from typing import Dict, List

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from pre import (
    EXTENDED_RESTYPES,
    RESTYPES,
    get_emphasize_locked_sequence_str,
    onehot_to_sequence,
    softmax,
)

from seqenv import Seqenv
from tree_node import TreeNode


def action_scale(
    action: int | NDArray,
    width: int,
):
    standard_index = 19
    action_col = action % width
    if isinstance(action_col, np.ndarray):
        probs_scale = np.ones_like(action_col)
        probs_scale = np.where(action_col > standard_index, 0.8, probs_scale)
        probs_scale = np.where(action_col == width - 1, 0.8, probs_scale)
        return probs_scale

    if action_col <= standard_index:
        return 1.0
    if action_col > standard_index and action_col < width - 1:
        return 0.8
    return 0.8


class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, action_scale_fn, c_puct=5, n_playout=10000):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._action_scale = action_scale_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    
    def _playout(self, seqenv: Seqenv):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        while True:
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct, self._action_scale)
            seqenv.do_move(action, playout=1)

        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.
        action_probs, leaf_value = self._policy(seqenv)

        sorted_action_probs = sorted(action_probs, key=lambda x: x[1], reverse=True)
        width_per_layer = int(seqenv.current_state().shape[0]*0.7)+1
        top5_action_probs = sorted_action_probs[:width_per_layer]

        # Check for end of game.
        end = seqenv.game_end()
        # todo: expand addtional length
        if not end:
            node.expand(top5_action_probs)
        else:
            leaf_value = seqenv.reward

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(leaf_value)

        return seqenv.global_best_reward,seqenv.global_best_state

    def get_move_probs(self, state: Seqenv, temp=1e-3):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        for n in range(self._n_playout):
            state_copy: Seqenv = Seqenv.copy(state)
            self._playout(state_copy)
            # add log msg
            current_seq, ptms = onehot_to_sequence(
                state_copy._state, state_copy.restypes
            )
            extend = ""
            extend_num = extend_num = len(current_seq) - state_copy.peptide_len
            if extend_num > 0:
                extend = f" <green>Extended:</green> <red>{extend_num}</red>"
            logger.opt(colors=True).info(
                f"Playout:{n} Sequence: {get_emphasize_locked_sequence_str(current_seq, state_copy.peptide_locked_mask,ptms)} {extend}"
            )

        # calc the move probabilities based on visit counts at the root node
        act_visits = [
            (act, node._n_visits) for act, node in self._root._children.items()
        ]
        acts = []
        visits = []

        for act, node in act_visits:
            if act not in state.availables:
                continue
            acts.append(act)
            visits.append(node)
        # acts, visits = zip(*act_visits)

        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def update_init(self, state, play_best_reward, play_best_state):
        is_better = (
            np.all(play_best_reward >= state.global_best_reward)
            and np.any(play_best_reward > state.global_best_reward)
        )

        if not is_better:
            return

        state.global_best_reward = copy.deepcopy(play_best_reward)
        state.global_best_state = copy.deepcopy(play_best_state)
        state.init_state = copy.deepcopy(play_best_state)
        state.init_state_count = 0

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """AI player based on MCTS"""

    def __init__(
        self,
        policy_value_function,
        action_scale_fn,
        c_puct=5,
        n_playout=2000,
        is_selfplay=True,
    ):
        self.mcts = MCTS(policy_value_function, action_scale_fn, c_puct, n_playout)
        self._is_selfplay = is_selfplay
        self.action_scale_fn = action_scale_fn

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, seqenv: Seqenv, temp=1e-3, return_prob=True):
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        peptide_length = max(seqenv.peptide_len, seqenv.max_extend_length)
        aattpes = seqenv.restypes

        move_probs = np.zeros(peptide_length * len(aattpes))
        acts, probs = self.mcts.get_move_probs(seqenv, temp)
        move_probs[list(acts)] = self.action_scale_fn(np.array(acts)) * probs
        if self._is_selfplay:
            # add Dirichlet Noise for exploration (needed for
            # self-play training)
            #move = np.random.choice(acts,p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))),)
            move = np.random.choice(acts,p=probs,)
            # update the root node and reuse the search tree
            self.mcts.update_with_move(move)
        else:
            # with the default temp=1e-3, it is almost equivalent
            # to choosing the move with the highest prob
            move = np.random.choice(acts, p=probs)
            # reset the root node
            self.mcts.update_with_move(-1)
        #                location = board.move_to_location(move)
        #                print("AI move: %d,%d\n" % (location[0], location[1]))

        if return_prob:
            return move, move_probs
        else:
            return move

    def __str__(self):
        return "MCTS {}".format(self.player)
