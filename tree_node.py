import numpy as np
import random
from typing import Dict, Callable
from loguru import logger
import os
import json


script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "user_extra_restypes.json")

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

ptms_list = data.get("ptms", [])


class TreeNode:
    """
    A node in the MCTS tree with multi-objective optimization and Pareto-UCB selection.

    Selection logic:
    1. If there are unvisited children, prioritize them first.
       Among unvisited children, select according to prior probability P.
    2. After all children have been visited at least once, use Q + UCB for Pareto-based selection.

    Important:
    All objectives are assumed to be maximized.
    If your original score is a loss or energy to be minimized,
    you should convert it before calling update(), for example:
        reward = -loss
        reward = -energy
    """

    def __init__(
        self,
        parent: "TreeNode" = None,
        prior_p: float = 1.0,
        num_objectives: int = 2,
    ):
        self._parent = parent
        self._children: Dict[str, TreeNode] = {}
        self._n_visits = 0
        self._P = prior_p

        self.num_objectives = num_objectives

        # Q stores the mean value of each objective.
        # Larger Q means better node.
        self._Q = np.zeros(num_objectives, dtype=np.float32)

        self._u = 0.0

        # Fraction of Pareto-front nodes retained after sorting by the first objective.
        self.TOP_FRACTION = 0.5

    def expand(self, action_priors):
        """
        Expand tree by creating new children.

        Parameters
        ----------
        action_priors:
            list of (action, prior probability)
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(
                    parent=self,
                    prior_p=prob,
                    num_objectives=self.num_objectives,
                )

    def select(
        self,
        c_puct: float,
        action_scale_fn: Callable = None,
        top_fraction: float = None,
    ):

        if top_fraction is None:
            top_fraction = self.TOP_FRACTION

        if len(self._children) == 0:
            raise ValueError("Cannot select from a leaf node with no children.")
        
        def get_action_weight(action):
                try:
                    remainder = int(action) % (20 + len(ptms_list))
                except Exception:
                    remainder = 0
                return 1.0 if remainder >= 19 else 0.8
        
        def calc_score(node: "TreeNode") -> np.ndarray:
            """
            Compute multi-objective UCB score.

            score = Q + U

            Q: mean reward vector
            U: exploration bonus, broadcast to all objectives
            """

            parent_visits = max(1, node._parent._n_visits)

            u = (
                c_puct
                * node._P
                * np.sqrt(parent_visits)
                / (1 + node._n_visits)
            )

            u_vec = np.full(
                node.num_objectives,
                u,
                dtype=np.float32,
            )

            return node._Q + u_vec

        unvisited_items = [
            (action, node)
            for action, node in self._children.items()
            if node._n_visits == 0
        ]

        if len(unvisited_items) > 0:
            return max(unvisited_items,key=lambda item: item[1]._P * get_action_weight(item[0]))

        pareto_front = self._build_pareto_front(
            score_fn=calc_score,
            top_fraction=top_fraction,
        )

        front_items = list(pareto_front.items())
        
        weights = np.array([get_action_weight(action) for action, _ in front_items], dtype=np.float32)
        probs = weights / weights.sum()

        return front_items[np.random.choice(len(front_items), p=probs)]

    def _build_pareto_front(
        self,
        score_fn: Callable,
        top_fraction: float,
    ) -> Dict[str, "TreeNode"]:
        """
        Compute Pareto front using vectorized non-dominated sorting.

        Important:
        All objectives are assumed to be maximized.

        dominates[j, i] = True means node j dominates node i.
        """

        if len(self._children) == 0:
            return {}

        actions = list(self._children.keys())
        nodes = [self._children[action] for action in actions]

        scores = np.array(
            [score_fn(node) for node in nodes],
            dtype=np.float32,
        )

        # scores shape: (N, M)
        # N = number of child nodes
        # M = number of objectives

        # ge[j, i, k] = scores[j, k] >= scores[i, k]
        ge = scores[:, None, :] >= scores[None, :, :]

        # gt[j, i, k] = scores[j, k] > scores[i, k]
        gt = scores[:, None, :] > scores[None, :, :]

        # dominates[j, i] = True means node j dominates node i
        dominates = ge.all(axis=2) & gt.any(axis=2)

        # A node i is dominated if any node j dominates it
        dominated = dominates.any(axis=0)

        # Pareto front = nodes that are not dominated
        # Pareto front = nodes that are not dominated
        pareto_front = np.where(~dominated)[0]

        return pareto_front

    def update(self, leaf_value: np.ndarray):
        """
        Update node values with multi-objective leaf evaluation.

        Parameters
        ----------
        leaf_value:
            np.ndarray with shape = (num_objectives,)

        Important:
        Each value in leaf_value should follow the same convention:
            larger value = better node

        Examples
        --------
        If your model returns rewards:
            leaf_value = np.array([affinity_reward, permeability_reward])

        If your model returns losses:
            leaf_value = np.array([-affinity_loss, -permeability_loss])

        If your model returns energy values to be minimized:
            leaf_value = np.array([-energy, permeability_reward])
        """

        assert isinstance(leaf_value, np.ndarray), "leaf_value must be np.ndarray"

        assert leaf_value.shape[0] == self.num_objectives, (
            f"leaf_value shape mismatch: "
            f"{leaf_value.shape[0]} != {self.num_objectives}"
        )

        assert not np.isnan(leaf_value).any(), "leaf_value contains NaN"

        self._n_visits += 1

        self._Q += (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value: np.ndarray):
        """
        Recursively update this node and all its ancestors.
        """

        if self._parent:
            self._parent.update_recursive(leaf_value)

        self.update(leaf_value)

    def get_value(self, c_puct: float):
        """
        Return Q + U for this node.

        This function is kept for compatibility with other MCTS implementations.
        The current select() function already calculates Q + U internally.
        """

        if self._parent is None:
            self._u = 0.0
        else:
            parent_visits = max(1, self._parent._n_visits)

            self._u = (
                c_puct
                * self._P
                * np.sqrt(parent_visits)
                / (1 + self._n_visits)
            )

        return self._Q + self._u

    def is_leaf(self):
        """
        Check whether this node is a leaf node.
        """

        return len(self._children) == 0

    def is_root(self):
        """
        Check whether this node is the root node.
        """

        return self._parent is None