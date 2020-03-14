from random import randint
import tensorflow as tf
import numpy as np
import copy
from game import Referee, Board
import math
from network import PolicyValueNet

class TreeNode:
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def is_root(self):
        return self._parent is None

    def is_leaf(self):
        return len(self._children) == 0 

    def select(self, c_puct):
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def expand(self, action_priors):
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def update(self, leaf_value):
        if self._parent:
            self._parent.update(-leaf_value)

        self._n_visits += 1
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def get_value(self, c_puct):
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

class MCTS:
    def __init__(self, net, c_puct, n_playout):
        self._root = TreeNode(None, 1.0)
        self.net = net
        self._n_playout = n_playout
        self.c_puct = c_puct

    def _playout(self, board, current_player):
        node = self._root
        player = current_player
        action = 0
        while(not node.is_leaf()):
            action, node = node.select(self.c_puct)
            board.do_move(action, player)
            player = 1 if player == 2 else 2
        
        probs, leaf_value = self.net.policy_value(board, player)
        is_end, winner = board.game_end()

        if not is_end:
            node.expand(probs)
        else:
            if winner == 0:
                leaf_value = 0
            else:
                leaf_value = (1.0 if winner == current_player else -1.0)

        node.update(leaf_value)

    def get_move_probs(self, state, player, temp=1e-3):
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy, player)

        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]

        acts, visits = zip(*act_visits)

        def softmax(x):
            probs = np.exp(x - np.max(x))
            probs /= np.sum(probs)
            return probs

        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))
        return acts, act_probs

    def update_with_move(self, last_move):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)


class MCTSPlayer:
    def __init__(self, net,
                 c_puct=5, n_playout=1000, is_selfplay=False):
        self.mcts = MCTS(net, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def prestart(self):
        pass

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_next_move(self, board, player):
        temp = 1e-3

        sensible_moves = board.availables
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(board.width * board.height)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(board, player, temp)
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                move = np.random.choice(
                    acts,
                    p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                )
                self.mcts.update_with_move(move)
            else:
                move = np.random.choice(acts, p=probs)
                self.mcts.update_with_move(-1)

            return move, move_probs
        else:
            print("WARNING: the board is full")
