from random import randint
import tensorflow as tf
import numpy as np
import copy
from game import Referee, Board
import math
from network import PolicyValueNet

class TreeNode:
    def __init__(self, parent):
        self._parent = parent
        self._children = {}
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def is_root(self):
        return self._parent is None

    def is_leaf(self):
        return len(self.children) == 0

    def get_root(self):
        root = self
        while root.parent != None:
            root = root.parent
        return root

    def get_move_by_child(self, target_child):
        for move, child in self.children.items():
            if target_child == child:
                return move

        raise NameError("child not found")
        

    def get_utc(self):
        t = self.get_root().played
        c = math.sqrt(2)
        wi = self.win
        ni = self.played
        return wi/ni + c * math.sqrt(math.log(t) / ni)

    def get_board(self):
        board = Board(15,15)
        node = self
        while node.parent != None:
            move = node.parent.get_move_by_child(node)
            board.move(move[0], move[1], node.player)
            node = node.parent
        return board

    def select(self):
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def expand(self, action_priors):
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def update(self, win, played, player):
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

        self._n_visits += 1
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def get_value(self, c_puct):
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u
        
            

class MCTS:
    def __init__(self):
        self.root = TreeNode(None, 1.0)
        self.curr_node = self.root
        self.predictor = Predictor(15, 15)
        self.net = PolicyValueNet(15, 15)


    def train_self(self):
        selected_node = self.root.select()
        expanded_node = selected_node.expand()
        win = expanded_node.simulate(100)
        expanded_node.update(win, 100, expanded_node.player)

    def playout(self, board):
        node = self.root
        player = 1

        while(not node.is_leaf()):
            action, node = node.select()
            board.move(action[0], action[1], player)
            if player == 1:
                player = 2
            else:
                player = 1
        
        probs, leaf_value = self.net.policy_value(board, player)

        referee = Referee()
        state = referee.get_game_state(board)
        if state < 3:
            node.expand()
        else:
            if state == 0:
                leaf_value = 0
            else:
                leaf_value = 

        

        
        
        


        


    def get_next_move(self, board, player):
        if len(self.curr_node.children) == 0:
            expanded_node = self.curr_node.expand()
            win = expanded_node.simulate(100)
            expanded_node.update(win, 100, expanded_node.player)
            #raise NameError("I don't know!!")
        
        next_move, next_child = max(self.curr_node.children, key=lambda p: p[1].get_utc())
        self.curr_node = next_child
        return next_move

class Predictor:
    def __init__(self, width, height):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation='relu',input_shape=(height, width, 3)),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
            tf.keras.layers.Conv2D(4, (1,1), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def reverse_player(self, board):
        for i in board.shape[0]:
            for j in board.shape[1]:
                if board[i, j, 0] == 1:
                    if board[i, j, 1] == 1:
                        board[i, j, 1] = 0
                        board[i, j, 2] = 1
                    else:
                        board[i, j, 1] = 1
                        board[i, j, 2] = 0

    def predict(self, board, player):
        availables = board.available_moves()
        train_data = np.zeros((len(availables), board.width(), board.height(), 3))

        for i in range(len(availables)):
            move = availables[i]
            b = copy.deepcopy(board)
            if player == 2:
                self.reverse_player(b.board)

            b.move(move[0], move[1], 1)
            train_data[i, :] = b.board
            
        scores = self.model.predict(train_data)

        result = []
        for i in range(len(availables)):
            result.append((availables[i], scores[i,0]))

        result.sort(key=lambda x : x[1], reverse=True)
        return result

class MCTSPlayer:
    def __init__(self, width, height, player):
        self.predictor = Predictor(width, height)
        self.mcts = MCTS(width, height, policy)

    def policy(self, board, player):
        return self.predictor.predict(board,player)

    def prestart(self):
        self.mcts.train_self()
        
    def get_next_move(self, board, player):
        return self.mcts.get_next_move(board, player)

if __name__ == "__main__":
    board = Board(15, 15)
    predictor = Predictor(15, 15)
    print(predictor.predict(board, 1))
    
