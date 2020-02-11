from random import randint
import tensorflow as tf
import numpy as np
import copy
from game import Referee, Board
import math

class TreeNode:
    def __init__(self, parent, policy):
        self.played = 0
        self.win = 0
        self.parent = parent
        if parent == None or parent.player == 2:
            self.player = 1
        else:
            self.player = 2

        self.children = {}
        self.policy = policy
        

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
        if len(self.children) == 0:
            return self
        return max(self.children, key = lambda move, child : child.get_utc())

    def expand(self):
        board = self.get_board()
        availables = board.available_moves()

        if len(availables) == 0:
            raise NameError("no availables")

        result = policy(board, player)
        for move, value in result:
            self.children[move] = TreeNode(self, self.policy)
            

        # select random move now, but it will be selected by nn
        next_move = availables[randint(0, len(availables)-1)]
        new_child = TreeNode(self, self.policy)
        self.children.append((next_move, new_child))
        return new_child

    def update(self, win, played, player):
        if player == self.player:            
            self.win += win
        self.played += played

        if self.parent != None:
            self.parent.update(win, played, player) 

    def simulate(self, playout):
        class TrainingPlayer:
            def prestart(self):
                pass

            def get_next_move(self, board, player):
                availables = board.available_moves()
                return availables[randint(0, len(availables) - 1)]

        player = TrainingPlayer()

        win = 0
        for i in range(playout):
            referee = Referee()
            board = Board(15, 15)
            game_state = referee.start_game(board, player, player)
            if player == game_state:
                win += 1

        return win

            

class MCTS:
    def __init__(self, policy):
        self.root = TreeNode(None, policy)
        self.curr_node = self.root
        self.predictor = Predictor(15, 15)

    def train_self(self):
        selected_node = self.root.select()
        expanded_node = selected_node.expand()
        win = expanded_node.simulate(100)
        expanded_node.update(win, 100, expanded_node.player)

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
    
