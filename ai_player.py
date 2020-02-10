from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
from game import Board
import copy

class AIPlayer:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.history = []

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

    def on_finish_game(self, win_player):
        train_data = np.zeros((len(self.history), self.width, self.height, 3))

        for i in range(len(self.history)):
            board, move = self.history[i]
            train_data[i, :] = board
        
        predict_data = self.model.predict(train_data)
        


    def get_next_move(self, board, player):
        predict_result = self.predict(board, player)
        self.history.append((copy.deepcopy(board.board), player))
        return predict_result[0][0]

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


        
def main():
    selector = Selector(15, 15)
    selector.train()
    
    
if __name__ == "__main__":
    target = np.zeros((2, 2, 2))
    d = np.ones((2,2))
    target[0, :] = np.ones((2,2))
    target[1, :] = np.ones((2,2)) * 2
    print(target)


