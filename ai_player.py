from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
from game import Board
import copy

class AIPlayer:
    def __init__(self, width, height, player):
        self.width = width
        self.height = height
        self.player = player
        self.history = []

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation='relu',input_shape=(height, width, 3)),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
            tf.keras.layers.Conv2D(4, (1,1), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(3)
        ])

        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        print(self.model.summary())
        

    def reverse_player(self, board):
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                if board[i, j, 0] == 1:
                    if board[i, j, 1] == 1:
                        board[i, j, 1] = 0
                        board[i, j, 2] = 1
                    else:
                        board[i, j, 1] = 1
                        board[i, j, 2] = 0

    def prestart(self):
        pass

    def on_finish_game(self, win_player):
        data_size = len(self.history)
        train_data = np.zeros((len(self.history), self.width, self.height, 3))
        train_result = np.zeros((len(self.history)))

        for i in range(len(self.history)):
            board = self.history[i]
            train_data[i, :] = board.board
        
        predict_data = self.model.predict(train_data)

        print("predict_data!!!!\n")
        print(predict_data.shape)

        train_result = np.ones(data_size) * win_player
        #train_result[:, win_player] = 1
        #train_result = predict_data

        #train_result = predict_data + acc
        print("train_data!!!!\n")

        print("winplayer " + str(win_player))
        self.model.fit(x=train_data, y=train_result)
        
        


    def get_next_move(self, board, player):
        curr_board = copy.deepcopy(board)
        self.history.append(curr_board)

        predict_result = self.predict(board, player)
        next_move = predict_result[0][0]
        next_board = copy.deepcopy(board)
        next_board.move(next_move[0], next_move[1], player)
        self.history.append(next_board)
        
        return next_move

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
            result.append((availables[i], scores[i, self.player]))

        result.sort(key=lambda x : x[1], reverse=True)
        return result


    


