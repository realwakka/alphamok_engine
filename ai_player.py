from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
from game import Board
import copy

model_file = 'simple_ai_player.h5'

class AIPlayer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.history = []
        try:
            self.model = tf.keras.models.load_model(model_file)
            return
        except OSError:
            print('no model file... creating new model...')

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
                           loss='categorical_crossentropy',
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
    
    def save(self):
        self.model.save(model_file)

    def on_finish_game(self, win_player):
        
        sample_size = len(self.history)
        train_input = np.zeros((sample_size * 4, self.width, self.height, 3))

        
        for i in range(sample_size):
            board = self.history[i]
            train_input[i*4, :] = board.board
            train_input[i*4+1, :] = np.rot90(train_input[i*4, :])
            train_input[i*4+2, :] = np.rot90(train_input[i*4+1, :])
            train_input[i*4+3, :] = np.rot90(train_input[i*4+2, :])
        
        
        
        predict_data = self.model.predict(train_input)
        
        train_output = predict_data
        train_output[:, win_player] += 0.1

        print(train_output.shape)
        
        self.save()

        """
        train_data = np.zeros((1, self.width, self.height, 3))
        train_data[0, :] = self.history[-1].board
        train_result = np.ones((1, 1)) * win_player
        """
        self.model.fit(x=train_input, y=train_output)


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
            b.move(move[0], move[1], player)
            train_data[i, :] = b.board
            
        scores = self.model.predict(train_data)
        result = []
        for i in range(len(availables)):
            result.append((availables[i], scores[i, player]))

        result.sort(key=lambda x : x[1], reverse=True)
        return result


    


