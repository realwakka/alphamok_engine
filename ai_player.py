from __future__ import absolute_import, division, print_function, unicode_literals

import random
import tensorflow as tf
import numpy as np
from game import Board
import copy

model_file = 'simple_ai_player.h5'

class AIPlayer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.data_buffer = []
        
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
            tf.keras.layers.Dense(64, activation='relu'),            
            tf.keras.layers.Dense(3)
        ])

        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])
        

    def reverse_player(self, board):
        reversed = np.copy(board)
        for i in range(reversed.shape[0]):
            for j in range(reversed.shape[1]):
                if reversed[i, j, 0] == 0:
                    if reversed[i, j, 1] == 1:
                        reversed[i, j, 1] = 0
                        reversed[i, j, 2] = 1
                    else:
                        reversed[i, j, 1] = 1
                        reversed[i, j, 2] = 0
                        
        return reversed

    def prestart(self):
        pass
    
    def save(self):
        self.model.save(model_file)

    def on_finish_game(self, win_player, history):
        # save history to data_buffer
        for board in history:
            b = board.board
            self.data_buffer.append([b, win_player])
            b = np.rot90(b)
            self.data_buffer.append([b, win_player])
            b = np.rot90(b)
            self.data_buffer.append([b, win_player])
            b = np.rot90(b)
            self.data_buffer.append([b, win_player])

        sample_size = 512
        if len(self.data_buffer) < sample_size:
            return
        
        sample_data = random.sample(self.data_buffer, sample_size)
        train_input = np.zeros((sample_size, self.width, self.height, 3))
        train_output = np.ones((sample_size))
        for i in range(0, sample_size):
            train_input[i, :] = sample_data[i][0]
            train_output[i] = sample_data[i][1]

        self.model.fit(x=train_input, y=train_output, epochs=5)
        self.save()        

    def get_next_move(self, board, player):
        predict_result = self.predict(board, player)
        next_move = predict_result[0][0]
        return next_move

    def predict(self, board, player):
        availables = board.available_moves()
        predict_input = np.zeros((len(availables), board.width(), board.height(), 3))

        for i in range(len(availables)):
            move = availables[i]
            if player == 1:
                b = copy.deepcopy(board.board)
            else:
                b = self.reverse_player(board.board)
            b[move[0], move[1], player] = 1
            predict_input[i, :] = b
            
        scores = self.model.predict(predict_input)
        result = []
        for i in range(len(availables)):
            result.append((availables[i], scores[i, player]))

        result.sort(key=lambda x : x[1], reverse=False)
        return result
