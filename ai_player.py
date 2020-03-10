from __future__ import absolute_import, division, print_function, unicode_literals

import random
import tensorflow as tf
import numpy as np
from game import Board
import copy

model_file = 'simple_ai_player.h5'

class AIPlayer:
    def __init__(self, width, height, training):
        self.width = width
        self.height = height
        self.data_buffer = []
        self.training = training
        self.e = 0.2

        try:
            self.model = tf.keras.models.load_model(model_file)
            return
        except OSError:
            print('no model file... creating new model...')

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation='relu',input_shape=(width, height, 3)),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
            tf.keras.layers.Conv2D(4, (1,1), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),            
            tf.keras.layers.Dense(3, activation='relu')
        ])

        self.model.compile(optimizer='adam',
                           loss="mean_squared_error",
                           metrics=['accuracy'])
        


    def prestart(self):
        pass
    
    def save(self):
        self.model.save(model_file)

    def on_finish_game(self, win_player, history):
        # save history to data_buffer
        for board in history:
            b = board.board
            result = np.ones((3)) * -1
            result[win_player] = 1
            
            for i in range(4):
                
                rotated = np.rot90(b)
                self.data_buffer.append([rotated, result])

                flipped = np.fliplr(rotated)
                self.data_buffer.append([flipped, result])


        sample_size = 512
        if len(self.data_buffer) < sample_size:
            return
        
        sample_data = random.sample(self.data_buffer, sample_size)
        train_input = np.zeros((sample_size, self.width, self.height, 3))
        train_output = np.ones((sample_size, 3))
        for i in range(0, sample_size):
            train_input[i, :] = sample_data[i][0]
            train_output[i, :] = sample_data[i][1]
            
        self.model.fit(x=train_input, y=train_output, epochs=5)
        self.save()        

    def get_next_move(self, board, player):
        if self.training and self.e > random.random():
            availables = board.available_moves_pair()
            return random.choice(availables)
            
        predict_result = self.predict(board, player)
        next_move = predict_result[0][0]
        return next_move

    def predict(self, board, player):
        availables = board.available_moves_pair()
        #predict_input = np.zeros((len(availables), board.width(), board.height(), 3))
        
        predict_input = []
        for i in range(len(availables)):
            move = availables[i]
            b = copy.deepcopy(board.board)
            b[move[0], move[1], player] = 1
            predict_input.append(b)
            #predict_input[i, :] = b
        predict_input = np.array(predict_input)
        scores = self.model.predict(predict_input)
        result = []
        for i in range(len(availables)):
            result.append((availables[i], scores[i, player]))

        result.sort(key=lambda x : x[1], reverse=True)
        return result
