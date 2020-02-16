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
            tf.keras.layers.Dense(1)
        ])

        self.model.compile(optimizer='adam',
                           loss='mean_squared_error',
                           metrics=['accuracy'])
        print(self.model.summary())
        

    def reverse_player(self, board):
        reversed = board
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
        if win_player == 0:
            return
        
        # if win_player == 1 increase value, win_player == 2 decrease value
        sample_size = len(history)
        train_input = np.zeros((sample_size * 4, self.width, self.height, 3))
        
        for i in range(sample_size):
            board = history[i]
            train_input[i*4, :] = board.board
            train_input[i*4+1, :] = np.rot90(train_input[i*4, :])
            train_input[i*4+2, :] = np.rot90(train_input[i*4+1, :])
            train_input[i*4+3, :] = np.rot90(train_input[i*4+2, :])
        
        predict_data = self.model.predict(train_input)
        
        train_output = predict_data
        if win_player == 1:
            train_output += 1
        elif win_player == 2:
            train_output -= 1
            
        #self.save()

        self.model.fit(x=train_input, y=train_output, epochs=10)


    def get_next_move(self, board, player):
        predict_result = self.predict(board, player)
        next_move = predict_result[0][0]
        return next_move

    def predict(self, board, player):
        availables = board.available_moves()
        predict_input = np.zeros((len(availables), board.width(), board.height(), 3))

        for i in range(len(availables)):
            move = availables[i]
            b = copy.deepcopy(board)
            b.move(move[0], move[1], player)
            predict_input[i, :] = b.board
            
        scores = self.model.predict(predict_input)
        result = []
        for i in range(len(availables)):
            result.append((availables[i], scores[i]))

        if player == 1:
            result.sort(key=lambda x : x[1], reverse=True)
        else:
            result.sort(key=lambda x : x[1], reverse=False)
            
        return result


    


