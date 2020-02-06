import tensorflow as tf
import numpy as np

from tensorflow.keras import layers, models

class Selector:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.input_state = tf.placeholder(
            tf.float, shape=[None, 4, height, width])

        self.model = models.Sequantial()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape(width, height, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    def train(self, board, value):
        origin_b = board.board
        

        

        

        
        
        
