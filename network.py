import tensorflow as tf
import numpy as np
import 
class PolicyValueNet:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        inputs = tf.keras.Input(shape=(width, height, 3))
        
        network = tf.keras.layers.Conv2D(32, (3,3), activation='relu')(inputs)
        network = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(network)
        network = tf.keras.layers.Conv2D(128, (3,3), activation='relu')(network)

        policy_net = tf.keras.layers.Conv2D(4, (1,1), activation='relu')(network)
        policy_net = tf.keras.layers.Flatten()(policy_net)
        self.policy_net = tf.keras.layers.Dense(self.width * self.height, activation='tanh')(policy_net)

        value_net = tf.keras.layers.Conv2D(2, (1,1), activation='relu')(network)
        value_net = tf.keras.layers.Flatten()(value_net)
        value_net = tf.keras.layers.Dense(64)(value_net)
        self.value_net = tf.keras.layers.Dense(1, activation='tanh')(value_net)

        self.model = tf.keras.Model(inputs=inputs, outputs=[self.policy_net, self.value_net])
        
        self.model.compile(optimizer='adam',
                           loss=['categorical_crossentropy', 'mean_squared_error'],
                           metrics=['accuracy'])

    def policy_value(self, board, player):
        availables = board.available_moves()
        predict_input = np.zeros((len(availables), board.width(), board.height(), 3))

        for i in range(len(availables)):
            move = availables[i]
            b = np.copy(board.board)
            b[move[0], move[1], player] = 1
            predict_input[i, :] = b


        
        