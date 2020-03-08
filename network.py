import tensorflow as tf
import numpy as np

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

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available action and the score of the board state
        """
        legal_positions = board.availables
        current_state = board.current_state()
        act_probs, value = self.policy_value(current_state.reshape(-1, 4, self.board_width, self.board_height))
        act_probs = zip(legal_positions, act_probs.flatten()[legal_positions])
        return act_probs, value[0][0]

    def policy_value(self, board, player):
        availables = board.available_moves()

        act_probs, value = self.model.predict(board.board.reshape(-1, self.width, self.height, 3))
        act_probs =  zip(availables, act_probs.flatten()[availables])
        return act_probs, value[0][0]

    def train(self, state_input, mcts_probs, winner):
        state_input_union = np.array(state_input)
        mcts_probs_union = np.array(mcts_probs)
        winner_union = np.array(winner)
        loss = self.model.evaluate(state_input_union, [mcts_probs_union, winner_union], batch_size=len(state_input), verbose=0)
        action_probs, _ = self.model.predict_on_batch(state_input_union)

        def self_entropy(probs):
            return -np.mean(np.sum(probs * np.log(probs + 1e-10), axis=1))

        entropy = self_entropy(action_probs)
        # K.set_value(self.model.optimizer.lr, learning_rate)
        self.model.fit(state_input_union, [mcts_probs_union, winner_union], batch_size=len(state_input), verbose=0)
        return loss[0], entropy




        
        