import tensorflow as tf
import numpy as np

model_file = "mcts.h5"

class PolicyValueNet:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        try:
            self.model = tf.keras.models.load_model(model_file)
            return
        except OSError:
            print('no model file... creating new model...')

        inputs = tf.keras.Input(shape=(width, height, 5))
        
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
        availables = board.availables
        state = board.current_state(player)
        act_probs, value = self.model.predict(state.reshape((-1, 15, 15, 5)))
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

        self.model.save(model_file)
        return loss[0], entropy




        
        