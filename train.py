from game import Referee, Board, GameState
from mcts import MCTSPlayer
from ai_player import AIPlayer
from collections import defaultdict, deque
from network import PolicyValueNet
import numpy as np
import random

class Trainer:
    def __init__(self, width, height, net):
        self.buffer_size = 10000
        self.batch_size = 512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.net = net
        self.width = width
        self.height = height

        
    def get_equi_data(self, play_data):
        extend_data = []
        for state, mcts_porb, winner in play_data:
            equi_state = state
            equi_mcts_prob = mcts_porb.reshape((self.width, self.height))
            
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.rot90(m=equi_state, k=1, axes=(0,1))
                equi_mcts_prob = np.rot90(m=equi_mcts_prob, axes=(0,1))
                extend_data.append((equi_state, equi_mcts_prob.flatten(), winner))
                
                # flip horizontally
                flipped_state = np.fliplr(equi_state)
                flipped_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((flipped_state, flipped_mcts_prob.flatten(), winner))


        return extend_data

    def simulate(self, board, player):
        states, mcts_probs, current_players = [], [], []
        stone_color = 1
        referee = Referee()
        while True:
            next_move, prob = player.get_next_move(board, stone_color)
            board.move_pos(next_move, stone_color)

            current_players.append(stone_color)
            mcts_probs.append(prob)
            states.append(np.copy(board.board))
            game_state = referee.get_game_state_hint(board, next_move, stone_color)
            if game_state < 3:
                break
            stone_color = 2 if stone_color == 1 else 1

        print("GameEnd : " + str(game_state))
        winners_z = np.zeros(len(current_players))
        winner = game_state
        print(board)
        if game_state != 0:
            winners_z[np.array(current_players) == winner] = 1.0
            winners_z[np.array(current_players) != winner] = -1.0
        # reset MCTS root node
        player.reset_player()

        episode_data = self.get_equi_data(zip(states, mcts_probs, winners_z))
        
        self.data_buffer.extend(episode_data)

    def train(self):
        if len(self.data_buffer) < self.batch_size:
            return

        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        self.net.train(state_batch, mcts_probs_batch, winner_batch)
    

def main():
    width = 15
    height = 15
    net = PolicyValueNet(width, height)
    player = MCTSPlayer(net, n_playout=100, is_selfplay=True)
    trainer = Trainer(width, height, net)
    for i in range(1500):
        print("episode " + str(i) + "...\n")
        board = Board(width,height)
        trainer.simulate(board,player)
        trainer.train()

       
    
if __name__ == "__main__":
    main()
