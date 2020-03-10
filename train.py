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

    def max_combo(self, board, x, y, direction_func, prev_player):
        try:
            player = board.get(x, y)
        except NameError:
            return 0

        if player == prev_player:
            nx, ny = direction_func(x, y)
            return self.max_combo(board, nx, ny, direction_func, player) + 1
        else:
            return 0


    def get_game_state(self, board):
        cache = np.zeros((board.width(),board.height()))
        direction_pairs = [(lambda x, y : (x + 1, y), lambda x, y : (x - 1, y)),
                           (lambda x, y : (x, y + 1), lambda x, y : (x, y - 1)),
                           (lambda x, y : (x - 1, y + 1), lambda x, y : (x + 1, y - 1)),
                           (lambda x, y : (x + 1, y + 1), lambda x, y : (x - 1, y - 1))]

        is_full = True
        moved_count = 0

        for i in range(board.height()):
            for j in range(board.width()):
                player = board.get(i, j)
                if player == 0:
                    is_full = False
                    continue
                
                moved_count += 1
                
                for direction_pair in direction_pairs:
                    combo = self.max_combo(board, i, j, direction_pair[0], player)
                    combo += self.max_combo(board, i, j, direction_pair[1], player)
                    if combo == 6:
                        return player

        if is_full:
            return GameState.END_DRAW

        return moved_count % 2 + 3
        
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
        
        while True:
            next_move, prob = player.get_next_move(board, stone_color)
            board.move_pos(next_move, stone_color)
            
            current_players.append(stone_color)
            mcts_probs.append(prob)
            states.append(np.copy(board.board))

            game_state = self.get_game_state(board)
            if game_state < 3:
                break
            stone_color = 2 if stone_color == 1 else 1
            print(board)
            input()

        print("GameEnd : " + str(game_state))
        winners_z = np.zeros(len(current_players))
        winner = game_state

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
    player = MCTSPlayer(net, n_playout=400, is_selfplay=False)
    trainer = Trainer(15, 15, net)
    for i in range(1000):
        print("episode " + str(i) + "...\n")
        board = Board(15,15)
        trainer.simulate(board,player)
        trainer.train()

       
    
if __name__ == "__main__":
    main()
