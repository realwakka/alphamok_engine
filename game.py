import numpy as np
import enum
import random
import copy

def get_game_state(board):
    cache = np.zeros((board.width(),board.height()))
    direction_pairs = [(lambda x, y : (x + 1, y), lambda x, y : (x - 1, y)),
                        (lambda x, y : (x, y + 1), lambda x, y : (x, y - 1)),
                        (lambda x, y : (x - 1, y + 1), lambda x, y : (x + 1, y - 1)),
                        (lambda x, y : (x + 1, y + 1), lambda x, y : (x - 1, y - 1))]

    is_full = True
    moved_count = 0

    for i in range(board.width()):
        for j in range(board.height()):
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

class GameState(enum.Enum):
    END_DRAW = 0
    END_BLACK_WIN = 1
    END_WHITE_WIN = 2
    BLACK_TURN = 3
    WHITE_TURN = 4

class NewBoard():
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.states = {}
        self.n_in_row = 5

        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1
    
    def max_combo(self, x, y, direction_func, prev_player):
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return 0

        
        move = y * self.width + x

        if not move in self.states:
            return 0

        player = self.states[move]

        if player == prev_player:
            nx, ny = direction_func(x, y)
            return self.max_combo(nx, ny, direction_func, player) + 1
        else:
            return 0

    def get_max_combo(self, move):
        direction_pairs = [(lambda x, y : (x + 1, y), lambda x, y : (x - 1, y)),
                        (lambda x, y : (x, y + 1), lambda x, y : (x, y - 1)),
                        (lambda x, y : (x - 1, y + 1), lambda x, y : (x + 1, y - 1)),
                        (lambda x, y : (x + 1, y + 1), lambda x, y : (x - 1, y - 1))]
        

        if move in self.states:
            y = move // self.width
            x = move % self.width
            player = self.states[move]
            max_combo = 0
            for direction_pair in direction_pairs:
                combo = self.max_combo(x, y, direction_pair[0], player)
                combo += self.max_combo(x, y, direction_pair[1], player)
                max_combo = max(combo, max_combo)

            return max_combo
        else:
            return 0
    def __str__(self):
        ret = "\n"

        for i in range(self.height):
            for j in range(self.width):
                move = i * self.width + j
                if move in self.states:
                    ret += str(self.states[move])
                else:
                    ret += " "
                
                ret += " "
            ret += "\n"
        return ret

    def current_state(self, current_player):
        square_state = np.zeros((self.height, self.width, 5))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == current_player]
            move_oppo = moves[players != current_player]
            square_state[move_curr // self.width,
                            move_curr % self.width, 0] = 1.0
            square_state[move_oppo // self.width,
                            move_oppo % self.width, 1] = 1.0
            # indicate the last move location
            square_state[self.last_move // self.width,
                            self.last_move % self.width][2] = 1.0

        for m, s in self.states.items():
            combo = self.get_max_combo(m)
            square_state[m // self.width, m % self.width, 3] = combo

        if current_player == 2:
            square_state[:, :][4] = 1.0  # indicate the colour to play
        return square_state[:, ::-1, :]

    def do_move(self, move, current_player):
        self.states[move] = current_player
        self.availables.remove(move)
        self.last_move = move

    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row *2-1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

class Board:
    def __init__(self, width, height):
        self.states = {}
        
        self.board = np.zeros((width, height, 3))
        self.last_move = (-1, -1)
        for i in range(width):
            for j in range(height):
                self.board[i, j, 0] = 1

    def get_state(self):
        data = np.zeros((self.width, self.height, 5))
        data[1:3, :, :] = self.board


    def is_full(self):
        for i in range(self.height()):
            for j in range(self.width()):
                if self.board[i, j, 0] == 1:
                    return False

        return True

    def width(self):
        return self.board.shape[0]
    
    def height(self):
        return self.board.shape[1]

    def get(self, y, x):
        
        if x < 0 or x >= self.width() or y < 0 or y >= self.height():
            raise NameError("Out of bounds")
        
        for i in range(self.board.shape[2]):
            if self.board[y, x, i] == 1:
                return i

    def move_pos(self, pos, player):
        return self.move(pos // self.width(), pos % self.width(), player)

    def move(self, y, x, player):
        if player == 0 or player > 2:
            raise NameError("Wrong player")

        if x < 0 or x >= self.width() or y < 0 or y >= self.height():
            raise NameError("Out of bounds")

        if self.board[y, x, 0] != 1:
            raise NameError("Already moved")

        self.board[y, x, 0] = 0
        self.board[y, x, player] = 1
        self.last_move = (y, x)


    def available_moves(self):
        ret = []
        for i in range(self.height()):
            for j in range(self.width()):
                if self.board[i, j, 0] == 1:
                    ret.append(i * self.width() + j)
                
        return ret

    def available_moves_pair(self):
        ret = []
        for i in range(self.height()):
            for j in range(self.width()):
                if self.board[i, j, 0] == 1:
                    ret.append((i, j))
                
        return ret

    def __str__(self):
        ret = ""
        for i in range(self.height()):
            ret += "\n"
            for j in range(self.width()):
                ret += " " + str(self.get(i, j))
        return ret
                


class Referee:
    def __init__(self):
        self.id = 1
        self.direction_pairs = [(lambda x, y : (x + 1, y), lambda x, y : (x - 1, y)),
                                (lambda x, y : (x, y + 1), lambda x, y : (x, y - 1)),
                                (lambda x, y : (x - 1, y + 1), lambda x, y : (x + 1, y - 1)),
                                (lambda x, y : (x + 1, y + 1), lambda x, y : (x - 1, y - 1))]
    def get_game_state_hint(self, board, move, player):
        return self.get_game_state_hint_pos(board, move // board.width(), move % board.width(), player)

    def get_game_state_hint_pos(self, board, y, x, player):
        for direction_pair in self.direction_pairs:
            combo = self.max_combo(board, x, y, direction_pair[0], player)
            combo += self.max_combo(board, x, y, direction_pair[1], player)
            if combo == 6:
                return player
        

        if board.is_full():
            return 0
            
        return 3 if player == 2 else 4


    def max_combo(self, board, x, y, direction_func, prev_player):
        try:
            player = board.get(y, x)
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

        for i in range(board.width()):
            for j in range(board.height()):
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
        
    
    def start_game(self, board, player1, player2):
        history = []
        
        player1.prestart()
        player2.prestart()

        while True:
            next_move = player1.get_next_move(board, 1)
            board.move(next_move[0], next_move[1], 1)
            history.append(copy.deepcopy(board))
        
            game_state = self.get_game_state(board)
            if game_state < 3:
                player1.on_finish_game(game_state, history)
                #player2.on_finish_game(game_state)
                return game_state

            next_move = player2.get_next_move(board, 2)
            board.move(next_move[0], next_move[1], 2)
            history.append(copy.deepcopy(board))            

            game_state = self.get_game_state(board)
            if game_state < 3:
                player1.on_finish_game(game_state, history)
                #player2.on_finish_game(game_state)
                return game_state

            


