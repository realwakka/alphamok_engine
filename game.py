import numpy as np
import enum

class GameState(enum.Enum):
    END_DRAW = 0
    END_BLACK_WIN = 1
    END_WHITE_WIN = 2
    BLACK_TURN = 3
    WHITE_TURN = 4

class Board:
    def __init__(self, width, height):
        self.board = np.zeros((width, height, 3))
        for i in range(width):
            for j in range(height):
                self.board[i, j, 0] = 1

    def width(self):
        return self.board.shape[0]
    
    def height(self):
        return self.board.shape[1]

    def get(self, x, y):
        if x < 0 or x >= self.width() or y < 0 or y >= self.height():
            raise NameError("Out of bounds")
        
        for i in range(self.board.shape[2]):
            if self.board[x, y, i] == 1:
                return i
        
    def move(self, x, y, player):
        if player == 0 or player > 2:
            raise NameError("Wrong player")

        if x < 0 or x >= self.width() or y < 0 or y >= self.height():
            raise NameError("Out of bounds")

        if self.board[x, y, 0] != 1:
            raise NameError("Already moved")

        self.board[x, y, 0] = 0
        self.board[x, y, player] = 1

    def available_moves(self):
        ret = []
        for i in range(self.width()):
            for j in range(self.height()):
                if self.board[i, j, 0] == 1:
                    ret.append((i, j))
                
        return ret

    def __str__(self):
        ret = ""
        for i in range(self.width()):
            ret += "\n"
            for j in range(self.height()):
                ret += " " + str(self.get(i, j))
        return ret
                


class Referee:
    def __init__(self):
        pass

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
        player1.prestart()
        player2.prestart()

        while True:
            while True:
                next_move = player1.get_next_move(board, 1)
                board.move(next_move[0], next_move[1], 1)
                break

            game_state = self.get_game_state(board)
            if game_state < 3:
                return game_state
        
            while True:
                next_move = player2.get_next_move(board, 2)
                board.move(next_move[0], next_move[1], 2)
                break
            
            game_state = self.get_game_state(board)
            if game_state < 3:
                return game_state

            


