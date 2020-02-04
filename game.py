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
        
        for i in range(self.board.shape[3]):
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
        for i in range(width):
            for j in range(height):
                if self.board[i, j, 0] == 1:
                    ret.append((i, j))
                
        return ret





class Referee:
    def __init__(self, width, height):
        self.board = Board(width, height)

    def max_combo(self, x, y, direction_func, prev_player):
        try:
            player = self.board.get(x, y)
        except:
            return 0
        
        if player == prev_player:
            nx, ny = direction_func(x, y)
            return self.max_combo(nx, ny, direction_func, player)
        else:
            return 0

    def get_game_state(self):
        cache = np.zeros((width,height))
        direction_pairs = [(lambda x, y : x + 1, y, lambda x, y : x - 1, y),
                           (lambda x, y : x, y + 1, lambda x, y : x, y - 1),
                           (lambda x, y : x - 1, y + 1, lambda x, y : x + 1, y - 1),
                           (lambda x, y : x + 1, y + 1, lambda x, y : x + 1, y + 1)]

        for i in range(self.board.width):
            for j in range(self.board.height):
                player = self.board.get(i, j)
                if player == 0:
                    continue
                
                for direction_pair : direction_pairs:
                    combo = self.max_combo(i, j, direction_pair[0], player)
                    combo += self.max_combo(i, j, direction_pair[1], player)
                if combo == 5:
                    return player + 2

                
         
        
        

    def start_game(self, player1, player2):
        while True:
            try:
                next_move = player1.get_next_move(self.board, 1);
                self.board.move(next_move)
            except:
                continue
            break

        
        while True:
            try:
                next_move = player2.get_next_move(self.board, 2);
                self.board.move(next_move)
            except:
                continue
            break
            


