from game import Referee, Board
from mcts import MCTSPlayer
from ai_player import AIPlayer

class Player:
    def prestart(self):
        pass

    def get_next_move(self, board, player):
        print(board)
        x = int(input("x = "))
        y = int(input("y = "))
        return (x, y)

def main():
    
    board = Board(15,15)
    referee = Referee()
    player1 = AIPlayer(board.width(), board.height(), 1)
    player2 = AIPlayer(board.width(), board.height(), 2)
    result = referee.start_game(board, player1, player2)
    player1.on_finish_game(result)
    player2.on_finish_game(result)
    
    
    
if __name__ == "__main__":
    # execute only if run as a script
    main()
