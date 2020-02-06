from game import Referee, Board
from mcts import MCTSPlayer

class Player:
    def prestart(self):
        pass

    def get_next_move(self, board, player):
        print(board)
        x = int(input("x = "))
        y = int(input("y = "))
        return (x, y)

def main():
    player = Player()
    player1 = MCTSPlayer()
    board = Board(15,15)
    referee = Referee()
    referee.start_game(board, player1, player)
    
    
    
if __name__ == "__main__":
    # execute only if run as a script
    main()
