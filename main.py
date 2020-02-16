from game import Referee, Board
from mcts import MCTSPlayer
from ai_player import AIPlayer

class CommandPlayer:
    def prestart(self):
        pass

    def get_next_move(self, board, player):
        print(board)
        x = int(input("x = "))
        y = int(input("y = "))
        return (x, y)

def main():

    width = 15
    height = 15
    
    player1 = AIPlayer(width, height)
    player2 = CommandPlayer()

    for i in range(500):
        board = Board(15,15)
        referee = Referee()
        result = referee.start_game(board, player1, player1)

    player1.save()
        
    
    
    
if __name__ == "__main__":
    # execute only if run as a script
    main()
