from game import Referee, Board
from mcts import MCTSPlayer
from ai_player import AIPlayer
from network import PolicyValueNet
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
    
    player1 = AIPlayer(width, height, True)
    player2 = CommandPlayer()

    for i in range(50000):
        board = Board(width,height)
        referee = Referee()
        result = referee.start_game(board, player1, player1)
        print(board)
        print("win : " + str(result) + "\n")
        print("episode" + str(i) + "\n")

    player1.save()
    
if __name__ == "__main__":
    net = PolicyValueNet(15,15)
    # execute only if run as a script
    main()
