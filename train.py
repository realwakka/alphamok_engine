from game import Referee, Board
from mcts import MCTSPlayer
from ai_player import AIPlayer
from network import PolicyValueNet

def main():
    width = 15
    height = 15
    net = PolicyValueNet(width, height)
    player = MCTSPlayer(net, is_selfplay=True)
    
    for i in range(500):
        board = Board(15,15)
        referee = Referee()
        result = referee.start_game(board, player1, player1)
        print("win : " + str(result) + "\n")
        print("episode" + str(i) + "\n")

    player1.save()
    
if __name__ == "__main__":
    net = PolicyValueNet(15,15)
    # execute only if run as a script
    main()
