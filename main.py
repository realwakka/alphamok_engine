from game import Referee

class Player:
    def get_next_move(self, board, player):
        print(board)
        x = int(input("x = "))
        y = int(input("y = "))
        return (x, y)

def main():
    player = Player()
    referee = Referee(10, 10)
    referee.start_game(player, player)
    
    
    
if __name__ == "__main__":
    # execute only if run as a script
    main()
