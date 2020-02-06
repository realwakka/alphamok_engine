from random import randint
import math

class TreeNode:
    def __init__(self, parent):
        self.played = 0
        self.won = 0
        self.parent = parent
        self.player = parent.player == 1 ? 2 : 1
        self.children = []

    def get_root(self):
        root = self
        while root.parent != None:
            root = root.parent
        return root

    def get_move_by_child(self, target_child):
        for move, child in children:
            if target_child == child:
                return move

        raise NameError("child not found")
        

    def get_utc(self):
        t = self.get_root().played
        c = math.sqrt(2)
        wi = self.won
        ni = self.played
        return wi/ni + c * sqrt(log(t) / ni)

    def get_board(self):
        board = Board(15,15)
        node = self
        while node.parent != None:
            move = node.parent.get_move_by_child(self. node)
            board.move(move[0], move[1])
            node = node.parent
        return board

    def select(self):
        max(self.children, key = lambda move, child : child.get_utc())

    def expand(self):
        board = self.get_board()
        availables = board.available_moves()

    def simulate(self, playout):
        class TrainingPlayer:
            def get_next_move(self, board, player):
                availables = board.available_moves()
                return availables[randint(0, len(availables) - 1)]

        player = TrainningPlayer()

        win = 0
        for i in range(playout):
            referee = Referee(15, 15)
            game_state = referee.start_game(player, player)
            if player == game_state:
                win += 1

        self.won += win
        self.played = playout

            

class MCTS:
    def __init__(self):
        self.root = TreeNode(None, (-1, -1))
        self.curr_node = self.root;

    def train_self(self):
        selected_node = self.root.select()
        

class MCTSPlayer:
    def __init__(self):
        self.mcts = MCTS()

    def prestart(self):
        mcts.train_self()
        
    





