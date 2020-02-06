from random import randint
from game import Referee, Board
import math

class TreeNode:
    def __init__(self, parent):
        self.played = 0
        self.win = 0
        self.parent = parent
        if parent == None or parent.player == 2:
            self.player = 1
        else:
            self.player = 2

        self.children = []

    def get_root(self):
        root = self
        while root.parent != None:
            root = root.parent
        return root

    def get_move_by_child(self, target_child):
        for move, child in self.children:
            if target_child == child:
                return move

        raise NameError("child not found")
        

    def get_utc(self):
        t = self.get_root().played
        c = math.sqrt(2)
        wi = self.win
        ni = self.played
        return wi/ni + c * math.sqrt(math.log(t) / ni)

    def get_board(self):
        board = Board(15,15)
        node = self
        while node.parent != None:
            move = node.parent.get_move_by_child(node)
            board.move(move[0], move[1], node.player)
            node = node.parent
        return board

    def select(self):
        if len(self.children) == 0:
            return self
        return max(self.children, key = lambda move, child : child.get_utc())

    def expand(self):
        board = self.get_board()
        availables = board.available_moves()

        if len(availables) == 0:
            raise NameError("no availables")

        next_move = availables[randint(0, len(availables)-1)]
        new_child = TreeNode(self)
        self.children.append((next_move, new_child))
        return new_child

    def update(self, win, played, player):
        if player == self.player:            
            self.win += win
        self.played += played

        if self.parent != None:
            self.parent.update(win, played, player) 

    def simulate(self, playout):
        class TrainingPlayer:
            def prestart(self):
                pass

            def get_next_move(self, board, player):
                availables = board.available_moves()
                return availables[randint(0, len(availables) - 1)]

        player = TrainingPlayer()

        win = 0
        for i in range(playout):
            referee = Referee()
            board = Board(15, 15)
            game_state = referee.start_game(board, player, player)
            if player == game_state:
                win += 1

        return win

            

class MCTS:
    def __init__(self):
        self.root = TreeNode(None)
        self.curr_node = self.root

    def train_self(self):
        selected_node = self.root.select()
        expanded_node = selected_node.expand()
        win = expanded_node.simulate(100)
        expanded_node.update(win, 100, expanded_node.player)

    def get_next_move(self, board, player):
        if len(self.curr_node.children) == 0:
            expanded_node = self.curr_node.expand()
            win = expanded_node.simulate(100)
            expanded_node.update(win, 100, expanded_node.player)
            #raise NameError("I don't know!!")
        
        next_move, next_child = max(self.curr_node.children, key=lambda p: p[1].get_utc())
        self.curr_node = next_child
        return next_move
        

class MCTSPlayer:
    def __init__(self):
        self.mcts = MCTS()

    def prestart(self):
        self.mcts.train_self()
    def get_next_move(self, board, player):
        return self.mcts.get_next_move(board, player)
        
    





