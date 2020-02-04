class TreeNode:
    def __init__(self, parent):
        self.played = 0
        self.won = 0
        self.parent = parent
        self.children = []

    def select(self):
        max_winrate = max(self.children, key = lambda move, child : child.won / child.played)

    def simulate(self):
        

class MCTS:
    def __init__(self):
        self.root = TreeNode(None)

