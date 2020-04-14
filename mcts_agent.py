from mcts import MCTreeSearch

class MCTSAgent:
    def __init__(self, model):
        self.mt = MCTreeSearch(model, nr_sim=120)

    def move(self, game, mc=0):
        return self.mt.get_move(game, mc)
