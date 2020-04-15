from mcts import MCTreeSearch

class MCTSAgent:
    def __init__(self, model, nr_sim=101, training=True):
        self.training = training
        self.mt = MCTreeSearch(model, nr_sim=101)

    def move(self, game, mc=0):
        if self.training:
            t = 0 if mc > 14 else 1
        else:
            t = 0
        return self.mt.get_move(game, mc)
