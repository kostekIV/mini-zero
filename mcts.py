import numpy as np

from copy import deepcopy


def _array_hash(a):
    return hash(a.data.tobytes())

class MCTreeSearch:
    def __init__(self, model, C=1, nr_sim=800):
        self.model = model
        self.C = C
        self.nr_sim = nr_sim
        self.Q = {}
        self.W = {}
        self.N = {}
        self.Na = {}
        self.policy = {}
        self.visited = {}

    def expand(self, game):
        state = game.state()
        s_hash = _array_hash(state)
        player = game.turn

        if game.winner:
            return abs(game.winner)

        nn_format = self._state(state, player)
        moves_size = len(game.allowed_moves())

        if not self.visited.get(s_hash, False):
            pol, v = self.model.predict(nn_format.reshape(1, *nn_format.shape))

            pol = pol[0]
            v = v[0]

            pol *= game.allowed_moves()
            self.policy[s_hash] = pol / np.sum(pol)
            self.visited[s_hash] = True
            self.N[s_hash] = 0
            self.Na[s_hash] = np.zeros(moves_size)
            self.Q[s_hash] = np.zeros(moves_size)
            self.W[s_hash] = np.zeros(moves_size)

            return -v

        action_best = None,
        val_best = float('-inf')

        for action, allowed in enumerate(game.allowed_moves()):
            if allowed:
                qa = self.Q[s_hash][action]
                na = self.Na[s_hash][action]
                n = self.N[s_hash]
                pa = self.policy[s_hash][action] 
                val = qa + self.C * pa * np.sqrt(n / (1 + na))
                if val > val_best:
                    action_best = action
                    val_best = val

        game.move(action_best)
        v = self.expand(game)
        
        if s_hash in self.Na.keys():
            self.W[s_hash][action_best] += val_best
            self.Na[s_hash][action_best] += 1
            self.Q[s_hash][action_best] = self.W[s_hash][action_best] / self.Na[s_hash][action_best]
        else:
            self.W[s_hash][action_best] = val_best
            self.Q[s_hash][action_best] = self.W[s_hash][action_best]
            self.Na[s_hash][action_best] = 1

        self.N[s_hash] += 1

    def _state(self, state, player):
        ones = (state == 1).astype(np.float)
        m_ones = (state == -1).astype(np.float)
        if player == 1:
            return np.stack((ones, m_ones), axis=-1)
        return np.stack((m_ones, ones), axis=-1)

    def get_move(self, game, mc=0):
        for _ in range(self.nr_sim):
            self.expand(deepcopy(game))

        t = 1
        if mc > 14:
            t = 0
        phi = self.get_phi(game.state(), t)
        move = np.random.choice(len(phi), p=phi)

        return move, phi, self._state(game.state(), game.turn)

    def get_phi(self, state, t=1):
        s_hash = _array_hash(state)

        nas = self.Na[s_hash]
        phi = np.zeros(nas.shape)
        if t == 0:
            phi[np.argmax(nas)] = 1
            return phi
        
        phi = nas ** (1/t)
        phi = phi / sum(phi)

        return phi
