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
        states = []
        actions = []
        v = None
        while True:
            state = game.state()
            s_hash = _array_hash(state)
            player = game.turn

            if game.winner:
                v = abs(game.winner)

            nn_format = self._state(state, player)
            allowed_moves = game.allowed_moves()
            moves_size = len(allowed_moves)

            if v is None and not self.visited.get(s_hash, False):
                pol, v = self.model.predict(nn_format.reshape(1, *nn_format.shape))

                pol = pol[0]
                v = v[0]

                pol *= allowed_moves
                if np.sum(pol) == 0:
                    print("policy zero out")
                    self.policy[s_hash] = allowed_moves
                else:
                    self.policy[s_hash] = pol / np.sum(pol)
                self.visited[s_hash] = True
                self.N[s_hash] = 0
                self.Na[s_hash] = np.zeros(moves_size)
                self.Q[s_hash] = np.zeros(moves_size)
                self.W[s_hash] = np.zeros(moves_size)

                v = -v

            if v is not None and len(actions) > 0:
                for state, a in zip(reversed(states), reversed(actions)):
                    self.W[state][a] += v
                    self.Na[state][a] += 1
                    self.Q[state][a] = self.W[state][a] / self.Na[state][a]

                    self.N[state] += 1
                    v *= -1
                return

            u = np.ma.array(
                self.Q[s_hash] + self.C * self.policy[s_hash] * np.sqrt((self.N[s_hash] + 1e-10)/ (1 + self.Na[s_hash])),
                mask=abs(1 - allowed_moves),
                fill_value=float('-inf')
            )

            action_best = np.argmax(u)
            game.move(action_best)
            
            actions.append(action_best)
            states.append(s_hash)

    def _state(self, state, player):
        ones = (state == 1).astype(np.float)
        m_ones = (state == -1).astype(np.float)
        if player == 1:
            return np.stack((ones, m_ones), axis=-1)
        return np.stack((m_ones, ones), axis=-1)

    def get_move(self, game, t):
        for _ in range(self.nr_sim):
            self.expand(deepcopy(game))

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
