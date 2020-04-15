#pragma once

#include <unordered_map>
#include <vector>
#include <numeric>
#include <cmath>

#include "game.hpp"
#include "model.hpp"


class mcts {
private:
  std::unordered_map<TicTacToe, float> N;
  std::unordered_map<TicTacToe, std::vector<float>> ps;
  std::unordered_map<TicTacToe, std::vector<float>> W;
  std::unordered_map<TicTacToe, std::vector<float>> Q;
  std::unordered_map<TicTacToe, std::vector<float>> Na;
  model m;
  int nr_sim;
  float c = 1;
public:
  mcts(model m, int nr_sim): m(m), nr_sim(nr_sim) {};

  float expand(TicTacToe &t) {
    if (t.get_winner() != 2) {
      return t.get_winner() * t.get_winner();
    }
    auto allowed_moves = t.allowed_moves();
    auto state = t.state();
    if (ps.find(t) == ps.end()) {
      auto v_pol = m.predict(state.data());
      auto v = v_pol.first;
      auto pol = v_pol.second;

      for (int i = 0; i < pol.size(); i++) {
        pol[i] *= allowed_moves[i];
      }
      auto pol_sum = std::accumulate(pol.begin(), pol.end(), 0);
      for (int i = 0; i < pol.size(); i++) {
        if (pol_sum != 0) {
          pol[i] /= pol_sum;
        } else {
          pol[i] = allowed_moves[i];
        }
      }

      ps[t] = pol;
      N[t] = 0;
      W.try_emplace(t, allowed_moves.size(), 0);
      Q.try_emplace(t, allowed_moves.size(), 0);
      Na.try_emplace(t, allowed_moves.size(), 0);

      return -v;
    }

    int best_a = -1;
    float best_u = INT32_MIN;
    for (int i = 0; i < allowed_moves.size(); i++) {
      if (allowed_moves[i] == 1) {
        float n = sqrt(N[t]);
        if (n == 0) {
          n += 1e-10;
        }
        float u = Q[t][i] + c * ps[t][i] * n/(Na[t][i] + 1);
        if (u > best_u) {
          best_a = i;
          best_u = u;
        }
      }
    }
    auto t_cp = t;
    t_cp.move(best_a);
    auto v = expand(t_cp);

    W[t][best_a] += v;
    Na[t][best_a] += 1;
    Q[t][best_a] = W[t][best_a] / Na[t][best_a];

    N[t] += 1;

    return -v;
  }

  std::vector<float> expand_probs(TicTacToe &t) {
    for (int i = 0; i < nr_sim; i++) {
      expand(t);
    }

    std::vector<float> phi(t.allowed_moves().size());
    auto Ns = N[t];
    auto Nsa = Na[t];
    for (int i = 0; i < phi.size(); i++) {
      phi[i] = Nsa[i] / Ns;
    }

    return phi;
  }
};
