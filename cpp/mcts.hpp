#pragma once

#include <unordered_map>
#include <vector>
#include <numeric>
#include <cmath>
#include <mutex>
#include <thread>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>

#include "game.hpp"
#include "model.hpp"


class mcts {
private:
  gsl_rng * r;
  model *m;
  int nr_sim;
  double c = 1;
  std::vector<double> alphas;
  gsl_vector* th;
  gsl_vector* am_vec;
  gsl_vector* u_vec;
  gsl_vector* temp_vec;
  std::unordered_map<TicTacToe, double> N;
  std::unordered_map<TicTacToe, gsl_vector*> ps;
  std::unordered_map<TicTacToe, gsl_vector*> W;
  std::unordered_map<TicTacToe, gsl_vector*> Q;
  std::unordered_map<TicTacToe, gsl_vector*> Na;
public:
  mcts() {};
  mcts(const mcts&) = delete;
  mcts& operator=(const mcts&) = delete;
  mcts(model *m, int nr_sim, gsl_rng *r = NULL): m(m), nr_sim(nr_sim), r(r), alphas(100, 0.2) {
    th = gsl_vector_alloc(100);
    am_vec = gsl_vector_alloc(100);
    u_vec = gsl_vector_alloc(100);
    temp_vec = gsl_vector_alloc(100);
  };

  void expand(TicTacToe t) {
    float v = -10;
    std::vector<TicTacToe> states;
    std::vector<int> actions;

    while (true) {
      if (t.get_winner() != 2) {
        v = -t.get_winner() * t.get_winner();
      }
      auto allowed_moves = t.allowed_moves();
      for (int i = 0; i < allowed_moves.size(); i++) {
        gsl_vector_set(am_vec, i, allowed_moves[i]);
      }
      auto state = t.state();
      if (ps.find(t) == ps.end()) {
        auto v_pol = m->predict(state.data());
        v = v_pol.first;
        auto pol = v_pol.second;


        auto pol_vec = gsl_vector_alloc(pol.size());

        auto w = gsl_vector_alloc(pol.size());
        auto q = gsl_vector_alloc(pol.size());
        auto na = gsl_vector_alloc(pol.size());
        gsl_vector_set_zero(w);
        gsl_vector_set_zero(q);
        gsl_vector_set_zero(na);
        for (int i = 0; i < pol.size(); i++) {
          gsl_vector_set(pol_vec, i, pol[i]);
        }
        if (r != NULL) {
          gsl_vector_set_zero(th);
          gsl_ran_dirichlet(r, pol.size(), alphas.data(), gsl_vector_ptr(th, 0));
          gsl_vector_scale(th, 0.25);
          gsl_blas_daxpy(0.75, pol_vec, th);
          // for (int i = 0; i < pol.size(); i++) {
          //   pol[i] = 0.75 * pol[i] + 0.25 * thethas[i];
          // }
        }

        gsl_vector_mul(pol_vec, am_vec);
        auto pol_sum = gsl_blas_dasum(pol_vec);
        if (pol_sum == 0) {
          gsl_vector_memcpy(pol_vec, am_vec);
          pol_sum = gsl_blas_dasum(pol_vec);
        }
        gsl_vector_scale(pol_vec, 1 / pol_sum);
        // for (int i = 0; i < pol.size(); i++) {
        //   pol[i] *= allowed_moves[i];
        // }
        // auto pol_sum = std::accumulate(pol.begin(), pol.end(), 0.0);
        // auto am_sum = std::accumulate(allowed_moves.begin(), allowed_moves.end(), 0.0);
        // for (int i = 0; i < pol.size(); i++) {
        //   if (pol_sum != 0) {
        //     pol[i] /= pol_sum;
        //   } else {
        //     pol[i] = allowed_moves[i] / am_sum;
        //   }
        // }


        ps[t] = pol_vec;
        N[t] = 0;
        W[t] = w;
        Q[t] = q;
        Na[t] = na;
        // ps[t] = pol_vec;
        // N[t] = 0;
        // W.try_emplace(t, allowed_moves.size(), 0);
        // Q.try_emplace(t, allowed_moves.size(), 0);
        // Na.try_emplace(t, allowed_moves.size(), 0);
      }

      if (v != -10) {
        for (int i = states.size() - 1; i >= 0; i--) {
          v *= -1;

          auto a = actions[i];
          auto state = states[i];

          gsl_vector_set(W[state], a, v + gsl_vector_get(W[state], a));
          gsl_vector_set(Na[state], a, 1 + gsl_vector_get(Na[state], a));
          gsl_vector_set(Q[state], a, v + gsl_vector_get(W[state], a) / gsl_vector_get(Na[state], a));

          // Na[state][a] += 1;
          // Q[state][a] = W[state][a] / Na[state][a];

          N[state] += 1;
        }
        return;
      }

      double best_u = INT32_MIN;
      double n = sqrt(N[t]);

      gsl_vector_memcpy(u_vec, ps[t]);
      gsl_vector_memcpy(temp_vec, Na[t]);
      gsl_vector_add_constant(temp_vec, 1);
      gsl_vector_div(u_vec, temp_vec);
      gsl_blas_daxpy(c*n, u_vec, Q[t]);
      gsl_vector_set_all(temp_vec, -1);
      gsl_vector_add(temp_vec, am_vec);
      gsl_vector_scale(temp_vec, 1000);
      gsl_vector_add(u_vec, temp_vec);

      auto best_a = gsl_vector_max_index(u_vec);
      // for (int i = 0; i < 100; i++)
      //   std::cout << gsl_vector_get(u_vec, i) << " ";
      // std::cout << "\n";
      // std::cout << best_a << "\n";
      // for (int i = 0; i < allowed_moves.size(); i++) {
      //   if (allowed_moves[i] == 1) {
      //     float u = Q[t][i] + c * ps[t][i] * n/(Na[t][i] + 1);
      //     if (u > best_u) {
      //       best_a = i;
      //       best_u = u;
      //     }
      //   }
      // }
      states.push_back(t);
      actions.push_back(best_a);
      t.move(best_a);
    }
  }

  std::vector<float> expand_probs(TicTacToe &t) {
    for (int i = 0; i < nr_sim; i++) {
      expand(t);
    }

    std::vector<float> phi(t.allowed_moves().size());
    auto Nsa = Na[t];
    auto Ns = 1 / N[t];
    gsl_vector_memcpy(temp_vec, Nsa);
    gsl_vector_scale(temp_vec, Ns);
    // auto Ns = std::accumulate(Nsa.begin(), Nsa.end(), 0.);
    for (int i = 0; i < phi.size(); i++) {
      phi[i] = gsl_vector_get(temp_vec, i);
    }

    return phi;
  }

  ~mcts() {
    gsl_vector_free(temp_vec);
    gsl_vector_free(u_vec);
    gsl_vector_free(am_vec);
    gsl_vector_free(th);

    for(auto const& x : ps)
      gsl_vector_free(x.second);
    for(auto const& x : W)
      gsl_vector_free(x.second);
    for(auto const& x : Q)
      gsl_vector_free(x.second);
    for(auto const& x : Na)
      gsl_vector_free(x.second);

    ps.clear();
    W.clear();
    Q.clear();
    Na.clear();
  }
};
