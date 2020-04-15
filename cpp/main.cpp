#include <tensorflow/c/c_api.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <iomanip>
#include <random>
#include <string>

#include "model.hpp"
#include "game.hpp"
#include "mcts.hpp"

#include "cnpy.h"

#define N 150


int main(int argc, char *argv[]) {
  int model_id = atoi(argv[1]);
  std::ostringstream s;
  s << "./models/mini-zero-" << model_id;
  model m(s.str().c_str(), "serve");
  std::ostringstream sp;
  sp << "./data/" << model_id << "/";
  std::string save_path = sp.str();
  std::random_device r;
  std::default_random_engine generator(r());
  std::vector<float> phis;
  std::vector<float> states;
  std::vector<float> vs;
  uint size = 0;
  float xs = 0;
  float ys = 0;
  for (int _k = 0; _k < N; _k++) {
    int gc = 0;
    auto g = TicTacToe(9, 9, 5);
    auto mct1 = mcts(m, 201);
    auto mct2 = mcts(m, 201);
    mcts x[] = {mct1, mct2};
    while (g.get_winner() == 2) {
      auto mct = x[gc % 2];
      auto phi = mct.expand_probs(g);
      auto state = g.state();
      std::discrete_distribution<int> du(phi.begin(), phi.end());

      gc += 1;
      size += 1;
      std::copy(phi.begin(), phi.end(), std::back_inserter(phis));
      std::copy(state.begin(), state.end(), std::back_inserter(states));

      auto move = du(generator);
      if (_k % 50 == 0) {
        std::cout << g;
      }
      g.move(move);
    }
    if (g.get_winner() == 1) {
      xs += 1;
    } else if (g.get_winner() == -1) {
      ys += 1;
    }
    if (_k % 50 == 0) {
      std::cout << g;
    }
    if ((_k + 1) % 10 == 0) {
      float xwr = xs / (_k + 1);
      float ywr = ys / (_k + 1);
      std::cout << "iteration: " << _k << "\n";
      std::cout << "win rate of X: " << xwr << "\n";
      std::cout << "win rate of Y: " << ywr << "\n";
    }
    auto winner = g.get_winner();
    for(int i = 0; i < gc; i++) {
      vs.push_back(winner);
      winner *= -1;
    }
  }
  cnpy::npy_save(save_path + "phis.npy", phis.data(), {size, 81}, "w");
  cnpy::npy_save(save_path + "vals.npy", vs.data(), {size, 1}, "w");
  cnpy::npy_save(save_path + "states.npy", states.data(), {size, 9, 9, 2}, "w");
}