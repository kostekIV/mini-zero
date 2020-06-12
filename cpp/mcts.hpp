#pragma once

#include <memory>
#include <vector>
#include <random>

#include "node.hpp"
#include "game.hpp"


class mcts {
private:
  std::default_random_engine generator;
  std::unique_ptr<mcts_node> root_node;
  TicTacToe game;
  model *m;
  int nr_sim;
public:
  mcts(model *m, TicTacToe game, std::random_device::result_type r, int nr_sim=800);
  void advance(int move);
  void expand(bool training=false, bool verbose=false);
  float expected_winrate();
  float model_eval();
  float get_q();
  int get_move(float t=1.);
  std::vector<float> get_phi(float t = 1);

  void set_sims(int n) {
    nr_sim = n;
  }

  void reset() {
    root_node->clear();
  }

  void n_paths(int n) {
    root_node->get_paths(n);
  }
};
