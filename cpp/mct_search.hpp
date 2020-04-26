#pragma once

#include <memory>
#include <vector>
#include <random>

#include "node.hpp"
#include "game.hpp"


class mct_search {
private:
  std::default_random_engine generator;
  std::unique_ptr<mcts_node> root_node;
  TicTacToe game;
  model *m;
  int nr_sim;
public:
  mct_search(model *m, TicTacToe game, std::random_device::result_type r, int nr_sim=800);
  void advance(int move);
  void expand(bool training=false);
  int get_move(double t=1.);
  std::vector<double> get_phi();
};
