#pragma once
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>
#include <atomic>

#include "model.hpp"
#include "game.hpp"


class node_selection;

class mcts_node {
private:
  const double vloss = 4.f;
  int vl_count = 0;
  int size;
  int action;
  double visit_count;
  double pol;
  double w;
  double q;
  TicTacToe t;
  mcts_node *parent;
  std::unordered_map<int, std::unique_ptr<mcts_node>> children;
  std::vector<double> th;
  using node_pair = decltype(children)::value_type;
public:
  mcts_node(mcts_node *parent, int action, double pol, TicTacToe t);
  node_selection select();
  void expand_eval(std::vector<float> &pol, double val);
  double node_value(double c, double n) const;
  std::vector<double> get_phi(int size);
  double get_visit_count();
  std::unique_ptr<mcts_node> move(int move);
  void apply_dirichlet(std::vector<double> &th);
  void apply_dirichlet(double th);
  void apply_dirichlet_to_children(double alpha, std::default_random_engine &generator);
  void nullify_parent();
  void backup(double val);
  std::vector<float> state();
};

class node_selection {
public:
  node_selection() = default;
  node_selection(mcts_node *node, bool final_position, double v):
    result(node),
    final_position(final_position),
    v(v) {};
  mcts_node* result;
  bool final_position;
  double v;
};
