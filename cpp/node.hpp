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
  const float vloss = 4.f;
  int vl_count = 0;
  int size;
  int action;
  float visit_count;
  float pol;
  float w;
  float q;
  TicTacToe t;
  mcts_node *parent;
  std::vector<std::unique_ptr<mcts_node>> children;
  std::vector<float> th;
  using node_pointer = decltype(children)::value_type;
public:
  mcts_node(mcts_node *parent, int action, float pol, TicTacToe t);
  node_selection select();
  void expand_eval(std::vector<float> &pol, float val);
  float node_value(float c, float n) const;
  std::vector<float> get_phi(int size, float t);
  float get_visit_count() const;
  std::unique_ptr<mcts_node> move(int move);
  void apply_dirichlet(std::vector<float> &th);
  void apply_dirichlet(float th);
  void apply_dirichlet_to_children(float alpha, std::default_random_engine &generator);
  void nullify_parent();
  void backup(float val);
  float get_winrate();
  std::vector<float> state();
  std::vector<float> most_visited(int n);

  int get_action() const {
    return action;
  }

  void clear() {
    children.clear();
    visit_count = 0;
    w = 0;
    q = 0;
    vl_count = 0;
  }

  void get_paths(int n);
  void get_best_path(int deep);
};

class node_selection {
public:
  node_selection() = default;
  node_selection(mcts_node *node, bool final_position, float v):
    result(node),
    final_position(final_position),
    v(v) {};
  mcts_node* result;
  bool final_position;
  float v;
};
