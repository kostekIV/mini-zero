#include "node.hpp"

#include <algorithm>
#include <memory>
#include <numeric>
#include <cmath>


namespace {
  struct {
    bool operator()(const std::unique_ptr<mcts_node>&a, const std::unique_ptr<mcts_node>&b) const {
      return a->get_visit_count() > b->get_visit_count();
    }
  } visit_compare;
}

mcts_node::mcts_node(mcts_node* parent, int action, float pol, TicTacToe t):
    parent(parent),
    action(action),
    pol(pol),
    t(t),
    visit_count(0),
    q(0),
    w(0) {
  if (action >= 0) {
    this->t.move(action);
  }
};

node_selection mcts_node::select() {
  vl_count++;

  if (t.get_winner() != 2 or visit_count == 0) {
    return {this, t.get_winner() != 2, 1.f * abs(t.get_winner())};
  }

  float c = log((1.f + visit_count + 10.f) / 10.f) + .2f;
  float n = sqrt(visit_count);
  mcts_node *best_child;
  float best_v = INT32_MIN;

  for (auto &x: children) {
    auto r = x->node_value(c, n);
    if (r > best_v) {
      best_v = r;
      best_child = x.get();
    }
  }
  return best_child->select();
}

void mcts_node::expand_eval(std::vector<float> &pol, float v) {
  if (t.get_winner() != 2) {
    return backup(v);
  }

  if (visit_count == 0) {
    std::vector<int> idx(pol.size());
    std::iota(idx.begin(), idx.end(), 0);
    auto am = t.allowed_moves();
    // zero out illegall moves
    std::transform(idx.begin(), idx.end(), pol.begin(), [&pol, &am](int i) -> float { return pol[i] * am[i]; });

    // normalize policy
    auto pol_sum = std::accumulate(pol.begin(), pol.end(), 0.0);
    if (pol_sum  == 0) {
      std::transform(idx.begin(), idx.end(), pol.begin(), [&pol, &am](int i) -> float { return am[i]; });
      pol_sum = std::accumulate(pol.begin(), pol.end(), 0.0);
    }
    pol_sum = 1 / pol_sum;
    std::transform(pol.begin(), pol.end(), pol.begin(), [&pol_sum](float a) -> float { return a * pol_sum; });

    // for each legal action create child
    for (int i = 0; i < am.size(); i++) {
      if (am[i] == 1)
        children.emplace_back(std::make_unique<mcts_node>(this, i, pol[i], t));
    }

    // late apply dirichlet
    if (th.size() > 0) {
      apply_dirichlet(th);
    }
  }
  backup(-v);
}

void mcts_node::backup(float v) {
  vl_count--;
  visit_count++;
  w += v;
  q = w / visit_count;
  if (parent) {
    parent->backup(-v);
  }
}

inline float mcts_node::node_value(float c, float n) const {
  return q + c * n * pol / (1. + visit_count) - vl_count * vloss;
}

float mcts_node::get_visit_count() const {
  return visit_count;
}

std::vector<float> mcts_node::get_phi(int size, float t) {
  std::vector<float> phi(size, 0);
  int m = 1;
  if (t != 0) {
    m = 1 / t;
  }
  std::for_each(children.begin(), children.end(), [&phi, m](const node_pointer &np) {
    phi[np->get_action()] = pow(np->get_visit_count(), m);
  });
  float dn = 1. / std::accumulate(phi.begin(), phi.end(), 0.0);
  std::for_each(children.begin(), children.end(), [&phi, dn, t](const node_pointer &np) {
    phi[np->get_action()] *= dn;
  });
  return phi;
}

std::unique_ptr<mcts_node> mcts_node::move(int move) {
  if (visit_count == 0) {
    return std::make_unique<mcts_node>(nullptr, move, 0., t);
  }
  for (auto &child: children) {
    if (child->get_action() == move) {
      child->nullify_parent();
      return std::move(child);
    }
  }
  throw std::invalid_argument("No child with move " + move);
}

void mcts_node::apply_dirichlet_to_children(float alpha, std::default_random_engine &generator) {
  auto am = t.allowed_moves();
  int moves_count = std::accumulate(am.begin(), am.end(), 0);
  std::vector<float> thetas(moves_count);
  std::gamma_distribution<float> gamma(alpha, 1);
  std::for_each(thetas.begin(), thetas.end(), [&generator, &gamma](float &x) {
    x = gamma(generator);
  });
  auto thetas_sum = 1 / std::accumulate(thetas.begin(), thetas.end(), 0.0);
  std::transform(thetas.begin(), thetas.end(), thetas.begin(), [&thetas_sum](float a) -> float { return a * thetas_sum; });

  apply_dirichlet(thetas);
};

void mcts_node::apply_dirichlet(std::vector<float> &th) {
  if (children.size() == 0) {
    std::copy(th.begin(), th.end(), std::back_inserter(this->th));
    return;
  }

  int i = 0;
  std::for_each(children.begin(), children.end(), [&th, &i](const node_pointer &np) {
    np->apply_dirichlet(th[i++]);
  });
}

void mcts_node::apply_dirichlet(float th) {
  pol = 0.5 * pol + 0.5 * th;
}

void mcts_node::nullify_parent() {
  parent = nullptr;
}

std::vector<float> mcts_node::state() {
  return t.state();
}

float mcts_node::get_winrate() {
  return q;
}

std::vector<float> mcts_node::most_visited(int n) {
  std::vector<float> res;
  if (children.size() == 0) {
    return res;
  }
  int m = std::min(n, (int) children.size());
  partial_sort(children.begin(), children.begin() + m, children.end(), visit_compare);

  for (int i = 0; i < m; i++) {
    res.push_back(children[i]->get_action());
    res.push_back(children[i]->get_winrate());
    res.push_back(children[i]->get_visit_count());
  }
  return res;
}

void mcts_node::get_paths(int n) {
  int m = std::min(n, (int) children.size());
  partial_sort(children.begin(), children.begin() + m, children.end(), visit_compare);

  for (int i = 0; i < m; i++) {
  std::cout << "(" << children[i]->get_visit_count() << ", " << children[i]->get_action() << ")";
    children[i]->get_best_path(5);
    std::cout << "\n";
  }
}

void mcts_node::get_best_path(int deep) {
  if (deep == 0 or visit_count == 0 or t.get_winner() != 2) {
    return;
  }
  partial_sort(children.begin(), children.begin() + 1, children.end(), visit_compare);
  std::cout << ", (" << children[0]->get_visit_count() << ", " << children[0]->get_action() << ")";
  children[0]->get_best_path(deep - 1);
}