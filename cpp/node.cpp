#include "node.hpp"

#include <algorithm>
#include <memory>
#include <numeric>
#include <cmath>


mcts_node::mcts_node(mcts_node* parent, int action, double pol, TicTacToe t):
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
    return {this, t.get_winner() != 2, 1. * abs(t.get_winner())};
  }

  double c = 2.2;
  auto n = sqrt(visit_count);
  mcts_node *best_child;
  double best_v = INT32_MIN;
  for (auto &x: children) {
    auto r = x.second->node_value(c, n);
    if (r > best_v) {
      best_v = r;
      best_child = x.second.get();
    }
  }
  return best_child->select();
}

void mcts_node::expand_eval(std::vector<float> &pol, double v) {
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
        children.try_emplace(i, std::make_unique<mcts_node>(this, i, pol[i], t));
    }

    // late apply dirichlet
    if (th.size() > 0) {
      apply_dirichlet(th);
    }
  }

  backup(v);
}

void mcts_node::backup(double v) {
  vl_count--;
  visit_count++;
  w += v;
  q = w / visit_count;
  if (parent) {
    parent->backup(-v);
  }
}

double mcts_node::node_value(double c, double n) const {
  return q + c * n * pol / (1. + visit_count) - vl_count * vloss;
}

double mcts_node::get_visit_count() {
  return visit_count;
}

std::vector<double> mcts_node::get_phi(int size) {
  std::vector<double> phi(size, 0);
  auto dn = 1. / (visit_count - 1);
  std::for_each(children.begin(), children.end(), [&phi, dn](const node_pair &np) {
    phi[np.first] = np.second->get_visit_count() * dn;
  });

  return phi;
}

std::unique_ptr<mcts_node> mcts_node::move(int move) {
  if (visit_count == 0) {
    return std::make_unique<mcts_node>(nullptr, move, 0., t);
  }
  children[move]->nullify_parent();
  return std::move(children[move]);
}

void mcts_node::apply_dirichlet_to_children(double alpha, std::default_random_engine &generator) {
  auto am = t.allowed_moves();
  int moves_count = std::accumulate(am.begin(), am.end(), 0);
  std::vector<double> thetas(moves_count);
  std::gamma_distribution<double> gamma(alpha, 1);
  std::for_each(thetas.begin(), thetas.end(), [&generator, &gamma](double &x) {
    x = gamma(generator);
  });
  auto thetas_sum = 1 / std::accumulate(thetas.begin(), thetas.end(), 0.0);
  std::transform(thetas.begin(), thetas.end(), thetas.begin(), [&thetas_sum](float a) -> float { return a * thetas_sum; });

  apply_dirichlet(thetas);
};

void mcts_node::apply_dirichlet(std::vector<double> &th) {
  if (children.size() == 0) {
    std::copy(th.begin(), th.end(), std::back_inserter(this->th));
    return;
  }

  int i = 0;
  std::for_each(children.begin(), children.end(), [&th, &i](const node_pair &np) {
    np.second->apply_dirichlet(th[i++]);
  });
}

void mcts_node::apply_dirichlet(double th) {
  pol = 0.6 * pol + 0.4 * th;
}

void mcts_node::nullify_parent() {
  parent = nullptr;
}


std::vector<float> mcts_node::state() {
  return t.state();
}
