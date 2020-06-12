#include "mcts.hpp"

#include <omp.h>
#include <thread>
#include <memory>
#include <vector>
#include <algorithm>

namespace {
template <typename T>
int argmax(std::vector<T> v) {
  std::vector<int> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  return *std::max_element(idx.begin(), idx.end(), [&v](int i, int j) {
    return v[i] < v[j];
  });
}
}

mcts::mcts(model *m, TicTacToe game, std::random_device::result_type r, int nr_sim):
  m(m),
  game(game),
  root_node(std::make_unique<mcts_node>(nullptr, -1, -1, game)),
  nr_sim(nr_sim),
  generator(r) {
}

void mcts::advance(int move) {
  game.move(move);
  root_node = root_node->move(move);
}

// void mcts::expand(bool training) {
//   if (training) {
//     root_node->apply_dirichlet_to_children(0.7, generator);
//   }
//   for (int i = 0; i < nr_sim; i += 1) {
//     auto selected_node = root_node->select();
//     if (not selected_node.final_position) {
//       auto val_pol = m->predict(selected_node.result->state().data())[0];
//       selected_node.result->expand_eval(val_pol.second, val_pol.first);
//     } else {
//       std::vector<float> x;
//       selected_node.result->expand_eval(x, selected_node.v);
//     }
//   }
// }

void mcts::expand(bool training, bool verbose) {
  if (training) {
    root_node->apply_dirichlet_to_children(0.9, generator);
  }
  std::vector<float> pol;
  for (int i = 0; i < nr_sim; i += 4) {

    // store not final selected nodes;
    std::vector<node_selection> not_final;
    for (int j = 0; j < 4; j++) {
      auto sn = root_node->select();
      if (sn.final_position) {
        // expand final positions without waiting.
        sn.result->expand_eval(pol, sn.v);
      } else {
        not_final.push_back(sn);
      }
    }

    // batch couple of positions to model and expand them.
    if (not_final.size() > 0) {
      int st_size = root_node->state().size();
      std::vector<float> n_states;
      n_states.reserve(not_final.size() * st_size);

      for (auto &x: not_final) {
        auto state = x.result->state();
        n_states.insert(n_states.end(), state.begin(), state.end());
      }

      auto n_preds = m->predict(n_states.data(), not_final.size());
      for (int k = 0; k < n_preds.size(); k++) {
        auto &val_pol = n_preds[k];
        not_final[k].result->expand_eval(val_pol.second, val_pol.first);
      }
    }
    if (verbose) {
      auto res = root_node->most_visited(6);
      std::cout << "\r\033[K" << "nr sims: " << i + 4;
      for (int j = 0; j < res.size(); j += 3) {
        std::cout << ", (" << (int) res[j] << ", " << res[j + 1] << ", " << (int) res[j + 2] << ")";
      }
      std::cout << std::flush;
    }
  }

  if (verbose) {
    std::cout << std::endl;
  }
}

float mcts::expected_winrate() {
  return 1 - (1 + root_node->get_winrate()) / 2;
}

float mcts::get_q() {
  return -root_node->get_winrate();
}

float mcts::model_eval() {
  return m->predict(game.state().data())[0].first;
}

int mcts::get_move(float t) {
  if (root_node->get_visit_count() == 0) {
    return -1;
  }
  auto phi = get_phi(t);
  if (t == 0) {
    auto move = argmax(phi);
    return move;
  }
  std::discrete_distribution<int> du(phi.begin(), phi.end());
  auto move = du(generator);
  return move;
}

std::vector<float> mcts::get_phi(float t) {
  auto phi = root_node->get_phi(game.allowed_moves().size(), t);
  if (t == 0) {
    int max_i = argmax(phi);
    std::vector<float> _phi(phi.size(), 0);
    _phi[max_i] = 1;
    return _phi;
  }
  return phi;
}
