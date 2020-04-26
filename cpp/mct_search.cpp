#include "mct_search.hpp"

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

mct_search::mct_search(model *m, TicTacToe game, std::random_device::result_type r, int nr_sim):
  m(m),
  game(game),
  root_node(std::make_unique<mcts_node>(nullptr, -1, -1, game)),
  nr_sim(nr_sim),
  generator(r) {
}

void mct_search::advance(int move) {
  game.move(move);
  root_node = root_node->move(move);
}

// void mct_search::expand(bool training) {
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

void mct_search::expand(bool training) {
  if (training) {
    root_node->apply_dirichlet_to_children(0.7, generator);
  }
  std::vector<float> pol;
  for (int i = 0; i < nr_sim; i += 8) {
    std::vector<node_selection> not_final;
    for (int j = 0; j < 8; j++) {
      auto sn = root_node->select();
      if (sn.final_position) {
        sn.result->expand_eval(pol, sn.v);
      } else {
        not_final.push_back(sn);
      }
    }

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
  }
}

int mct_search::get_move(double t) {
  auto phi = get_phi();
  if (t == 0) {
    auto move = argmax(phi);
    return move;
  }
  std::discrete_distribution<int> du(phi.begin(), phi.end());
  auto move = du(generator);
  return move;
}

std::vector<double> mct_search::get_phi() {
  return root_node->get_phi(game.allowed_moves().size());
}
