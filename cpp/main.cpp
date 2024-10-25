#include <tensorflow/c/c_api.h>

#include <assert.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <iomanip>
#include <random>
#include <string>
#include <algorithm>
#include <fstream>

#include "model.hpp"
#include "game.hpp"
#include "mcts.hpp"


void self_play(
    model &m1, model &m2, int model_id1, int model_id2, std::vector<float> &phis, std::vector<float> &states, std::vector<float> &vs, int nr_games=200) {

  std::random_device r;
  std::default_random_engine generator(r());
  std::uniform_real_distribution<float> uniform(0.0, 1.0);

  int play_as_x = 0;

  float wins_as_x = 0;
  float wins_as_o = 0;

  int played_games = 0;
  int played_games_as_x = 0;
  for (int z = 0; z < nr_games; z++) {
    int move_count = 0;
    int model_move_count = 0;
    auto g = TicTacToe(10, 10, 5);

    mcts x[] = {mcts(&m1, g, r(), 420), mcts(&m2, g, r(), 420)};

    float t = 1;
    while (g.get_winner() == 2) {
      if (move_count > 15) {
        t = 0;
      }
      if (z == nr_games - 1) {
        std::cout << g;
      }

      auto &player = x[(move_count + play_as_x) % 2];
      player.expand(true);
      auto phi = player.get_phi(t);
      auto state = g.state();

      model_move_count += 1;
      std::copy(phi.begin(), phi.end(), std::back_inserter(phis));
      std::copy(state.begin(), state.end(), std::back_inserter(states));
      vs.push_back(player.get_q());

      auto move = player.get_move(t);
      x[0].advance(move);
      x[1].advance(move);
      g.move(move);
      move_count += 1;
    }
    if (z == nr_games - 1) {
      std::cout << g;
    }
    // update vs
    float winner = g.get_winner();
    // if (2*(1 - play_as_x) - 1 == winner) {
    //   winner = 1;
    // } else if (winner == 0) {
    //   winner = 0; // no op
    // } else {
    //   winner = -1;
    // }

    // collect statistics to display
    played_games += 1;
    if (play_as_x == 0) {
      played_games_as_x += 1;
    }
    if (winner == 1) {
      wins_as_x += 1;
    } else if (winner == -1) {
      wins_as_o += 1;
    }
    // if (play_as_x != 0) {
    //   winner *= -1;
    // }

    int start = vs.size() - model_move_count;
    for(int i = 0; i < model_move_count; i++) {
      float z = vs[start + i];
      vs[start + i] = (z + winner) / 2.f;
      // vs.push_back(winner);
      winner *= -1;
    }

    assert(winner == -1 or winner == 0);

    play_as_x = (play_as_x + 1) % 2;
    std::cout << "\r\033[K" << played_games << "/" << nr_games << std::flush;
  }
  int played_games_as_y = played_games - played_games_as_x;
  std::cout << "\n";
  std::cout << "Games played: " << played_games << ", model under training " << model_id1 << "\n";
  std::cout << model_id1 << " vs " << model_id2 << "\n";
  std::cout << "Winrate as x: " << wins_as_x / nr_games << "\n";
  std::cout << "Winrate as o: " << wins_as_o / nr_games << "\n";
}

void helper(std::string path, std::vector<float>& p);
int main(int argc, char *argv[]) {
  int model_id1 = atoi(argv[1]);

  std::ostringstream s1;
  s1 << "./models/mini-zero-" << model_id1;
  model m1(s1.str().c_str(), "serve");

  std::vector<float> phis;
  std::vector<float> states;
  std::vector<float> vs;
  self_play(m1, m1, model_id1, model_id1, phis, states, vs, 300);


  std::ostringstream sp;
  sp << "./data/" << model_id1 << "/";
  std::string save_path = sp.str();

  helper(save_path + "phis.npy", phis);
  helper(save_path + "vals.npy", vs);
  helper(save_path + "states.npy", states);
}

void helper(std::string path, std::vector<float>& p) {
  std::ofstream ofs(path);
  for (int i = 0; i < p.size(); i++) {
    ofs << p[i];
    if (i != p.size() - 1)
      ofs << ",";
  }
  ofs.close();
}
