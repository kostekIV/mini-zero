#include <tensorflow/c/c_api.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <iomanip>
#include <random>
#include <string>
#include <algorithm>
#include <fstream>
#include <chrono>

#include "model.hpp"
#include "game.hpp"
#include "mcts.hpp"


void play_with_human(int model_id) {
  std::ostringstream s;
  s << "./models/mini-zero-" << model_id;
  model m(s.str().c_str(), "serve");
  mcts ai(&m, 600);

  auto g = TicTacToe(10, 10, 5);
  int i = 1;
  int a;
  while (g.get_winner() == 2) {
    // std::cout << g;
    if (i % 2 == 0) {
      auto am = g.allowed_moves();
      do {
        std::cin >> a;
      } while (am[a] == 0);
    } else {
      auto p = ai.expand_probs(g);
      float best = -1;
      for (int i = 0; i < p.size(); i++) {
        if (best < p[i]) {
          best = p[i];
          a = i;
        }
      }
    }
    i += 1;
    g.move(a);
  }
  std::cout << g;
}

void self_play(
    model &m1, model &m2, int model_id1, int model_id2, std::vector<float> &phis, std::vector<float> &states, std::vector<float> &vs, int nr_games=200 , int nr_sim=800) {

  int play_as_x = 0;

  float model_1_wins_as_x = 0;
  float model_1_wins_as_o = 0;
  float model_1_losses_as_x = 0;
  float model_1_losses_as_o = 0;

  int played_games = 0;
  int played_games_as_x = 0;
  for (int z = 0; z < nr_games; z++) {
    int move_count = 0;
    int model_move_count = 0;
    auto g = TicTacToe(10, 10, 5);

    mcts x[] = {mcts(&m1, nr_sim), mcts(&m2, nr_sim)};

    while (g.get_winner() == 2) {
      auto phi = x[(move_count + play_as_x) % 2].expand_probs(g);
      auto state = g.state();
      int move = -1;
      float hp = 0;
      for (int i = 0; i < phi.size(); i++) {
        if (hp < phi[i]) {
          move = i;
          hp = phi[i];
        }
      }
      std::cout << g;
      move_count += 1;
      g.move(move);
    }
    std::cout << g;
    // update vs
    float winner = g.get_winner();
    if (2*(1 - play_as_x) - 1 == winner) {
      winner = 1;
    } else if (winner == 0) {
      winner = 0; // no op
    } else {
      winner = -1;
    }

    // collect statistics to display
    played_games += 1;
    if (play_as_x == 0) {
      played_games_as_x += 1;
    }
    if (winner == 1) {
      if (play_as_x == 0) {
        model_1_wins_as_x += 1;
      } else {
        model_1_wins_as_o += 1;
      }
    } else if (winner == -1) {
      if (play_as_x == 0) {
        model_1_losses_as_x += 1;
      } else {
        model_1_losses_as_o += 1;
      }
    }

    play_as_x = (play_as_x + 1) % 2;
  }
  auto played_games_as_y = played_games - played_games_as_x;
  std::cout << "\n";
  std::cout << model_id1 << " vs " << model_id2 << "\n";
  std::cout << "Winrate as x: " << model_1_wins_as_x / played_games_as_x << "\n";
  std::cout << "Winrate as o: " << model_1_wins_as_o / played_games_as_y << "\n";
}

int main(int argc, char *argv[]) {
  int model_id1 = atoi(argv[1]);
  int model_id2 = atoi(argv[2]);
  int nr_sim = 800;
  if (argc >= 4) {
    nr_sim = atoi(argv[3]);
  }

  std::ostringstream s1;
  s1 << "./models/mini-zero-" << model_id1;
  model m1(s1.str().c_str(), "serve");

  std::ostringstream s2;
  s2 << "./models/mini-zero-" << model_id2;
  model m2(s2.str().c_str(), "serve");

  std::vector<float> phis;
  std::vector<float> states;
  std::vector<float> vs;
  std::cout << "nr of sims " << nr_sim << "\n";
  auto s = std::chrono::steady_clock::now();
  self_play(m1, m2, model_id1, model_id2, phis, states, vs, 1, nr_sim);
  auto e = std::chrono::steady_clock::now();
  auto d = std::chrono::duration_cast<std::chrono::milliseconds>(e - s);
  std::cout <<  d.count() / 100. << std::endl;
}
