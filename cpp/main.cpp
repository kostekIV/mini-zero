#include <tensorflow/c/c_api.h>
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
#include "mct_search.hpp"



void self_play(
    model &m1, model &m2, int model_id1, int model_id2, std::vector<double> &phis, std::vector<double> &states, std::vector<double> &vs, int nr_games=200) {

  std::random_device r;
  std::default_random_engine generator(r());
  std::uniform_real_distribution<float> uniform(0.0, 1.0);

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

    mct_search x[] = {mct_search(&m1, g, r(), 600), mct_search(&m2, g, r(), 600)};


    while (g.get_winner() == 2) {
      if (z == nr_games - 1) {
        std::cout << g;
      }

      auto &player = x[(move_count + play_as_x) % 2];
      player.expand(true);
      auto phi = player.get_phi();
      auto state = g.state();

      model_move_count += 1;
      std::copy(phi.begin(), phi.end(), std::back_inserter(phis));
      std::copy(state.begin(), state.end(), std::back_inserter(states));
      auto move = player.get_move();
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

    if (play_as_x != 0) {
      winner *= -1;
    }
    for(int i = 0; i < model_move_count; i++) {
      vs.push_back(winner);
      winner *= - 1;
    }

    play_as_x = (play_as_x + 1) % 2;
    std::cout << "\r\033[K" << played_games << "/" << nr_games << std::flush;
  }
  auto played_games_as_y = played_games - played_games_as_x;
  std::cout << "\n";
  std::cout << "Games played: " << played_games << ", model under training " << model_id1 << "\n";
  std::cout << model_id1 << " vs " << model_id2 << "\n";
  std::cout << "Winrate as x: " << model_1_wins_as_x / played_games_as_x << "\n";
  std::cout << "Winrate as o: " << model_1_wins_as_o / played_games_as_y << "\n";
  std::cout << "Lose rate as x: " << model_1_losses_as_x / played_games_as_x << "\n";
  std::cout << "Lose rate as o: " << model_1_losses_as_o / played_games_as_y << "\n";
}

void helper(std::string path, std::vector<double>& p) ;
int main(int argc, char *argv[]) {
  int model_id1 = atoi(argv[1]);

  std::ostringstream s1;
  s1 << "./models/mini-zero-" << model_id1;
  model m1(s1.str().c_str(), "serve");

  std::vector<double> phis;
  std::vector<double> states;
  std::vector<double> vs;
  self_play(m1, m1, model_id1, model_id1, phis, states, vs, 750);


  std::ostringstream sp;
  sp << "./data/" << model_id1 << "/";
  std::string save_path = sp.str();

  helper(save_path + "phis.npy", phis);
  helper(save_path + "vals.npy", vs);
  helper(save_path + "states.npy", states);
}

void helper(std::string path, std::vector<double>& p) {
  std::ofstream ofs(path);
  for (int i = 0; i < p.size(); i++) {
    ofs << p[i];
    if (i != p.size() - 1)
      ofs << ",";
  }
  ofs.close();
}
