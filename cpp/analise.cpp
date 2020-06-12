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


template <typename T>
int argmax(std::vector<T> v) {
  std::vector<int> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  return *std::max_element(idx.begin(), idx.end(), [&v](int i, int j) {
    return v[i] < v[j];
  });
}

void play_with_human(int model_id) {
  std::ostringstream s;
  s << "./models/mini-zero-" << model_id;
  model m(s.str().c_str(), "serve");

  auto g = TicTacToe(10, 10, 5);
  mcts ai(&m, g, 0, 4200);
  int i = 1;
  int a;
  while (g.get_winner() == 2) {
    std::cout << g;
    if (i % 2 == 0) {
      auto am = g.allowed_moves();
      do {
        std::cin >> a;
      } while (am[a] == 0);
    } else {
      ai.expand(false, true);
      a = ai.get_move(0);
    }
    i += 1;
    g.move(a);
    ai.advance(a);
  }
  std::cout << g;
}

void self_play(model &m1, model &m2, int nr_sim=800, int nr_random=0) {
  std::random_device r;
  std::default_random_engine generator(r());
  int move_count = 0;
  int model_move_count = 0;
  auto g = TicTacToe(10, 10, 5);

  mcts x[] = {mcts(&m1, g, r(), nr_sim), mcts(&m2, g, r(), nr_sim)};

  int play_as_x = 0;
  std::string players ="xo";
  float t = 1;
  std::cout << g;
  while (g.get_winner() == 2) {
    if (move_count >= nr_random) {
      t = 0;
    }
    auto& player = x[(move_count + play_as_x) % 2];
    player.expand(false, true);
    auto move = player.get_move(t);
    move_count += 1;
    auto wr = player.expected_winrate();
    auto me = x[0].model_eval();

    std::cout << "Player \"" <<  players[(move_count - 1) % 2]<< "\" expects winrate: " << wr << ", model eval: " <<  me << "\n";
    x[0].advance(move);
    x[1].advance(move);
    g.move(move);
    std::cout << g;
  }
}

int main(int argc, char *argv[]) {
  int model_id1 = atoi(argv[1]);

  std::random_device r;
  std::default_random_engine generator(r());

  std::ostringstream s1;
  s1 << "./models/mini-zero-" << model_id1;
  model m1(s1.str().c_str(), "serve");

  auto g = TicTacToe(10, 10, 5);
  mcts ai(&m1, g, 0, 1200);

  std::string command;
  int value;
  while (true) {
    std::cin >> command;
    if (command == "sim") {
      ai.expand(false, true);
    } else if (command == "set_sims") {
      std::cin >> value;
      ai.set_sims(value);
    } else if (command == "exit") {
      break;
    } else if (command == "move") {
      std::cin >> value;
      g.move(value);
      ai.advance(value);
    } else if (command == "print") {
      std::cout << g;
    } else if (command == "ai_move") {
      std::cout << "best move" << ai.get_move();
    } else if (command == "paths") {
      std::cin >> value;
      ai.n_paths(value);
    } else {
      std::cout << "possible commands ...";
    } 
  }
}
