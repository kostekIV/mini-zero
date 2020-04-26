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
#include "mct_search.hpp"


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
  mct_search ai(&m, g, 0, 2000);
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
      ai.expand();
      a = ai.get_move(0);
    }
    i += 1;
    g.move(a);
    ai.advance(a);
  }
  std::cout << g;
}

// void self_play1(model &m1, model &m2, int nr_sim=800) {

//   int play_as_x = 0;

//   std::random_device r;
//   std::default_random_engine generator(r());
//   int move_count = 0;
//   int model_move_count = 0;
//   auto g = TicTacToe(10, 10, 5);

//   mcts x[] = {mcts(&m1, nr_sim), mcts(&m2, nr_sim)};

//   while (g.get_winner() == 2) {
//     auto phi = x[(move_count + play_as_x) % 2].expand_probs(g);
//     auto state = g.state();
//     // std::discrete_distribution<int> du(phi.begin(), phi.end());
//     // auto move = du(generator);
//     auto move = argmax(phi);
//     std::cout << g;
//     move_count += 1;
//     g.move(move);
//   }
//   std::cout << g;
// }

void self_play(model &m1, model &m2, int nr_sim=800) {
  std::random_device r;
  std::default_random_engine generator(r());
  int move_count = 0;
  int model_move_count = 0;
  auto g = TicTacToe(10, 10, 5);

  mct_search x[] = {mct_search(&m1, g, r(), nr_sim), mct_search(&m2, g, r(), nr_sim)};

  int play_as_x = 0;
  while (g.get_winner() == 2) {
    auto& player = x[(move_count + play_as_x) % 2];
    player.expand();
    auto move = player.get_move(0);
    move_count += 1;

    std::cout << g;
    x[0].advance(move);
    x[1].advance(move);
    g.move(move);
  }
  std::cout << g;
}

int main(int argc, char *argv[]) {
  int model_id1 = atoi(argv[1]);
  if (argc < 3) {
    play_with_human(model_id1);
    return 0;
  }
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
  for (int i = 0; i >= 0; i--)
    self_play(m1, m2, nr_sim);
  // self_play(m1, m2, model_id1, model_id2, phis, states, vs, 1, nr_sim);
  auto e = std::chrono::steady_clock::now();
  auto d = std::chrono::duration_cast<std::chrono::milliseconds>(e - s);
  std::cout <<  d.count() / 100. << std::endl;


  // s = std::chrono::steady_clock::now();
  // for (int i = 0; i >= 0; i--)
  //   self_play1(m1, m2, nr_sim);
  // // self_play(m1, m2, model_id1, model_id2, phis, states, vs, 1, nr_sim);
  // e = std::chrono::steady_clock::now();
  // d = std::chrono::duration_cast<std::chrono::milliseconds>(e - s);
  // std::cout <<  d.count() / 100. << std::endl;
}
