#pragma once

#include <vector>
#include <iostream>

class TicTacToe {
private:
  int last_move;
  int n, m;
  int req;
  int turn;
  int winner;
  std::vector<std::vector<int>> board;
  std::vector<float> _allowed_moves;
  int check_winner();
public:
  TicTacToe(int n, int m, int req);
  std::vector<float> allowed_moves();
  std::vector<float> state() const;
  std::vector<int> int_state() const;
  int move(int k);
  int get_winner();

  friend std::ostream& operator<<(std::ostream& os, const TicTacToe& t);
  bool operator==(const TicTacToe& t) const { 
    return board == t.board;
  }
};

template <class T>
inline void hash_combine(std::size_t& seed, const T& v) {
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
}

namespace std {
template <>
struct hash<TicTacToe> {
  size_t operator()(const TicTacToe& t) const {
    auto state = t.state();
    std::size_t seed = state.size();
    for (auto &x: state) {
      hash_combine(seed, x);
    }
    return seed;
  }
};

}