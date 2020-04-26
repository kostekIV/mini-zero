#include "game.hpp"

#include <stdexcept>
#include <sstream>
#include <numeric>

TicTacToe::TicTacToe(int n, int m, int req):
    n(n), m(m), req(req), turn(1), board(n, std::vector<int>(m, 0)), _allowed_moves(m*n, 1){
  winner = 2;
}

std::vector<float> TicTacToe::allowed_moves() {
  return _allowed_moves;
}

std::vector<float> TicTacToe::state() const {
  std::vector<float> res(n*m*3, 0.);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j< m; j++) {
      for (int t = 0; t < 3; t++) {
        int x = board[i][j];
        if ((t == 0 and x == 1) or (t == 1 and x == -1)) {
          res[3 *(i * m + j) + t] = 1;
        } else if (t == 2) {
          res[3 * (i * m + j) + t] = (1 - turn) / 2;
        }
      }
    }
  }
  return res;
}

int TicTacToe::move(int k) {
  int i = k / m;
  int j = k % m;
  if (board[i][j] != 0) {
    std::ostringstream oss;
    oss << "move [" << i << "][" << j << "] taken by " << turn << "\n" << *this;
    throw std::invalid_argument(oss.str());
  }

  board[i][j] = turn;
  _allowed_moves[k] = 0;
  turn *= -1;
  last_move = k;

  return check_winner();
}


int TicTacToe::check_winner() {
  int i = last_move / m;
  int j = last_move % m;

  int player = turn * -1;
  int in_row = 0;

  int end = j;
  while (end < m and board[i][end] == player) {
    end += 1;
  }
  int start = j;
  while (start >= 0 and board[i][start] == player) {
      start -= 1;
  }
  in_row = std::max(end - start - 1, in_row);

  end = i;
  while (end < n and board[end][j] == player) {
    end += 1;
  }
  start = i;
  while (start >= 0 and board[start][j] == player) {
      start -= 1;
  }
  in_row = std::max(end - start - 1, in_row);

  int end_i = i;
  int end_j = j;
  while (end_i < n and end_j < m and board[end_i][end_j] == player) {
    end_i += 1;
    end_j += 1;
  }

  int start_i = i;
  int start_j = j;
  while (start_i >= 0 and start_j >= 0 and board[start_i][start_j] == player) {
    start_i -= 1;
    start_j -= 1;
  }
  in_row = std::max(end_i - start_i - 1, in_row);

  end_i = i;
  end_j = j;
  while (end_i < n and end_j >= 0 and board[end_i][end_j] == player) {
    end_i += 1;
    end_j -= 1;
  }

  start_i = i;
  start_j = j;
  while (start_i >= 0 and start_j < m and board[start_i][start_j] == player) {
    start_i -= 1;
    start_j += 1;
  }
  in_row = std::max(end_i - start_i - 1, in_row);

  if (in_row >= req) {
    winner = player;
    return player;
  }

  if (std::accumulate(_allowed_moves.begin(), _allowed_moves.end(), 0) == 0) {
    winner = 0;
    return winner;
  }

  winner = 2;
  return 2;
}

int TicTacToe::get_winner() {
  return winner;
}

std::ostream& operator<<(std::ostream& os, const TicTacToe& t) {
  os << std::string(t.m * 7 + t.m, '-') << std::endl;
  std::string emp = "|";
  for (int i = 0; i < t.m; i++) emp += "       |";
  for (int i = 0; i < t.n; i ++) {
    os << emp << std::endl;
    os << "|";
    for (int j = 0; j < t.m; j++) {
      os << "   ";
      if (t.board[i][j] == 1) {
        os << "\033[92mX\033[0m";
      } else if (t.board[i][j] == -1) {
        os << "\033[94mO\033[0m";
      } else {
        os << " ";
      }
      os << "   |";
    }

    os << std::endl << emp << std::endl;
    os << std::string(t.m * 7 + t.m, '-') << std::endl;
  }
  return os;
}
