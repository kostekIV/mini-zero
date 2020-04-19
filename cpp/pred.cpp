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

int main(int argc, char *argv[]) {
  int model_id1 = atoi(argv[1]);

  std::ostringstream s1;
  s1 << "./models/mini-zero-" << model_id1;
  model m1(s1.str().c_str(), "serve");
  int n;
  std::cin >> n;
  std::vector<float> in(n);
  for (int i = 0; i< n; i++) {
    std::cin >> in[i];
  }
  auto x = m1.predict(in.data());

  for (auto z: x.second) {
    std::cout << z << ",";
  }
}
