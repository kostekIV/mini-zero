

build/main: cpp/main.cpp cpp/game.cpp cpp/model.cpp
	g++ $^ -lgsl -ltensorflow -lgslcblas -std=c++17 -O3 -o $@
