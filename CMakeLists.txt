cmake_minimum_required(VERSION 3.10)

project(mini-zero)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
add_compile_options(-O3)

add_executable(self_play cpp/main.cpp)
add_executable(eval cpp/evaluate.cpp)
add_executable(analise cpp/analise.cpp)

target_include_directories(self_play PRIVATE cpp/)
target_link_libraries(self_play tensorflow)
target_include_directories(eval PRIVATE cpp/)
target_link_libraries(eval tensorflow)
target_include_directories(analise PRIVATE cpp/)
target_link_libraries(analise tensorflow)

target_sources(self_play PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/cpp/game.cpp
    ${CMAKE_CURRENT_LIST_DIR}/cpp/mcts.cpp
    ${CMAKE_CURRENT_LIST_DIR}/cpp/node.cpp
    ${CMAKE_CURRENT_LIST_DIR}/cpp/model.cpp
)
target_sources(analise PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/cpp/game.cpp
    ${CMAKE_CURRENT_LIST_DIR}/cpp/mcts.cpp
    ${CMAKE_CURRENT_LIST_DIR}/cpp/node.cpp
    ${CMAKE_CURRENT_LIST_DIR}/cpp/model.cpp
)
target_sources(eval PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/cpp/game.cpp
    ${CMAKE_CURRENT_LIST_DIR}/cpp/mcts.cpp
    ${CMAKE_CURRENT_LIST_DIR}/cpp/node.cpp
    ${CMAKE_CURRENT_LIST_DIR}/cpp/model.cpp
)
