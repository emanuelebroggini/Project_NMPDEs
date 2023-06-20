cmake_minimum_required(VERSION 3.12.0)
project(lab01 LANGUAGES CXX C)

include(common/cmake-common.cmake)

add_executable(project_nmpde src/main.cpp src/Poisson3D.cpp)
deal_ii_setup_target(project_nmpde)


