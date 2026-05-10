cmake_minimum_required(VERSION 3.14)
project(PortfolioEngine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Core Math Engine Library
add_library(engine SHARED src/engine.cpp)
target_include_directories(engine PUBLIC include)
