# ------------------------------------------------------------------------------
# Set Up an OpenCL environment and build the Project
# ------------------------------------------------------------------------------
project(GEMM)
cmake_minimum_required(VERSION 3.1)

# -- OPTIONS -------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-O2")

# -- FIND ----------------------------------------------------------------------
#set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake") #needs some editing !
find_package(OpenCL REQUIRED)

# -- INCLUDE -------------------------------------------------------------------
include_directories(
	${OpenCL_INCLUDE_DIRS}
)

# -- TARGET --------------------------------------------------------------------
add_executable(GEMM
	src/main.cpp
)

# -- LINK ----------------------------------------------------------------------
target_link_libraries(GEMM
	${OpenCL_LIBRARY}
)
