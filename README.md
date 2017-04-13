# Blazing GEMM: High Performance Matrix Multiplication

Exploring General Matrix Multplication (GEMM) on GPU. 
The project consists of a C++ benchmark, which process several standard and custom techniques, implemented in OpenCL-1.2.

<p align="center">
  <img src="https://github.com/Cryst4L/Blazing-GEMM/blob/master/results.png"/>
</p>

## How to run the benchmark?

#### A. Using the command line tool

First, you want to build the project with CMake:

```sh
mkdir build && cd build
cmake ..
make
```

Then you can run the tool. Notice that it comes with several options:

* **-s** : specify the size of matrices processed (default is 512).
* **-i** : specify the number of iterations, which is used to refine the measurements (default is 20).
* **-r** : run the tool in **reduced mode**, i.e. with minimal verbose and no error checking.

A typical usage would be:

```sh
./GEMM -s 1024 -i 10
```

#### B. Using the Python script

Alternatively, you can use the _run.py_ script, which builds the project for you, run the benchmark on several configurations and plot the results.
Plotting is done with **matplotlib**, so you want to make sure it is installed on your machine before running the script:

```sh
python run.py
```

## Principal Files

The whole project is organized around few core sources:

* **kernels.cl** : contains all the OpenCL benchmarked kernels, along with some documentation.

* **main.cpp** : implements the benchmarking pipeline. This source is crystal clear, so it might be worth checking.

* **benchmark.h** : contains the benchmark class, written in 'header-only' style.

## Copyright

This project is released under MIT license.
