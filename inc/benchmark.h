#pragma once

#include <CL/cl.hpp>

#include <fstream>
#include <sstream>
#include <vector>

#include "matrix.h"

class Benchmark
{
  public :

	Benchmark();
	~Benchmark();

	void grabDevice();
	void printDeviceInfo();

	void loadProgram(const char * file_name);
	void selectKernel(std::string kernel_name);

	void setGridSize(int grid_size);
	void setBlockSize(int block_size);

	void performKernel(Matrix &A, Matrix &B, Matrix &C);

	float getProcessingTime();

  private :

	float m_processing_time;

	std::string m_sources;
	cl::Program m_program;

	cl::Device m_device;
	cl::Context m_context;
	cl::CommandQueue m_queue;
	cl::Kernel m_kernel;

	cl::NDRange m_local_range;
	cl::NDRange m_global_range;
};
