#pragma once

#include <CL/cl.hpp>

#include "timer.h"
#include "matrix.h"

#include <fstream>
#include <sstream>
#include <vector>

class Benchmark
{
  public :

	Benchmark()
	:	m_processing_time(0), m_program(""), m_local_range(cl::NullRange)
	{
		grabDevice();
		m_context = cl::Context({m_device});
		m_queue = cl::CommandQueue(m_context, m_device, CL_QUEUE_PROFILING_ENABLE);
	}

	~Benchmark() {}

	void grabDevice()
	{
		std::vector <cl::Platform> all_platforms;
		cl::Platform::get(&all_platforms);
		if (all_platforms.size() == 0) {
		    std::cerr << " No platforms found.";
			std::cerr << " Please ensure that OpenCL is installed!\n";
		    exit(1);
		}

		std::vector <cl::Device> all_gpus;
		for (std::size_t p = 0; p < all_platforms.size(); p++) {
			std::vector <cl::Device> devices;
			all_platforms[p].getDevices(CL_DEVICE_TYPE_GPU, &devices);
			all_gpus.insert(all_gpus.end(), devices.begin(), devices.end());
		}

		if (all_gpus.size() != 0) {
			m_device = all_gpus[0];
		} else {
		    cl_int err;
			m_device = cl::Device::getDefault(&err);
			if (err != CL_SUCCESS) {
		    	std::cerr << " No GPU found ... \n";
			} else {
		    	std::cerr << " No OpenCL compatible devices found!\n";
				std::cerr << " Please check the OpenCL installation ...\n";
				exit(1);
			}
		}
	}

	void loadProgram(const char * file_name)
	{
		// Open the program file
		std::ifstream file_stream(file_name);
		if (!file_stream.is_open()) {
			std::cerr << " Failed to open the kernel file !\n";
			exit(1);
		}

		// Convert the file into a string stream
		std::stringstream source_stream;
		source_stream << file_stream.rdbuf();

		// Push the content in the source string
		m_sources += source_stream.str();
	  	file_stream.close();

		// Build the program from the updated sources
		cl_int err;
		m_program = cl::Program(m_context, m_sources);
		if (m_program.build({m_device}) != CL_SUCCESS) {
		    std::cerr << " Error building the CL sources :\n" << \
			m_program.getBuildInfo <CL_PROGRAM_BUILD_LOG> (m_device) << "\n";
		    exit(1);
		}
	}

	void selectKernel(std::string kernel_name)
	{
		// Extract a specified kernel from the program
		cl_int err;
    	m_kernel = cl::Kernel(m_program, kernel_name.c_str(), &err);
		if (err != CL_SUCCESS) {
	    	std::cerr << " Failed to extract the kernel !\n";
			exit(1);
		}
	}

    void setGridSize(int grid_size)
    {
        m_global_range = cl::NDRange(grid_size, grid_size);
    }

	void setBlockSize(int block_size)
	{
		m_local_range = cl::NDRange(block_size, block_size);
	}

	void performKernel(Matrix &A, Matrix &B, Matrix &C)
	{
		int A_bytes = A.rows() * A.cols() * sizeof(float);
		int B_bytes = B.rows() * B.cols() * sizeof(float);
		int C_bytes = C.rows() * C.cols() * sizeof(float);

		// Allocate the OCL memory buffers int the device
		cl::Buffer A_buffer = cl::Buffer(m_context, CL_MEM_READ_WRITE, A_bytes);
		cl::Buffer B_buffer = cl::Buffer(m_context, CL_MEM_READ_WRITE, B_bytes);
		cl::Buffer C_buffer = cl::Buffer(m_context, CL_MEM_READ_WRITE, C_bytes);

		// Push the input matrices data into the corressponding buffers
		m_queue.enqueueWriteBuffer(A_buffer, CL_TRUE, 0, A_bytes, A.data());
		m_queue.enqueueWriteBuffer(B_buffer, CL_TRUE, 0, B_bytes, B.data());

		// Set up the kernel arguments
		m_kernel.setArg(0, A.cols());
		m_kernel.setArg(1, B.cols());
		m_kernel.setArg(2, A_buffer);
		m_kernel.setArg(3, B_buffer);
		m_kernel.setArg(4, C_buffer);

		// Spawn the Kernels according to the NDRange mapping
        cl::Event event;
		m_queue.enqueueNDRangeKernel(
			m_kernel, cl::NullRange, m_global_range, m_local_range,
            NULL, &event
		);
        event.wait();

        // Fetch the elapsed_time
        unsigned long elapsed_time =
            event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
            event.getProfilingInfo<CL_PROFILING_COMMAND_START>();

		m_queue.finish();

		// Write back the output into the device memory
		m_queue.enqueueReadBuffer(C_buffer, CL_TRUE, 0, C_bytes, C.data());

		// Save the processing time (ms)
        m_processing_time = elapsed_time / 1E6;
	}

	void printDeviceInfo()
	{
        std::cout << " ----------------------------------------------\n";

        // Name of the selected device
        std::cout << " Device used           : ";
		std::cout << m_device.getInfo <CL_DEVICE_NAME> () << "\n";

		// Number of Compute Cores on the device.
		cl_uint compute_units =
			m_device.getInfo <CL_DEVICE_MAX_COMPUTE_UNITS> ();
		std::cout << " Compute Units         : ";
		std::cout << compute_units << "\n";

		// Maximum number of work-items that can be specified
		// in each dimension of the work-group in the NDRange.
		std::vector <size_t> item_sizes =
			m_device.getInfo <CL_DEVICE_MAX_WORK_ITEM_SIZES> ();
		std::cout << " Max Work Group Ranges : [";
		std::cout << item_sizes[0] << ":";
		std::cout << item_sizes[1] << ":";
		std::cout << item_sizes[2] << "]\n";

		// Maximum number of work-items in a work-group
		// processing a kernel using the data parallel execution model.
		size_t workgroup_size =
			m_device.getInfo <CL_DEVICE_MAX_WORK_GROUP_SIZE> ();
		std::cout << " Max Work Group Size   : ";
		std::cout << workgroup_size << "\n";

		// Available memory in the global cache of the device
		cl_ulong global_cache =
			m_device.getInfo <CL_DEVICE_GLOBAL_MEM_SIZE> ();
		std::cout << " Global Cache Size     : ";
		std::cout << global_cache / 1e9 << " GB\n";

		// Available memory in the local caches of the device
		cl_ulong local_cache =
			m_device.getInfo <CL_DEVICE_LOCAL_MEM_SIZE> ();
		std::cout << " Local Cache Size      : ";
		std::cout << local_cache / 1e3 << " KB\n";

		// Max number of arguments declared with the
		// '__constant' qualifier in a kernel (minimum value: 8).
		cl_uint constant_number =
			m_device.getInfo <CL_DEVICE_MAX_CONSTANT_ARGS> ();
		std::cout << " Constants per Kernel  : ";
		std::cout << constant_number << "\n";

        std::cout << " ----------------------------------------------\n";
	}

	float getProcessingTime()
	{
		return m_processing_time;
	}

  private :

	Timer m_timer;
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
