////////////////////////////////////////////////////////////////////////////////
// High Performance Matrix Multiplication in OPENCL - B.Halimi, 2017
////////////////////////////////////////////////////////////////////////////////

#include "../inc/benchmark.h"
#include "../inc/constants.h"
#include "../inc/parser.h"

// Default benchmark configs
#define MATRIX_SIZE    512
#define N_ITERATION    20

int main(int argc, char* argv[])
{
	// SET UP //////////////////////////////////////////////////////////////////

	// Parse the program arguments
	Parser parser(argc, argv);
	bool reduced = parser.checkOption("-r");
	int matrix_size = std::atoi(parser.getOption("-s"));
	if (matrix_size == 0) matrix_size = MATRIX_SIZE;
	int n_iteration = std::atoi(parser.getOption("-i"));
	if (n_iteration == 0) n_iteration = N_ITERATION;

	// Initialize our OCL benchmarking tool
	Benchmark bench;
	if (!reduced) bench.printDeviceInfo();
	bench.loadProgram("../inc/constants.h");
	bench.loadProgram("../src/kernels.cl");

	// Create the host Matrices
	Matrix A(matrix_size, matrix_size);
	Matrix B(matrix_size, matrix_size);
	Matrix C(matrix_size, matrix_size);

	// Initialize our input matrices
	for (int e = 0; e < A.rows() * A.cols(); e++)
		A.data()[e] = (std::rand() % 256) / 256.f;

	for (int e = 0; e < B.rows() * B.cols(); e++)
		B.data()[e] = (std::rand() % 256) / 256.f;

	// Compute the reference result
	Matrix R;
	if (!reduced) {
		std::cout << " Computing the reference result ...\n";
   		R = A * B;
	}

	// Organize the kernel entries
	typedef struct {
	   std::string name;
	   int grid_size;
	   int block_size;
	} Entry;

	// Declare the different kernel entries
	std::vector <Entry> entries = {
		// Naive: each thread compute a row-col dot-product
		{        "GEMM",           matrix_size,               WGS},
		// SMB: processing by block using the shared memory
		{    "GEMM_SMB",           matrix_size,            SMB_TS},
		// CRB: caching in the shared memory and the registers
		{    "GEMM_CRB", matrix_size / CRB_STS,  CRB_TS / CRB_STS},
		// CRB-T: load transposed views of the tiles of the LHS
		{  "GEMM_CRB_T", matrix_size / CRB_STS,  CRB_TS / CRB_STS},
		// CRB-TR: reduce the amount private registers used
		{ "GEMM_CRB_TR", matrix_size / CRB_STS,  CRB_TS / CRB_STS}
	};

	// Setup the measurments
	int n_entries = entries.size();
	float time_records[n_entries] = {0};
	float error_counts[n_entries] = {0};

	// BENCHMARKING ////////////////////////////////////////////////////////////

	if (!reduced)
		std::cout << " Benchmarking the kernels (GPU) ...\n";

	for (size_t i = 0; i < n_iteration; i++) {
		for (size_t n = 0; n < n_entries; n++) {
			// Configure the Kernel
			bench.selectKernel(entries[n].name);
			bench.setGridSize(entries[n].grid_size);
			bench.setBlockSize(entries[n].block_size);
			// Perform the product
			bench.performKernel(A, B, C);
			time_records[n] += bench.getProcessingTime() / n_iteration;
			if (!reduced) error_counts[n] += ((R - C).norm() != 0);
		}
	}

	// PRINT RESULTS ///////////////////////////////////////////////////////////

	if (!reduced) {
		std::cout << ' ' << std::string(50, '-') << '\n';
		for (int n = 0; n < n_entries; n++) {
			std::cout << std::setprecision(3) << " [" << entries[n].name;
			std::cout << std::string(12 - entries[n].name.size(), ' ')  << "]";
			std::cout << " time : " << std::setw(5) << time_records[n] << "ms |";
			std::cout << " miss : " << std::setw(3) << error_counts[n];
			std::cout << "/" << n_iteration << "\n";
		}
		std::cout << ' ' << std::string(50, '-') << '\n';	
	} else {
		for (int i = 0; i < n_entries; i++)
			std::cout << time_records[i] << '\n';
	}

	return 0;
}
