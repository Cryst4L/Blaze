////////////////////////////////////////////////////////////////////////////////
// -----------------------------------------------------------------------------
// OPENCL Matrix Multiplication Kernel Codes - B.Halimi - 2017
// -----------------------------------------------------------------------------
// To stay consistent with the multi-dimensional array convention of the C, we
// use in this project the row-major memory layout, according to the following
// formula:
//
//    z M(row, col) = *(M.data + row * M.cols + col)
//
// As a result, elements of a same row will are coalescent.
//
// Nevertheless, in respect to the FORTRAN BLAS convention, threads sent to
// the Compute Units are packed by IDs of a same column.
//
// When using the shared memory this should induce an unefficient execution
// of the WARPs loading the coefficients of A and the WARPs writing back the
// coefficient of C:techniques
//
// In these WARPs each concurrent Scalar Processors would be attempting to use
// the same memory BANK thus resulting in sequential memory accesses !
//
// To address this issue we swap the X and Y thread-ID, and thereby ensure
// that a WARP always execute threads computing coefficients of a same row of
// our output matrix.
//
////////////////////////////////////////////////////////////////////////////////
#ifdef ROW_MAJOR_IDX
	#define IDX0 1
	#define IDX1 0
#else
	#define IDX0 0
	#define IDX1 1
#endif
////////////////////////////////////////////////////////////////////////////////
// 1. NAIVE:
// -----------------------------------------------------------------------------
// Each threads computes a full row-column dot product, pulling the
// coefficients directly from the global memory.
////////////////////////////////////////////////////////////////////////////////
__kernel void GEMM(
	const int A_width,
	const int B_width,
	const __global float* A_mem,
	const __global float* B_mem,
	      __global float* C_mem)
{
	const int row = get_global_id(IDX0);
	const int col = get_global_id(IDX1);

	float acc = 0.0f;
	for (int e = 0; e < A_width; e++)
		acc += A_mem[row * A_width + e] * B_mem[e * B_width + col];

	C_mem[row * B_width + col] = acc;
}

////////////////////////////////////////////////////////////////////////////////
// 2. SHARED MEMORY BLOCKING (SMB):
// -----------------------------------------------------------------------------
// Product performed block by block after pulling tiles of the given buffers to
// the shared (__local) memory to reduce computing latencies.
////////////////////////////////////////////////////////////////////////////////
__kernel void GEMM_SMB(
	const int A_width,
	const int B_width,
	const __global float* A_mem,
	const __global float* B_mem,
	      __global float* C_mem)
{
	const int row = get_local_id(IDX0);
	const int col = get_local_id(IDX1);

	const int tile_row = get_group_id(IDX0);
	const int tile_col = get_group_id(IDX1);

	__local float tile_A[SMB_TS][SMB_TS];
	__local float tile_B[SMB_TS][SMB_TS];

	float acc = 0.0f;

	for (int m = 0; m < (A_width / SMB_TS); ++m) {

		int tile_offset_A = (A_width * SMB_TS) * tile_row + (SMB_TS) * m;
		int tile_offset_B = (B_width * SMB_TS) * m + (SMB_TS) * tile_col;

		tile_A[row][col] = A_mem[tile_offset_A + row * A_width + col];
		tile_B[row][col] = B_mem[tile_offset_B + row * B_width + col];

		barrier(CLK_LOCAL_MEM_FENCE);

		for (int e = 0; e < SMB_TS; e++)
			acc += tile_A[row][e] * tile_B[e][col];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	int offset_C = (B_width * SMB_TS) * tile_row + (SMB_TS) * tile_col;

	C_mem[offset_C + row * B_width + col] = acc;
}

////////////////////////////////////////////////////////////////////////////////
// 3. CASCADED REGISTER BLOCKING (CRB) :
// -----------------------------------------------------------------------------
// To reduce the data throughput compared to the previous method, each thread
// will compute a sub-block of the output matrix rather than a single coeff.
//
// Previously each of the PxQxR elementary products involved in a full sub-tile
// product, where encompassed between two reads and one write in the shared
// memory, thus inducing a huge computation loss in memory accesses.
//
// To address this issue we can push the blocking principle one step further
// and compute the tiles themselves by block: rather than computing a single
// coefficcient of the output tile, each Scalar A_widthrocessor will compute a
// sub-tile of it.
//
// As a result computing M³ products between two MxM sub-blocks will only
// require 2xM² accesses in the shared memory compared to 2xM³ required without
// this additional caching technique.
////////////////////////////////////////////////////////////////////////////////
__kernel void GEMM_CRB(
	const int A_width,
	const int B_width,
	const __global float* A_mem,
	const __global float* B_mem,
	      __global float* C_mem)
{
	const int tile_row = get_group_id(IDX0);
	const int tile_col = get_group_id(IDX1);

	const int sub_tile_row = get_local_id(IDX0);
	const int sub_tile_col = get_local_id(IDX1);

	__local float tile_A[CRB_TS][CRB_TS];
	__local float tile_B[CRB_TS][CRB_TS];

	float sub_tile_A[CRB_STS][CRB_STS];
	float sub_tile_B[CRB_STS][CRB_STS];
	float sub_tile_C[CRB_STS][CRB_STS];

	for (int i = 0; i < CRB_STS; i++)
		for (int j = 0; j < CRB_STS; j++)
			sub_tile_C[i][j] = 0.0f;

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int m = 0; m < (A_width / CRB_TS); ++m) {

		int tile_offset_A = (A_width * CRB_TS) * tile_row + (CRB_TS) * m;
		int tile_offset_B = (B_width * CRB_TS) * m + (CRB_TS) * tile_col;

		for (int i = 0; i < CRB_STS; i++) {
			for (int j = 0; j < CRB_STS; j++) {
				int row = CRB_STS * sub_tile_row + i;
				int col = CRB_STS * sub_tile_col + j;
				tile_A[row][col] = A_mem[tile_offset_A + (A_width * row) + col];
				tile_B[row][col] = B_mem[tile_offset_B + (B_width * row) + col];
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		for (int n = 0; n < (CRB_TS / CRB_STS); n++) {

		for (int i = 0; i < CRB_STS; i++) // HERE: reverse ?
			for (int j = 0; j < CRB_STS; j++)
				sub_tile_A[i][j] =
					tile_A[CRB_STS * sub_tile_row + i][CRB_STS * n + j];

		for (int i = 0; i < CRB_STS; i++)
			for (int j = 0; j < CRB_STS; j++) // HERE: reverse ?
				sub_tile_B[i][j] =
					tile_B[CRB_STS * n + i][CRB_STS * sub_tile_col + j];

			barrier(CLK_LOCAL_MEM_FENCE);

			for (int i = 0; i < CRB_STS; i++)
				for (int j = 0; j < CRB_STS; j++)
					for (int e = 0; e < CRB_STS; e++)
						sub_tile_C[i][j] +=
							sub_tile_A[i][e] * sub_tile_B[e][j];
		}
	}

	int offset_C =
		(B_width * CRB_TS) * tile_row + (CRB_TS) * tile_col +
		(B_width * CRB_STS) * sub_tile_row + (CRB_STS) * sub_tile_col;

	for (int i = 0; i < CRB_STS; i++)
		for (int j = 0; j < CRB_STS; j++)
			C_mem[offset_C + i * B_width + j] = sub_tile_C[i][j];
}

////////////////////////////////////////////////////////////////////////////////
// 4. CRB TRANSPOSED:
// -----------------------------------------------------------------------------
// Despite the previous optimizations, most the processing time is still spent
// in memory accesses. The pulling of the sub-tiles of A into the private
// registers are, for example, critically slow.
//
// To understand why, lets consider an iteration of the inner 'for' loop:
// For each value of 'n', we attempt to concurently perform an outer product
// between a sub-tile-column of A and sub-tile-row of B.
//
// To do such thing, each thread of a Work-Group first loads the sub-tile of A
// aligned with the output sub-tile it is supposed to compute.
//
// Therefore, threads of a same row operate in a '1-to-n' manner (which is good)
// but threads of a same column attempt to read in the same bank in the CUDA
// shared memory space, resulting in sequential memory accesses.
//
// This issue does not arise when loading the sub-tile-row of B:
// Threads of a same column operate in a '1-to-n' manner and threads of a same
// row accesses different memory banks, thus triggering a SIMD read.
//
// To replicate this behaviour for the tile of A, we can load and manipulate
// a transposed view of this sub-tile. In practice, it does significantly reduce
// the processing time.
////////////////////////////////////////////////////////////////////////////////
__kernel void GEMM_CRB_T(
	const int A_width,
	const int B_width,
	const __global float* A_mem,
	const __global float* B_mem,
	      __global float* C_mem)
{
	const int tile_row = get_group_id(IDX0);
	const int tile_col = get_group_id(IDX1);

	const int sub_tile_row = get_local_id(IDX0);
	const int sub_tile_col = get_local_id(IDX1);

	__local float tile_A[CRB_TS][CRB_TS];
	__local float tile_B[CRB_TS][CRB_TS];

	float sub_tile_A[CRB_STS][CRB_STS];
	float sub_tile_B[CRB_STS][CRB_STS];
	float sub_tile_C[CRB_STS][CRB_STS];

	for (int i = 0; i < CRB_STS; i++)
		for (int j = 0; j < CRB_STS; j++)
			sub_tile_C[i][j] = 0.0f;

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int m = 0; m < (A_width / CRB_TS); ++m) {

		int tile_offset_A = (A_width * CRB_TS) * tile_row + (CRB_TS) * m;
		int tile_offset_B = (B_width * CRB_TS) * m + (CRB_TS) * tile_col;

		for (int i = 0; i < CRB_STS; i++) {
			for (int j = 0; j < CRB_STS; j++) {
				int row = CRB_STS * sub_tile_row + i;
				int col = CRB_STS * sub_tile_col + j;
				tile_A[col][row] = A_mem[tile_offset_A + (A_width * row) + col];
				tile_B[row][col] = B_mem[tile_offset_B + (B_width * row) + col];
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		for (int n = 0; n < (CRB_TS / CRB_STS); n++) {

			for (int i = 0; i < CRB_STS; i++)
				for (int j = 0; j < CRB_STS; j++)
					sub_tile_A[i][j] =
						tile_A[CRB_STS * n + j][CRB_STS * sub_tile_row + i];

			for (int i = 0; i < CRB_STS; i++)
				for (int j = 0; j < CRB_STS; j++)
					sub_tile_B[i][j] =
						tile_B[CRB_STS * n + i][CRB_STS * sub_tile_col + j];

			barrier(CLK_LOCAL_MEM_FENCE);

			for (int i = 0; i < CRB_STS; i++)
				for (int j = 0; j < CRB_STS; j++)
					for (int e = 0; e < CRB_STS; e++)
						sub_tile_C[i][j] +=
							sub_tile_A[i][e] * sub_tile_B[e][j];
		}
	}

	int offset_C =
		(B_width * CRB_TS) * tile_row + (CRB_TS) * tile_col +
		(B_width * CRB_STS) * sub_tile_row + (CRB_STS) * sub_tile_col;

	for (int i = 0; i < CRB_STS; i++)
		for (int j = 0; j < CRB_STS; j++)
			C_mem[offset_C + i * B_width + j] = sub_tile_C[i][j];
}

////////////////////////////////////////////////////////////////////////////////
// 4. CRBT REDUCED:
// -----------------------------------------------------------------------------
// Scalar Processors hold a very limited amount of registers, typically between
// 63 and 255. Nevertheless, to compute the sub-tile products, the last kernel
// had to buffer the 3 of them in the private memory.
//
// An optimization of the memory usage can then be obtained by remarking that
// the product of two sub-tiles can be computed as a sum of the outer-product
// between columns and rows of them.
//
// As a consequence, we only need to buffer a pair of thoses in the private
// field, which reduced the amount of registers needed from 3n² to (n² + 2*n).
//
// If the PTX information query does confirm this reduction, it unfortunately
// does not reduce the processing time on the developement device (GTX 1060 M)
// as the optimal sub-tile sizing (STS=4) already results in the allocation of
// only 72 on the 255 regiters available (without this optimization).
////////////////////////////////////////////////////////////////////////////////
__kernel void GEMM_CRB_TR(
	const int A_width,
	const int B_width,
	const __global float* A_mem,
	const __global float* B_mem,
	      __global float* C_mem)
{
	const int tile_row = get_group_id(IDX0);
	const int tile_col = get_group_id(IDX1);

	const int sub_tile_row = get_local_id(IDX0);
	const int sub_tile_col = get_local_id(IDX1);

	__local float tile_A[CRB_TS][CRB_TS];
	__local float tile_B[CRB_TS][CRB_TS];

	float sub_tile_col_A[CRB_STS];
	float sub_tile_row_B[CRB_STS];
	float sub_tile_C[CRB_STS][CRB_STS];

	for (int i = 0; i < CRB_STS; i++)
		for (int j = 0; j < CRB_STS; j++)
			sub_tile_C[i][j] = 0.0f;

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int m = 0; m < (A_width / CRB_TS); ++m) {

		int tile_offset_A = (A_width * CRB_TS) * tile_row + (CRB_TS) * m;
		int tile_offset_B = (B_width * CRB_TS) * m + (CRB_TS) * tile_col;

		for (int i = 0; i < CRB_STS; i++) {
			for (int j = 0; j < CRB_STS; j++) {
				int row = CRB_STS * sub_tile_row + i;
				int col = CRB_STS * sub_tile_col + j;
				tile_A[col][row] = A_mem[tile_offset_A + (A_width * row) + col];
				tile_B[row][col] = B_mem[tile_offset_B + (B_width * row) + col];
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		for (int n = 0; n < (CRB_TS / CRB_STS); n++) {

			for (int p = 0; p < CRB_STS; p++) {

				for(int e = 0; e < CRB_STS; e++)
					sub_tile_col_A[e] =
						tile_A[CRB_STS * n + p][CRB_STS * sub_tile_row + e];

				for(int e = 0; e < CRB_STS; e++)
					sub_tile_row_B[e] =
						tile_B[CRB_STS * n + p][CRB_STS * sub_tile_col + e];

				for (int i = 0; i < CRB_STS; i++)
					for (int j = 0; j < CRB_STS; j++)
						sub_tile_C[i][j] +=
							sub_tile_col_A[i] * sub_tile_row_B[j];

				barrier(CLK_LOCAL_MEM_FENCE);
			}
		}
	}

	int offset_C =
		(B_width * CRB_TS) * tile_row + (CRB_TS) * tile_col +
		(B_width * CRB_STS) * sub_tile_row + (CRB_STS) * sub_tile_col;

	for (int i = 0; i < CRB_STS; i++)
		for (int j = 0; j < CRB_STS; j++)
			C_mem[offset_C + i * B_width + j] = sub_tile_C[i][j];
}
