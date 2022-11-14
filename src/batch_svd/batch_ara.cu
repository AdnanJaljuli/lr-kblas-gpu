#include <stdio.h>
#include <cublas_v2.h>

#include "kblas.h"
#include "kblas_struct.h"
#include "kblas_gpu_util.ch"
#include "batch_rand.h"

#include "batch_ara.h"

#include "thrust_wrappers.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/for_each.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/logical.h>
#include <thrust/transform.h>
// #include <mkl.h>

#include "workspace_queries.ch"
#include "ara_util.cuh"

//////////////////////////////////////////////////////////////////////////////////////
// Debug routines
//////////////////////////////////////////////////////////////////////////////////////
template<class Real>
void printDenseMatrix(Real *matrix, int ldm, int m, int n, int digits, const char* name)
{
	char format[64];
	assert(ldm >= m);
	
    sprintf(format, "%%.%de ", digits);
    printf("%s = ([\n", name);
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++)
            printf(format, matrix[i + j * ldm]);
        printf(";\n");
    }
    printf("]);\n");
}

template<class Real>
inline void printDenseMatrixGPU(Real** block_ptrs, int* ld_batch, int* rows_batch, int* cols_batch, int index, int digits, const char* name)
{
	thrust::device_ptr<Real*> dev_ptr(block_ptrs + index);
	Real* dev_ptr_val = *dev_ptr;
	thrust::device_ptr<Real> dev_data(dev_ptr_val);

	thrust::device_ptr<int> ld_ptr(ld_batch + index), rows_ptr(rows_batch + index), cols_ptr(cols_batch + index);
	int rows = *rows_ptr, cols = *cols_ptr, ld = *ld_ptr;

	char format[64];
	sprintf(format, "%%.%de ", digits);
    printf("%s = ([\n", name);
    for(int i = 0; i < rows; i++) 
	{
        for(int j = 0; j < cols; j++)
		{
			Real entry = *(dev_data + i + j * ld);
            printf(format, entry);
		}
        printf(";\n");
    }
    printf("]);\n");
}

template<class Real>
void copyDenseMatrix(
	Real** block_ptrs, int* ld_batch, int* rows_batch, int* cols_batch, int index, 
	thrust::host_vector<Real>& copy_block, int& rows, int& cols
)
{
	thrust::device_ptr<Real*> dev_ptr(block_ptrs + index);
	Real* dev_ptr_val = *dev_ptr;
	thrust::device_ptr<Real> dev_data(dev_ptr_val);
	
	thrust::device_ptr<int> ld_ptr(ld_batch + index), rows_ptr(rows_batch + index), cols_ptr(cols_batch + index);
	int ld = *ld_ptr;
	
	rows = *rows_ptr;
	cols = *cols_ptr;
	copy_block.resize(rows * cols);
    
	for(int i = 0; i < rows; i++) 
        for(int j = 0; j < cols; j++)
			copy_block[i + j * rows] = *(dev_data + i + j * ld);
}

//////////////////////////////////////////////////////////////////////////////////////
template<class T> 
void kblas_ara_batch_wsquery(KBlasWorkspaceState& requested_ws, int bs, int num_ops, int external_only)
{
	gemm_batch_nonuniform_wsquery_core(&requested_ws);
	ara_trsm_batch_wsquery<T>(requested_ws, num_ops, 0);
	
	if(!external_only)
	{
		// Align to sizeof(double) bytes
		requested_ws.d_data_bytes += requested_ws.d_data_bytes % sizeof(double);
		
		requested_ws.d_data_bytes += ( 
			num_ops * bs * bs + // G
			num_ops * bs +      // diag_R
			num_ops             // max_diag
		) * sizeof(double);	
		
		// Align to sizeof(int) bytes
		requested_ws.d_data_bytes += requested_ws.d_data_bytes % sizeof(int);
		
		requested_ws.d_data_bytes += (
			num_ops + // block_ranks
			num_ops + // op_samples
			num_ops   // small_vectors
		) * sizeof(int);
		
		// Pointers
		requested_ws.d_ptrs_bytes += requested_ws.d_ptrs_bytes % sizeof(T*);
		requested_ws.d_ptrs_bytes += num_ops * sizeof(T*); // Y_batch
				
		// Align to sizeof(double*) bytes
		requested_ws.d_ptrs_bytes += requested_ws.d_ptrs_bytes % sizeof(double*);
		requested_ws.d_ptrs_bytes += num_ops * sizeof(double*); // G_batch
	}
}

///////////////////////////////////////////////////////////////////////////
// Samplers
///////////////////////////////////////////////////////////////////////////
template<class T>
struct DenseSampler 
{
	T **M_batch;
	int *ldm_batch, *rows_batch, *cols_batch;
	kblasHandle_t handle;
	int max_rows, max_cols;
	
	DenseSampler(T **M_batch, int *ldm_batch, int *rows_batch, int *cols_batch, int max_rows, int max_cols, kblasHandle_t handle) 
	{
		this->M_batch = M_batch;
		this->ldm_batch = ldm_batch;
		this->rows_batch = rows_batch;
		this->cols_batch = cols_batch;
		this->max_rows = max_rows;
		this->max_cols = max_cols;
		this->handle = handle;
	}

	// A = M * B or A = M' * B
	int sample(T** B_batch, int* ldb_batch, int* samples_batch, T** A_batch, int* lda_batch, int max_samples, int num_ops, int transpose) 
	{
		
		if(!transpose)
		{
			check_error( kblas_gemm_batch(
				handle, KBLAS_NoTrans, KBLAS_NoTrans, rows_batch, samples_batch, cols_batch, 
				max_rows, max_samples, max_cols, 1, (const T**)M_batch, ldm_batch, 
				(const T**)B_batch, ldb_batch, 0, A_batch, lda_batch, num_ops
			) );
		}
		else
		{
			check_error( kblas_gemm_batch(
				handle, KBLAS_Trans, KBLAS_NoTrans, cols_batch, samples_batch, rows_batch, 
				max_cols, max_samples, max_rows, 1, (const T**)M_batch, ldm_batch,  
				(const T**)B_batch, ldb_batch, 0, A_batch, lda_batch, num_ops
			) );
		}
		
		return 1;
	}
};

// __________________________________________________________________________________________________________________________________________________
// __________________________________________MODIFIED CODE STARTS HERE_______________________________________________________________________________
// __________________________________________________________________________________________________________________________________________________

__host__ __device__ int getMOfromXY(unsigned int x, unsigned int y){
    static const unsigned int B[] = {0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF};
    static const unsigned int S[] = {1, 2, 4, 8};

    x = (x | (x << S[3])) & B[3];
    x = (x | (x << S[2])) & B[2];
    x = (x | (x << S[1])) & B[1];
    x = (x | (x << S[0])) & B[0];

    y = (y | (y << S[3])) & B[3];
    y = (y | (y << S[2])) & B[2];
    y = (y | (y << S[1])) & B[1];
    y = (y | (y << S[0])) & B[0];

    int z = x | (y << 1);
    return z;
}

__host__ __device__ int IndextoMOIndex(int num_segments, int n){
    unsigned int i = n%num_segments;
    unsigned int j = n/num_segments;
    return getMOfromXY(j, i);
}

template<class T>
__global__ void lr_sampling_batched(
    int tile_size, int batch_unit_size, T** U_ptrs, T** V_ptrs, int** scan_ranks,
    T** B_batch, T** A_batch, int* samples_batch, int max_rows, int max_cols, int max_samples, int transpose) {		

		T outputValue = 0;
		T sum = 0;

		if(transpose == 0) {
			__shared__ T shmemArray1[32][16];
			__shared__ T shmemArray2[32][16];
			unsigned int batch = blockIdx.y;
			unsigned int blockInBatch = blockIdx.x;

			for(unsigned int tile = 0; tile < batch_unit_size; ++tile) {
				int MOIndex = IndextoMOIndex(batch_unit_size, tile*batch_unit_size + blockInBatch);
				int rank, scanRankVal;
				if(blockInBatch == 0 && tile == 0) {
					rank = scan_ranks[batch][0];
					scanRankVal = 0;
				}
				else {
					rank = scan_ranks[batch][MOIndex] - scan_ranks[batch][MOIndex - 1];
					scanRankVal = scan_ranks[batch][MOIndex - 1];
				}

				// load V and Omega into shared memory
				if(threadIdx.x < rank) {
					shmemArray1[threadIdx.y][threadIdx.x] = V_ptrs[batch][scanRankVal*tile_size + threadIdx.x*tile_size + threadIdx.y];
				}
				if(rank*2 <= max_samples) {
					if(threadIdx.x >= max_samples - rank) {
						shmemArray1[threadIdx.y][threadIdx.x] = U_ptrs[batch][scanRankVal*tile_size + (threadIdx.x + rank - max_samples)*tile_size + threadIdx.y];
					}
				}
				if(threadIdx.x < samples_batch[batch]) {
					shmemArray2[threadIdx.y][threadIdx.x] = B_batch[batch][threadIdx.x*batch_unit_size*tile_size + tile*tile_size + threadIdx.y];
				}
				__syncthreads();

				if(threadIdx.y < rank && threadIdx.x < samples_batch[batch]) {
					// TODO: use warp shuffling: allocate multiple threadblocks per output element and let each do a part of the multiplication and then use shuffling
					sum = 0;
					for(unsigned int j = 0; j < tile_size; ++j) {
						sum += shmemArray1[j][threadIdx.y]*shmemArray2[j][threadIdx.x];
					}
				}
				__syncthreads();

				if(threadIdx.y < rank && threadIdx.x < samples_batch[batch]) {
					shmemArray2[threadIdx.y][threadIdx.x] = sum;
				}
				if(2*rank > max_samples) {
					if(threadIdx.x >= max_samples - rank) {
						shmemArray1[threadIdx.y][threadIdx.x] = U_ptrs[batch][scanRankVal*tile_size + (threadIdx.x + rank - max_samples)*tile_size + threadIdx.y];
					}
				}
				__syncthreads();

				// multiply U by the result and add it to output
				sum = 0;
				if(threadIdx.x < samples_batch[batch]) {
					for(unsigned int j = 0; j < rank; ++j) {
						sum += shmemArray1[threadIdx.y][max_samples - rank + j]*shmemArray2[j][threadIdx.x];
					}
					outputValue += sum;
				}
				__syncthreads();
			}
			if(threadIdx.x < samples_batch[batch]) {
				A_batch[batch][threadIdx.x*batch_unit_size*tile_size + blockInBatch*tile_size + threadIdx.y] = outputValue;
			}
		}
		else {
			// if(blockIdx.x >= (samples_batch[blockIdx.z] + 31)/32) {
			// 	return;
			// }
			__shared__ T shmemArray1[32][32];
			__shared__ T shmemArray2[32][32];
			unsigned int colInBatch = blockIdx.x*blockDim.x + threadIdx.x;
			unsigned int rowInBatch = blockIdx.y*blockDim.y + threadIdx.y;

			for(unsigned int tile = 0; tile < batch_unit_size; ++tile) {
				// multiply V by the sampling vector and store it in buffer
				int MOIndex = IndextoMOIndex(batch_unit_size, blockIdx.y*batch_unit_size + tile);
				int rank, scanRankVal;
				if(blockIdx.y == 0 && tile == 0) {
					rank = scan_ranks[blockIdx.z][0];
					scanRankVal = 0;
				}
				else {
					rank = scan_ranks[blockIdx.z][MOIndex] - scan_ranks[blockIdx.z][MOIndex - 1];
					scanRankVal = scan_ranks[blockIdx.z][MOIndex - 1];
				}
				if(colInBatch < samples_batch[blockIdx.z]) {
					// load U transpose and Omega into shared memory
					shmemArray2[threadIdx.y][threadIdx.x] = B_batch[blockIdx.z][colInBatch*batch_unit_size*tile_size + tile*tile_size + threadIdx.y];
				}
				if(threadIdx.x < rank) {
					shmemArray1[threadIdx.y][threadIdx.x] = U_ptrs[blockIdx.z][scanRankVal*tile_size + threadIdx.x*tile_size + threadIdx.y];
				}
				__syncthreads();

				sum = 0;
				if(colInBatch < samples_batch[blockIdx.z]) {
					if(threadIdx.y < rank) {
						for(unsigned int j = 0; j < tile_size; ++j) {
							sum += shmemArray1[j][threadIdx.y]*shmemArray2[j][threadIdx.x];
						}
					}
				}
				__syncthreads();

				if(threadIdx.x < rank) {
					shmemArray1[threadIdx.y][threadIdx.x] = V_ptrs[blockIdx.z][scanRankVal*tile_size + threadIdx.x*tile_size + threadIdx.y];
				}
				if(colInBatch < samples_batch[blockIdx.z]) {
					if(threadIdx.y < rank) {
						shmemArray2[threadIdx.y][threadIdx.x] = sum;
					}
				}
				__syncthreads();

				// multiply V by the result and add it to output
				sum = 0;
				if(colInBatch < samples_batch[blockIdx.z]) {
					for(unsigned int j = 0; j < rank; ++j) {
						sum += shmemArray1[threadIdx.y][j]*shmemArray2[j][threadIdx.x];
					}
				}
				__syncthreads();

				if(colInBatch < samples_batch[blockIdx.z]) {
					outputValue += sum;
				}
				__syncthreads();
			}
			if(colInBatch < samples_batch[blockIdx.z]) {
				A_batch[blockIdx.z][colInBatch*batch_unit_size*tile_size + rowInBatch] = outputValue;
			}
		}
}

template<class T>
int lr_sample(int tile_size, int batch_unit_size, T** U_ptrs, T** V_ptrs, int** scan_ranks, T** B_batch, int* ldb_batch, int* samples_batch, T** A_batch, int* lda_batch, int max_samples, int max_rows, int max_cols, int num_ops, int transpose) {
	
	if(transpose == 0) {
		dim3 numBlocks(batch_unit_size, num_ops);
		dim3 numThreadsPerBlock(max_samples, 32);
		assert(max_samples == 16);
		lr_sampling_batched <T> <<< numBlocks, numThreadsPerBlock >>> (tile_size, batch_unit_size, U_ptrs, V_ptrs, scan_ranks,
    		B_batch, A_batch, samples_batch, max_rows, max_cols, max_samples, transpose);
	}
	else {
		unsigned int numBlockCols = (max_samples + 31)/32;
		dim3 numBlocks(numBlockCols, batch_unit_size, num_ops);
		dim3 numThreadsPerBlock(min(32, max_samples), 32);
		lr_sampling_batched <T> <<< numBlocks, numThreadsPerBlock >>> (tile_size, batch_unit_size, U_ptrs, V_ptrs, scan_ranks,
    		B_batch, A_batch, samples_batch, max_rows, max_cols, max_samples, transpose);
	}
	cudaDeviceSynchronize();
	assert(cudaPeekAtLastError() == 0);
    return 1;
}

template<class T>
struct lr_DenseSampler
{
	T **U_ptrs, **V_ptrs;
	int tile_size, batch_unit_size;
	int **scan_ranks;
	int *rows_batch, *cols_batch;
	kblasHandle_t handle;
	int max_rows, max_cols;

	lr_DenseSampler(int tile_size, int batch_unit_size, T **U_ptrs, T **V_ptrs, int** scan_ranks, int *rows_batch, int *cols_batch, int max_rows, int max_cols, kblasHandle_t handle) 
	{
		this->tile_size = tile_size;
		this->batch_unit_size = batch_unit_size;
		this->U_ptrs = U_ptrs;
		this->V_ptrs = V_ptrs;
		this->scan_ranks = scan_ranks;
		this->rows_batch = rows_batch;
		this->cols_batch = cols_batch;
		this->max_rows = max_rows;
		this->max_cols = max_cols;
		this->handle = handle;
	}

	// A = M * B or A = M' * B
	int sample(T** B_batch, int* ldb_batch, int* samples_batch, T** A_batch, int* lda_batch, int max_samples, int num_ops, int transpose) 
	{
        lr_sample(tile_size, batch_unit_size, U_ptrs, V_ptrs, scan_ranks, B_batch, ldb_batch, samples_batch, A_batch, lda_batch, max_samples, max_rows, max_cols, num_ops, transpose);
		return 1;
	}
};

// __________________________________________________________________________________________________________________________________________________
// __________________________________________MODIFIED CODE ENDS HERE_______________________________________________________________________________
// __________________________________________________________________________________________________________________________________________________

template<class T, typename FuncPtr>
struct FunctionSampler 
{
	FuncPtr func;
	void* data;
	
	FunctionSampler(FuncPtr func, void* data)
	{
		this->func = func;
		this->data = data;
	}
	
	// A = M * B or A = M' * B
	int sample(T** B_batch, int* ldb_batch, int* samples_batch, T** A_batch, int* lda_batch, int max_samples, int num_ops, int transpose) 
	{
		return func(data, B_batch, ldb_batch, samples_batch, A_batch, lda_batch, max_samples, num_ops, transpose);
	}
};

///////////////////////////////////////////////////////////////////////////
template<class Real, class Sampler>
int kblas_ara_batch_template(
	kblasHandle_t handle, int* rows_batch, int* cols_batch, Sampler& sampler, 
	Real** A_batch, int* lda_batch, Real** B_batch, int* ldb_batch, int* ranks_batch, 
	Real tol, int max_rows, int max_cols, int max_rank, int bs, int r, kblasRandState_t rand_state, 
	int relative, int num_ops
)
{
	int rank = 0;
	const Real tolerance_scale = 1; //7.978845608028654; // (10 * sqrt(2 / pi))
	tol *= tolerance_scale;
	
	Real** Y_batch;
	
	double* diag_R, *max_diag;
	double* G_strided;
	double** G_batch_mp;
	int* block_ranks, *op_samples, *small_vectors;
	
	///////////////////////////////////////////////////////////////////////////
	// Workspace allocation
	///////////////////////////////////////////////////////////////////////////
	KBlasWorkspaceState external_ws, total_ws;
	kblas_ara_batch_wsquery<Real>(external_ws, bs, num_ops, 1);
	kblas_ara_batch_wsquery<Real>(total_ws, bs, num_ops, 0);

	KBlasWorkspaceState available_ws = handle->work_space.getAvailable();
	
	if(!total_ws.isSufficient(&available_ws))
		return KBLAS_InsufficientWorkspace;
	
	// Align workspace to sizeof(double) bytes
	external_ws.d_data_bytes += external_ws.d_data_bytes % sizeof(double);
	
	G_strided = (double*)((KBlasWorkspace::WS_Byte*)handle->work_space.d_data + external_ws.d_data_bytes);
	diag_R    = G_strided + num_ops * bs * bs;
	max_diag  = diag_R + num_ops * bs;
	
	external_ws.d_data_bytes += ( 
			num_ops * bs * bs + // G
			num_ops * bs + // diag_R
			num_ops        // max_diag
	) * sizeof(double);

	// Align to sizeof(int) bytes
	external_ws.d_data_bytes += external_ws.d_data_bytes % sizeof(int);

	block_ranks = (int*)((KBlasWorkspace::WS_Byte*)handle->work_space.d_data + external_ws.d_data_bytes);
	op_samples = block_ranks + num_ops;
	small_vectors = op_samples + num_ops;

	// Align to sizeof(Real*) bytes
	external_ws.d_ptrs_bytes += external_ws.d_ptrs_bytes % sizeof(Real*);
	Y_batch = (Real**)((KBlasWorkspace::WS_Byte*)handle->work_space.d_ptrs + external_ws.d_ptrs_bytes);	

	external_ws.d_ptrs_bytes += num_ops * sizeof(Real*); // Y_batch
	
	// Align to sizeof(double*) bytes
	external_ws.d_ptrs_bytes += external_ws.d_ptrs_bytes % sizeof(double*);
	G_batch_mp = (double**)((KBlasWorkspace::WS_Byte*)handle->work_space.d_ptrs + external_ws.d_ptrs_bytes);	

	///////////////////////////////////////////////////////////////////////////
	// Initializations
	///////////////////////////////////////////////////////////////////////////
	cudaStream_t kblas_stream = kblasGetStream(handle);
	// printDenseMatrixGPU(M_batch, ldm_batch, rows_batch, cols_batch, 0, 15, "M");
	
	// Initialize operation data
	fillArray(max_diag, num_ops, -1, kblas_stream);
	fillArray(small_vectors, num_ops, 0, kblas_stream);
	fillArray(ranks_batch, num_ops, 0, kblas_stream);
	
	// Copy over A to Y so we can advance Y 
	copyGPUArray(A_batch, Y_batch, num_ops, kblas_stream);
	
	// Generate array of pointers from strided data
	generateArrayOfPointers(G_strided, G_batch_mp, bs * bs, 0, num_ops, kblas_stream);

	///////////////////////////////////////////////////////////////////////////
	// Main Loop
	///////////////////////////////////////////////////////////////////////////
	while(rank < max_rank)
	{
		int samples = std::min(bs, max_rank - rank);
		
		// Set the op samples to 0 if the operation has converged
		int converged = kblas_ara_batch_set_samples(
			op_samples, small_vectors, 
			samples, r, num_ops, kblas_stream
		);
		
		if(converged == 1) break;
		
		// Generate random matrices Omega stored in B
		// Omega = randn(n, samples)
		Real** Omega = B_batch;
		check_error( kblas_rand_batch(
			handle, cols_batch, op_samples, Omega, ldb_batch, 
			max_cols, rand_state, num_ops
		) );
		
		// printDenseMatrixGPU(Omega, ldb_batch, cols_batch, op_samples, 0, 16, "Omega");
		
		// Take samples and store them in A
		// Y = M * Omega
		sampler.sample(
			Omega, ldb_batch, op_samples, 
			Y_batch, lda_batch, samples, num_ops, 0
		);

		// Set diag(R) = 1
		fillArray(diag_R, num_ops * bs, 1, kblas_stream);
	
		// Block CGS with one reorthogonalization step
		for(int i = 0; i < 2; i++)
		{
			// Project samples
			// Y = Y - Q * (Q' * Y) = Y - Q * Z
			// Store Z = Q' * Y in B
			Real** Z_batch = B_batch, **Q_batch = A_batch;
			check_error( kblas_gemm_batch(
				handle, KBLAS_Trans, KBLAS_NoTrans, ranks_batch, op_samples, rows_batch, 
				rank, samples, max_rows, 1, (const Real**)Q_batch, lda_batch,  
				(const Real**)Y_batch, lda_batch, 0, Z_batch, ldb_batch, num_ops
			) );
			check_error( kblas_gemm_batch(
				handle, KBLAS_NoTrans, KBLAS_NoTrans, rows_batch, op_samples, ranks_batch, 
				max_rows, samples, rank, -1, (const Real**)Q_batch, lda_batch,  
				(const Real**)Z_batch, ldb_batch, 1, Y_batch, lda_batch, num_ops
			) );

			// Pivoted panel orthogonalization using syrk+pstrf+trsm
			// Compute G = A'*A in mixed precision
			Real **R_batch = B_batch;
			check_error( kblas_ara_mp_syrk_batch_template(
				handle, rows_batch, op_samples, max_rows, samples, 
				(const Real**)Y_batch, lda_batch, G_batch_mp, op_samples, num_ops
			) );
			check_error( kblas_ara_fused_potrf_batch_template( 
				op_samples, G_batch_mp, op_samples, R_batch, ldb_batch, diag_R, bs, 
				block_ranks, num_ops, kblas_stream
			) );
			
			/*check_error( kblas_gemm_batch(
				handle, KBLAS_Trans, KBLAS_NoTrans, op_samples, op_samples, rows_batch, 
				samples, samples, max_rows, 1, (const Real**)Y_batch, lda_batch, 
				(const Real**)Y_batch, lda_batch, 0, R_batch, ldb_batch, num_ops
			) ); 
			check_error( kblas_ara_fused_potrf_batch( 
				op_samples, R_batch, ldb_batch, R_batch, ldb_batch, diag_R, bs, 
				block_ranks, num_ops, kblas_stream
			) );
			*/
			// Copy the ranks over to the samples in case the rank was less than the samples
			copyGPUArray(block_ranks, op_samples, num_ops, kblas_stream);
			
			// printDenseMatrixGPU(G_batch, ldb_batch, block_ranks, block_ranks, 0, 8, "R");
			
			check_error( kblas_ara_trsm_batch_template(
				handle, Y_batch, lda_batch, R_batch, ldb_batch, rows_batch, block_ranks, num_ops, max_rows, bs
			) );
		}

		// Count the number of vectors that have a small magnitude
		// also updates the rank, max diagonal and advances the Y_batch pointers
		check_error( kblas_ara_svec_count_batch_template(
			diag_R, bs, op_samples, ranks_batch, max_diag, Y_batch, lda_batch, 
			tol, r, small_vectors, relative, num_ops, kblas_stream
		) );
		
		// Advance the rank
		rank += samples;
	}
	
	// Finally, B = M' * A
	#if 1
	sampler.sample(
		A_batch, lda_batch, ranks_batch, 
		B_batch, ldb_batch, max_rank, num_ops, 1
	);
	#endif

	// printDenseMatrixGPU(A_batch, lda_batch, rows_batch, ranks_batch, 0, 16, "A");
	// printDenseMatrixGPU(B_batch, ldb_batch, cols_batch, ranks_batch, 0, 16, "B");
	return KBLAS_Success;
}

//////////////////////////////////////////////////////////////
// Wrappers
//////////////////////////////////////////////////////////////
int kblas_sara_batch(
	kblasHandle_t handle, int* rows_batch, int* cols_batch, float** M_batch, int* ldm_batch, 
	float** A_batch, int* lda_batch, float** B_batch, int* ldb_batch, int* ranks_batch, 
	float tol, int max_rows, int max_cols, int max_rank, int bs, int r, kblasRandState_t rand_state, 
	 int relative, int num_ops
)
{
	DenseSampler<float> dense_sampler(M_batch, ldm_batch, rows_batch, cols_batch, max_rows, max_cols, handle);
	
	return kblas_ara_batch_template<float, DenseSampler<float> >(
		handle, rows_batch, cols_batch, dense_sampler, A_batch, lda_batch, B_batch, ldb_batch, 
		ranks_batch, tol, max_rows, max_cols, max_rank, bs, r, rand_state, relative, num_ops
	);
}

// __________________________________________________________________________________________________________________________________________________
// __________________________________________MODIFIED CODE STARTS HERE_______________________________________________________________________________
// __________________________________________________________________________________________________________________________________________________
int lr_kblas_dara_batch(
	kblasHandle_t handle, int tile_size, int batch_unit_size, int* rows_batch, int* cols_batch, double** U_ptrs, double** V_ptrs, int** scan_ranks,
	double** A_batch, int* lda_batch, double** B_batch, int* ldb_batch, int* ranks_batch, 
	float tol, int max_rows, int max_cols, int max_rank, int bs, int r, kblasRandState_t rand_state, 
	int relative, int num_ops
)
{
	lr_DenseSampler<double> lr_dense_sampler(tile_size, batch_unit_size, U_ptrs, V_ptrs, scan_ranks, rows_batch, cols_batch, max_rows, max_cols, handle);

	return kblas_ara_batch_template<double, lr_DenseSampler<double> >(
		handle, rows_batch, cols_batch, lr_dense_sampler, A_batch, lda_batch, B_batch, ldb_batch, 
		ranks_batch, tol, max_rows, max_cols, max_rank, bs, r, rand_state, relative, num_ops
	);
}
// __________________________________________________________________________________________________________________________________________________
// __________________________________________MODIFIED CODE ENDS HERE_______________________________________________________________________________
// __________________________________________________________________________________________________________________________________________________


int kblas_dara_batch(
	kblasHandle_t handle, int* rows_batch, int* cols_batch, double** M_batch, int* ldm_batch, 
	double** A_batch, int* lda_batch, double** B_batch, int* ldb_batch, int* ranks_batch, 
	double tol, int max_rows, int max_cols, int max_rank, int bs, int r, kblasRandState_t rand_state, 
	int relative, int num_ops
)
{
	DenseSampler<double> dense_sampler(M_batch, ldm_batch, rows_batch, cols_batch, max_rows, max_cols, handle);
	
	return kblas_ara_batch_template<double, DenseSampler<double> >(
		handle, rows_batch, cols_batch, dense_sampler, A_batch, lda_batch, B_batch, ldb_batch, 
		ranks_batch, tol, max_rows, max_cols, max_rank, bs, r, rand_state, relative, num_ops
	);
}

int kblas_sara_batch_fn(
	kblasHandle_t handle, int* rows_batch, int* cols_batch, sara_sampler sampler, void* sampler_data, 
	float** A_batch, int* lda_batch, float** B_batch, int* ldb_batch, int* ranks_batch, 
	float tol, int max_rows, int max_cols, int max_rank, int bs, int r, kblasRandState_t rand_state, 
	int relative, int num_ops
)
{
	FunctionSampler<float, sara_sampler> func_sampler(sampler, sampler_data);
		
	return kblas_ara_batch_template<float, FunctionSampler<float, sara_sampler> >(
		handle, rows_batch, cols_batch, func_sampler, A_batch, lda_batch, B_batch, ldb_batch, 
		ranks_batch, tol, max_rows, max_cols, max_rank, bs, r, rand_state, relative, num_ops
	);
}

int kblas_dara_batch_fn(
	kblasHandle_t handle, int* rows_batch, int* cols_batch, dara_sampler sampler, void* sampler_data, 
	double** A_batch, int* lda_batch, double** B_batch, int* ldb_batch, int* ranks_batch, 
	double tol, int max_rows, int max_cols, int max_rank, int bs, int r, kblasRandState_t rand_state, 
	int relative, int num_ops
)
{
	FunctionSampler<double, dara_sampler> func_sampler(sampler, sampler_data);
		
	return kblas_ara_batch_template<double, FunctionSampler<double, dara_sampler> >(
		handle, rows_batch, cols_batch, func_sampler, A_batch, lda_batch, B_batch, ldb_batch, 
		ranks_batch, tol, max_rows, max_cols, max_rank, bs, r, rand_state, relative, num_ops
	);
}

//////////////////////////////////////////////////////////////
// Workspace queries
//////////////////////////////////////////////////////////////
void kblasSara_batch_wsquery(kblasHandle_t handle, int bs, int num_ops)
{
	KBlasWorkspaceState ws;
	kblas_ara_batch_wsquery<float>(ws, bs, num_ops, 0);
	handle->work_space.requested_ws_state.pad(&ws);
}

void kblasDara_batch_wsquery(kblasHandle_t handle, int bs, int num_ops)
{
	KBlasWorkspaceState ws;
	kblas_ara_batch_wsquery<double>(ws, bs, num_ops, 0);
	handle->work_space.requested_ws_state.pad(&ws);
}

void kblasSara_trsm_batch_wsquery(kblasHandle_t handle, int num_ops)
{
	KBlasWorkspaceState ws;
	ara_trsm_batch_wsquery<float>(ws, num_ops, 0);
	handle->work_space.requested_ws_state.pad(&ws);
}

void kblasDara_trsm_batch_wsquery(kblasHandle_t handle, int num_ops)
{
	KBlasWorkspaceState ws;
	ara_trsm_batch_wsquery<double>(ws, num_ops, 0);
	handle->work_space.requested_ws_state.pad(&ws);
}

//////////////////////////////////////////////////////////////
// utility wrappers
//////////////////////////////////////////////////////////////
int kblas_sara_svec_count_batch(
	double* diag_R, int bs, int* op_samples, int* ranks_batch, double* max_diag, float** Y_batch, 
	int* ldy_batch, float tol, int r, int* small_vectors, int relative, int num_ops, cudaStream_t stream
)
{
	return kblas_ara_svec_count_batch_template(
		diag_R, bs, op_samples, ranks_batch, max_diag, Y_batch, ldy_batch, 
		tol, r, small_vectors, relative, num_ops, stream
	);
}

int kblas_dara_svec_count_batch(
	double* diag_R, int bs, int* op_samples, int* ranks_batch, double* max_diag, double** Y_batch, 
	int* ldy_batch, double tol, int r, int* small_vectors, int relative, int num_ops, cudaStream_t stream
)
{
	return kblas_ara_svec_count_batch_template(
		diag_R, bs, op_samples, ranks_batch, max_diag, Y_batch, ldy_batch, 
		tol, r, small_vectors, relative, num_ops, stream
	);	
}

int kblas_sara_trsm_batch(
	kblasHandle_t handle, float** B_batch, int* ldb_batch, float** A_batch, int* lda_batch, 
	int* rows_batch, int* cols_batch, int num_ops, int max_rows, int bs
)
{
	return kblas_ara_trsm_batch_template(
		handle, B_batch, ldb_batch, A_batch, lda_batch, 
		rows_batch, cols_batch, num_ops, max_rows, bs
	);
}

int kblas_dara_trsm_batch(
	kblasHandle_t handle, double** B_batch, int* ldb_batch, double** A_batch, int* lda_batch, 
	int* rows_batch, int* cols_batch, int num_ops, int max_rows, int bs
)
{
	return kblas_ara_trsm_batch_template(
		handle, B_batch, ldb_batch, A_batch, lda_batch, 
		rows_batch, cols_batch, num_ops, max_rows, bs
	);
}

int kblas_sara_fused_potrf_batch( 
	int* op_samples, double** A_batch, int* lda_batch, float** R_batch, int* ldr_batch, 
	double* diag_R, int bs, int* block_ranks, int num_ops, cudaStream_t stream
)
{
	return kblas_ara_fused_potrf_batch_template(
		op_samples, A_batch, lda_batch, R_batch, ldr_batch, 
		diag_R, bs, block_ranks, num_ops, stream
	);
}

int kblas_dara_fused_potrf_batch( 
	int* op_samples, double** A_batch, int* lda_batch, double** R_batch, int* ldr_batch, 
	double* diag_R, int bs, int* block_ranks, int num_ops, cudaStream_t stream
)
{
	return kblas_ara_fused_potrf_batch_template(
		op_samples, A_batch, lda_batch, R_batch, ldr_batch, 
		diag_R, bs, block_ranks, num_ops, stream
	);	
}

int kblas_sara_mp_syrk_batch(
	kblasHandle_t handle, int* m, int* n, int max_m, int max_n, 
	const float** A, int* lda, double** B, int* ldb, int batchCount 
)
{
	return kblas_ara_mp_syrk_batch_template(
		handle, m, n, max_m, max_n, A, lda, B, ldb, batchCount 
	);
}

int kblas_dara_mp_syrk_batch(
	kblasHandle_t handle, int* m, int* n, int max_m, int max_n, 
	const double** A, int* lda, double** B, int* ldb, int batchCount 
)
{
	return kblas_ara_mp_syrk_batch_template(
		handle, m, n, max_m, max_n, A, lda, B, ldb, batchCount 
	);
}
