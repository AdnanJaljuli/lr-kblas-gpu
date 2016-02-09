#ifndef _TEST_TRSM_
#define _TEST_TRSM_

#include "testing_Xtr_common.h"

//==============================================================================================
template<class T>
int test_trsm(kblas_opts& opts, T alpha, cublasHandle_t cublas_handle){

  
  int nruns = opts.nruns;
  double   gflops, ref_perf = 0.0, ref_time = 0.0, kblas_perf = 0.0, kblas_time = 0.0, ref_error = 0.0;
  int M, N;
  int Am, An, Bm, Bn;
  int sizeA, sizeB;
  int lda, ldb, ldda, lddb;
  int ione     = 1;
  int ISEED[4] = {0,0,0,1};
  
  T *h_A, *h_B, *h_R;
  T *d_A, *d_B;
  
  
  USING
  cudaError_t err;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  cublasSideMode_t  side  = (opts.side   == KBLAS_Left  ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT);
  cublasFillMode_t  uplo  = (opts.uplo   == KBLAS_Lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER);
  cublasOperation_t trans = (opts.transA == KBLAS_Trans ? CUBLAS_OP_T : CUBLAS_OP_N);
  cublasDiagType_t  diag  = (opts.diag   == KBLAS_Unit  ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT);

  printf("    M     N     kblasTRSM Gflop/s (ms)   cublasTRSM Gflop/s (ms)  MaxError\n");
  printf("====================================================================\n");
  for( int i = 0; i < opts.ntest; ++i ) {
    for( int iter = 0; iter < opts.niter; ++iter ) {
      M = opts.msize[i];
      N = opts.nsize[i];
      
      gflops = FLOPS_TRSM(alpha, opts.side, M, N ) / 1e9;
      
      printf("%5d %5d   ",
             (int) M, (int) N);
      fflush( stdout );
      
      if ( opts.side == KBLAS_Left ) {
        lda = Am = M;
        An = M;
      } else {
        lda = Am = N;
        An = N;
      }
      ldb = Bm = M;
      Bn = N;
      
      ldda = ((lda+31)/32)*32;
      lddb = ((ldb+31)/32)*32;
      
      sizeA = lda*An;
      sizeB = ldb*Bn;

      TESTING_MALLOC_CPU( h_A, T, sizeA);
      TESTING_MALLOC_CPU( h_B, T, sizeB);

      TESTING_MALLOC_DEV( d_A, T, ldda*An);
      TESTING_MALLOC_DEV( d_B, T, lddb*Bn);

      if(opts.check)
      {
        nruns = 1;
        TESTING_MALLOC_CPU( h_R, T, sizeB);
      }
      // Initialize matrix and vector
      //printf("Initializing on cpu .. \n");
      Xrand_matrix(Am, An, h_A, lda);
      Xrand_matrix(Bm, Bn, h_B, ldb);
      kblas_make_hpd( Am, h_A, lda );
      
      cudaStream_t curStream = NULL;
      check_error(cublasSetStream(cublas_handle, curStream));
      
      check_error( cublasSetMatrix( Am, An, sizeof(T), h_A, lda, d_A, ldda ) );

      float time = 0;
      
      for(int r = 0; r < nruns; r++)
      {
        check_error( cublasSetMatrix( Bm, Bn, sizeof(T), h_B, ldb, d_B, lddb ) );
        
        start_timing();
        check_error( kblasXtrsm(cublas_handle,
                                side, uplo, trans, diag,
                                M, N,
                                &alpha, d_A, ldda,
                                        d_B, lddb) );
        time = get_elapsed_time();
        kblas_time += time;//to be in sec
      }
      kblas_time /= nruns;
      kblas_perf = gflops / (kblas_time / 1000.);

      if(opts.check){
        double normA = kblas_lange<T,double>('M', Am, An, h_A, lda);
        check_error( cublasGetMatrix( Bm, Bn, sizeof(T), d_B, lddb, h_R, ldb ) );
        double normX = kblas_lange<T,double>('M', Bm, Bn, h_R, ldb);

        T one = make_one<T>();
        T mone = make_zero<T>() - one;
        T invAlpha = one / alpha;
        check_error( kblasXtrmm(cublas_handle,
                                side, uplo, trans, diag,
                                M, N,
                                &invAlpha, d_A, ldda,
                                           d_B, lddb) );
        check_error( cublasGetMatrix( Bm, Bn, sizeof(T), d_B, lddb, h_R, ldb ) );
        kblasXaxpy( Bm * Bn, mone, h_B, 1, h_R, 1 );
        double normR = kblas_lange<T,double>('M', Bm, Bn, h_R, ldb);
        ref_error = normR / (normX * normA);
        //ref_error = Xget_max_error_matrix(h_B, h_R, Bm, Bn, ldb);
        free( h_R );
      }
      if(opts.time){
        cudaDeviceSynchronize();
      
        for(int r = 0; r < nruns; r++)
        {
          check_error( cublasSetMatrix( Bm, Bn, sizeof(T), h_B, ldb, d_B, lddb ) );
          
          start_timing();
          check_error( cublasXtrsm( cublas_handle,
                                    side, uplo, trans, diag,
                                    M, N,
                                    &alpha, d_A, ldda,
                                    d_B, lddb) );
          time = get_elapsed_time();
          ref_time += time;//to be in sec
        }
        ref_time /= nruns;
        ref_perf = gflops / (ref_time / 1000.);

        check_error( cublasGetMatrix( Bm, Bn, sizeof(T), d_B, lddb, h_B, ldb ) );
      }
      
      free( h_A );
      free( h_B );
      check_error(  cudaFree( d_A ) );
      check_error(  cudaFree( d_B ) );
      
      printf(" %7.2f (%7.2f)      %7.2f (%7.2f)         %8.2e\n",
             kblas_perf, kblas_time,
             ref_perf, ref_time,
             ref_error );
    }
    if ( opts.niter > 1 ) {
      printf( "\n" );
    }
  }
    	

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}


#endif