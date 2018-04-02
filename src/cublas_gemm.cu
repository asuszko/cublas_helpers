#include <cuda.h>
#include <cublas_v2.h>
#include "cu_errchk.h"
#include "cublas_gemm.h"


/**
*  http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemm
*/
void cublas_gemm(cublasHandle_t *handle,
                 cublasOperation_t transa,
                 cublasOperation_t transb,
                 int m, int n, int k,
                 void *alpha,
                 void *d_A, int lda,
                 void *d_B, int ldb,
                 void *beta,
                 void *d_C, int ldc,
                 int dtype)
{

    switch(dtype) {

        case 0:
            gpuBlasErrchk(cublasSgemm(*handle,
                                      transa,transb,
                                      m,n,k,
                                      static_cast<float*>(alpha),
                                      static_cast<float*>(d_A), lda,
                                      static_cast<float*>(d_B), ldb,
                                      static_cast<float*>(beta),
                                      static_cast<float*>(d_C), ldc));
            break;

        case 1:
            gpuBlasErrchk(cublasDgemm(*handle,
                                      transa,transb,
                                      m,n,k,
                                      static_cast<double*>(alpha),
                                      static_cast<double*>(d_A), lda,
                                      static_cast<double*>(d_B), ldb,
                                      static_cast<double*>(beta),
                                      static_cast<double*>(d_C), ldc));
            break;

        case 2:
            gpuBlasErrchk(cublasCgemm(*handle,
                                      transa,transb,
                                      m,n,k,
                                      static_cast<float2*>(alpha),
                                      static_cast<float2*>(d_A), lda,
                                      static_cast<float2*>(d_B), ldb,
                                      static_cast<float2*>(beta),
                                      static_cast<float2*>(d_C), ldc));
            break;

        case 3:
            gpuBlasErrchk(cublasZgemm(*handle,
                                      transa,transb,
                                      m,n,k,
                                      static_cast<double2*>(alpha),
                                      static_cast<double2*>(d_A), lda,
                                      static_cast<double2*>(d_B), ldb,
                                      static_cast<double2*>(beta),
                                      static_cast<double2*>(d_C), ldc));
            break;
    }

    return;
}
