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
        {
            gpuBlasErrchk(cublasSgemm(*handle,
                                      transa,transb,
                                      m,n,k,
                                      static_cast<const float*>(alpha),
                                      static_cast<const float*>(d_A), lda,
                                      static_cast<const float*>(d_B), ldb,
                                      static_cast<const float*>(beta),
                                      static_cast<float*>(d_C), ldc));
            break;
        }

        case 1:
        {
            gpuBlasErrchk(cublasDgemm(*handle,
                                      transa,transb,
                                      m,n,k,
                                      static_cast<const double*>(alpha),
                                      static_cast<const double*>(d_A), lda,
                                      static_cast<const double*>(d_B), ldb,
                                      static_cast<const double*>(beta),
                                      static_cast<double*>(d_C), ldc));
            break;
        }

        case 2:
        {
            gpuBlasErrchk(cublasCgemm(*handle,
                                      transa,transb,
                                      m,n,k,
                                      static_cast<const float2*>(alpha),
                                      static_cast<const float2*>(d_A), lda,
                                      static_cast<const float2*>(d_B), ldb,
                                      static_cast<const float2*>(beta),
                                      static_cast<float2*>(d_C), ldc));
            break;
        }

        case 3:
        {
            gpuBlasErrchk(cublasZgemm(*handle,
                                      transa,transb,
                                      m,n,k,
                                      static_cast<const double2*>(alpha),
                                      static_cast<const double2*>(d_A), lda,
                                      static_cast<const double2*>(d_B), ldb,
                                      static_cast<const double2*>(beta),
                                      static_cast<double2*>(d_C), ldc));
            break;
        }
    }

    return;
}



void cublas_gemm_batched(cublasHandle_t *handle,
                         cublasOperation_t transa,
                         cublasOperation_t transb,
                         int m, int n, int k,
                         void *alpha,
                         void *d_A[], int lda,
                         void *d_B[], int ldb,
                         void *beta,
                         void *d_C[], int ldc,
                         int batchcount,
                         int dtype)
{
    
   switch(dtype) {
       
        case 0:
        {
            const float *A_ptr = static_cast<const float*>(*d_A);
            const float *B_ptr = static_cast<const float*>(*d_B);
            gpuBlasErrchk(cublasSgemmBatched(*handle,
                                             transa,transb,
                                             m,n,k,
                                             static_cast<const float*>(alpha),
                                             &A_ptr, lda,
                                             &B_ptr, ldb,
                                             static_cast<const float*>(beta),
                                             reinterpret_cast<float**>(d_C), ldc,
                                             batchcount));
            break;
        }
            
        case 1:
        {
            const double *A_ptr = static_cast<const double*>(*d_A);
            const double *B_ptr = static_cast<const double*>(*d_B);
            gpuBlasErrchk(cublasDgemmBatched(*handle,
                                             transa,transb,
                                             m,n,k,
                                             static_cast<const double*>(alpha),
                                             &A_ptr, lda,
                                             &B_ptr, ldb,
                                             static_cast<const double*>(beta),
                                             reinterpret_cast<double**>(d_C), ldc,
                                             batchcount));
            break;
        }
        
        case 2:
        {
            const float2 *A_ptr = static_cast<const float2*>(*d_A);
            const float2 *B_ptr = static_cast<const float2*>(*d_B);
            gpuBlasErrchk(cublasCgemmBatched(*handle,
                                             transa,transb,
                                             m,n,k,
                                             static_cast<const float2*>(alpha),
                                             &A_ptr, lda,
                                             &B_ptr, ldb,
                                             static_cast<const float2*>(beta),
                                             reinterpret_cast<float2**>(d_C), ldc,
                                             batchcount));
            break;
        }
            
        case 3:
        {
            const double2 *A_ptr = static_cast<const double2*>(*d_A);
            const double2 *B_ptr = static_cast<const double2*>(*d_B);
            gpuBlasErrchk(cublasZgemmBatched(*handle,
                                             transa,transb,
                                             m,n,k,
                                             static_cast<const double2*>(alpha),
                                             &A_ptr, lda,
                                             &B_ptr, ldb,
                                             static_cast<const double2*>(beta),
                                             reinterpret_cast<double2**>(d_C), ldc,
                                             batchcount));
            break;
        }
    }

    return; 
}



void cublas_gemm_strided_batched(cublasHandle_t *handle,
                                 cublasOperation_t transa,
                                 cublasOperation_t transb,
                                 int m, int n, int k,
                                 void *alpha,
                                 void *d_A, int lda, long long int strideA,
                                 void *d_B, int ldb, long long int strideB,
                                 void *beta,
                                 void *d_C, int ldc, long long int strideC,
                                 int batchcount,
                                 int dtype)
{
    switch(dtype) {
       
        case 0:
        {
            gpuBlasErrchk(cublasSgemmStridedBatched(*handle,
                                                    transa, transb,
                                                    m, n, k,
                                                    static_cast<const float*>(alpha),
                                                    static_cast<const float*>(d_A), lda, strideA,
                                                    static_cast<const float*>(d_B), ldb, strideB,
                                                    static_cast<const float*>(beta),
                                                    static_cast<float*>(d_C), ldc, strideC,
                                                    batchcount));
            break;
        }
        
        case 1:
        {
            gpuBlasErrchk(cublasDgemmStridedBatched(*handle,
                                                    transa, transb,
                                                    m, n, k,
                                                    static_cast<const double*>(alpha),
                                                    static_cast<const double*>(d_A), lda, strideA,
                                                    static_cast<const double*>(d_B), ldb, strideB,
                                                    static_cast<const double*>(beta),
                                                    static_cast<double*>(d_C), ldc, strideC,
                                                    batchcount));
            break;
        }
        
        case 2:
        {
            gpuBlasErrchk(cublasCgemmStridedBatched(*handle,
                                                    transa, transb,
                                                    m, n, k,
                                                    static_cast<const float2*>(alpha),
                                                    static_cast<const float2*>(d_A), lda, strideA,
                                                    static_cast<const float2*>(d_B), ldb, strideB,
                                                    static_cast<const float2*>(beta),
                                                    static_cast<float2*>(d_C), ldc, strideC,
                                                    batchcount));
            break;
        }
        
        case 3:
        {
            gpuBlasErrchk(cublasZgemmStridedBatched(*handle,
                                                    transa, transb,
                                                    m, n, k,
                                                    static_cast<const double2*>(alpha),
                                                    static_cast<const double2*>(d_A), lda, strideA,
                                                    static_cast<const double2*>(d_B), ldb, strideB,
                                                    static_cast<const double2*>(beta),
                                                    static_cast<double2*>(d_C), ldc, strideC,
                                                    batchcount));
            break;
        }
    }
    
    return;
}