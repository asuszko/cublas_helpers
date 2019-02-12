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
    const void *alpha,
    const void *d_A, int lda,
    const void *d_B, int ldb,
    const void *beta,
    void *d_C, int ldc,
    int dtype,
    bool m3m)
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
            if (m3m) {
                gpuBlasErrchk(cublasCgemm3m(*handle,
                    transa,transb,
                    m,n,k,
                    static_cast<const float2*>(alpha),
                    static_cast<const float2*>(d_A), lda,
                    static_cast<const float2*>(d_B), ldb,
                    static_cast<const float2*>(beta),
                    static_cast<float2*>(d_C), ldc));
                break;
            }
            else {
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
        }

        case 3:
        {
            if (m3m) {
                gpuBlasErrchk(cublasZgemm3m(*handle,
                    transa,transb,
                    m,n,k,
                    static_cast<const double2*>(alpha),
                    static_cast<const double2*>(d_A), lda,
                    static_cast<const double2*>(d_B), ldb,
                    static_cast<const double2*>(beta),
                    static_cast<double2*>(d_C), ldc));
                break;
            }
            else {
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

        case 4:
        {
            gpuBlasErrchk(cublasHgemm(*handle,
                transa,transb,
                m,n,k,
                static_cast<const __half *>(alpha),
                static_cast<const __half *>(d_A), lda,
                static_cast<const __half *>(d_B), ldb,
                static_cast<const __half *>(beta),
                static_cast<__half *>(d_C), ldc));
            break;
        }


    }

    return;
}



void cublas_gemm_strided_batched(cublasHandle_t *handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const void *alpha,
    const void *d_A, int lda, uint64_t strideA,
    const void *d_B, int ldb, uint64_t strideB,
    const void *beta,
    void *d_C, int ldc, uint64_t strideC,
    int batchcount,
    int dtype,
    bool m3m)
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
            if (m3m) {
                gpuBlasErrchk(cublasCgemm3mStridedBatched(*handle,
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
            else {
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

        case 4:
        {
            gpuBlasErrchk(cublasHgemmStridedBatched(*handle,
                transa, transb,
                m, n, k,
                static_cast<const __half *>(alpha),
                static_cast<const __half *>(d_A), lda, strideA,
                static_cast<const __half *>(d_B), ldb, strideB,
                static_cast<const __half *>(beta),
                static_cast<__half *>(d_C), ldc, strideC,
                batchcount));
            break;
        }
    }
    
    return;
}
