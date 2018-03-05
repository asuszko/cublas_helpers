#include <cuda.h>
#include <cublas_v2.h>
#include "cu_errchk.h"
#include "cublas_gemm.h"


template<typename T>
inline cublasStatus_t cublasTgemm(cublasHandle_t *handle,
      cublasOperation_t transa, cublasOperation_t transb,
      int m, int n, int k,
      const T *alpha,
      const T *A, int lda,
      const T *B, int ldb,
      const T *beta,
      T *C, int ldc)
{
    if (std::is_same<T, float>::value) {
        return cublasSgemm(*handle, transa, transb, m, n, k,
                          (float *)alpha, (float *)A, lda,
                          (float *)B, ldb, (float *)beta,
                          (float *)C, ldc);
    }
    else
    if (std::is_same<T, double>::value) {
        return cublasDgemm(*handle, transa, transb, m, n, k,
                          (double *)alpha, (double *)A, lda,
                          (double *)B, ldb, (double *)beta,
                          (double *)C, ldc);
    }
    else
    if (std::is_same<T, cuComplex>::value) {
        return cublasCgemm(*handle, transa, transb, m, n, k,
                          (cuComplex *)alpha, (cuComplex *)A, lda,
                          (cuComplex *)B, ldb, (cuComplex *)beta,
                          (cuComplex *)C, ldc);
    }
    else
    if (std::is_same<T, cuDoubleComplex>::value) {
        return cublasZgemm(*handle, transa, transb, m, n, k,
                          (cuDoubleComplex *)alpha, (cuDoubleComplex *)A, lda,
                          (cuDoubleComplex *)B, ldb, (cuDoubleComplex *)beta,
                          (cuDoubleComplex *)C, ldc);
    }
    else {
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }
}




/* C compatible version that requires a dtype_id to be converted
to the proper data type. */
void cublas_gemm(cublasHandle_t *handle,
                 cublasOperation_t transa,
                 cublasOperation_t transb,
                 int m, int n, int k,
                 const void *alpha,
                 const void *d_A, int lda,
                 const void *d_B, int ldb,
                 const void *beta,
                 void *d_C, int ldc,
                 int dtype)
{

    switch(dtype) {

        case 0: {
            gpuBlasErrchk(cublasTgemm(handle,
                                      transa,transb,
                                      m,n,k,
                                      (float*)alpha,
                                      (float*)d_A, lda,
                                      (float*)d_B, ldb,
                                      (float*)beta,
                                      (float*)d_C, ldc));
            break;
        }

        case 1: {
            gpuBlasErrchk(cublasTgemm(handle,
                                      transa,transb,
                                      m,n,k,
                                      (double*)alpha,
                                      (double*)d_A, lda,
                                      (double*)d_B, ldb,
                                      (double*)beta,
                                      (double*)d_C, ldc));
            break;
        }

        case 2: {
            gpuBlasErrchk(cublasTgemm(handle,
                                      transa,transb,
                                      m,n,k,
                                      (cuComplex*)alpha,
                                      (cuComplex*)d_A, lda,
                                      (cuComplex*)d_B, ldb,
                                      (cuComplex*)beta,
                                      (cuComplex*)d_C, ldc));
            break;
        }

        case 3: {
            gpuBlasErrchk(cublasTgemm(handle,
                                      transa,transb,
                                      m,n,k,
                                      (cuDoubleComplex*)alpha,
                                      (cuDoubleComplex*)d_A, lda,
                                      (cuDoubleComplex*)d_B, ldb,
                                      (cuDoubleComplex*)beta,
                                      (cuDoubleComplex*)d_C, ldc));
            break;
        }
    }

    return;
}
