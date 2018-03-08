#include <cuda.h>
#include <cublas_v2.h>
#include "cu_errchk.h"
#include "cublas_dgmm.h"


template<typename T>
inline cublasStatus_t cublasTdgmm(cublasHandle_t *handle,
                                  cublasSideMode_t mode,
                                  int m, int n,
                                  const T *A, int lda,
                                  const T *x, int incx,
                                  T *C, int ldc)
{
    if (std::is_same<T, float>::value) {
        return cublasSdgmm(*handle, mode, m, n,
                          (float *)A, lda,
                          (float *)x, incx,
                          (float *)C, ldc);
    }
    else
    if (std::is_same<T, double>::value) {
        return cublasDdgmm(*handle, mode, m, n,
                          (double *)A, lda,
                          (double *)x, incx,
                          (double *)C, ldc);
    }
    else
    if (std::is_same<T, cuComplex>::value) {
        return cublasCdgmm(*handle, mode, m, n,
                          (cuComplex *)A, lda,
                          (cuComplex *)x, incx,
                          (cuComplex *)C, ldc);
    }
    else
    if (std::is_same<T, cuDoubleComplex>::value) {
        return cublasZdgmm(*handle, mode, m, n,
                          (cuDoubleComplex *)A, lda,
                          (cuDoubleComplex *)x, incx,
                          (cuDoubleComplex *)C, ldc);
    }
    else {
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }
}


/**
*  http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-dgmm
*/
void cublas_dgmm(cublasHandle_t *handle,
                 cublasSideMode_t mode,
                 int m, int n,
                 const void *d_A, int lda,
                 const void *d_x, int incx,
                 void *d_C, int ldc,
                 int dtype)
{

    switch(dtype) {

        case 0: {
            gpuBlasErrchk(cublasTdgmm(handle,
                                      mode,
                                      m, n,
                                      (float*)d_A, lda,
                                      (float*)d_x, incx,
                                      (float*)d_C, ldc));
            break;
        }

        case 1: {
            gpuBlasErrchk(cublasTdgmm(handle,
                                      mode,
                                      m, n,
                                      (double*)d_A, lda,
                                      (double*)d_x, incx,
                                      (double*)d_C, ldc));
            break;
        }

        case 2: {
            gpuBlasErrchk(cublasTdgmm(handle,
                                      mode,
                                      m, n,
                                      (cuComplex*)d_A, lda,
                                      (cuComplex*)d_x, incx,
                                      (cuComplex*)d_C, ldc));
            break;
        }

        case 3: {
            gpuBlasErrchk(cublasTdgmm(handle,
                                      mode,
                                      m, n,
                                      (cuDoubleComplex*)d_A, lda,
                                      (cuDoubleComplex*)d_x, incx,
                                      (cuDoubleComplex*)d_C, ldc));
            break;
        }
    }

    return;
}
