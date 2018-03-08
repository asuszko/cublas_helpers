#include <cuda.h>
#include <cublas_v2.h>
#include "cu_errchk.h"
#include "cublas_nrm2.h"


template<typename T>
inline cublasStatus_t cublasTnrm2(cublasHandle_t *handle,
                                  int n,
                                  const T *x, int incx,
                                  void *result)
{
    if (std::is_same<T, float>::value) {
        return cublasSnrm2(*handle, n,
                          (float *)x, incx,
                          (float *)result);
    }
    else
    if (std::is_same<T, double>::value) {
        return cublasDnrm2(*handle, n,
                          (double *)x, incx,
                          (double *)result);
    }
    else
    if (std::is_same<T, cuComplex>::value) {
        return cublasScnrm2(*handle, n,
                           (cuComplex *)x, incx,
                           (float *)result);
    }
    else
    if (std::is_same<T, cuDoubleComplex>::value) {
        return cublasDznrm2(*handle, n,
                           (cuDoubleComplex *)x, incx,
                           (double *)result);
    }
    else {
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }
}



/**
*  http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-nrm2
*/
void cublas_nrm2(cublasHandle_t *handle,
                 int n,
                 const void *x, int incx,
                 void *result,
                 int dtype)
{

    switch(dtype) {
        case 0: {
            gpuBlasErrchk(cublasTnrm2(handle,
                                      n,
                                      (float *)x, incx,
                                      (float *)result));
            break;
        }
        case 1: {
            gpuBlasErrchk(cublasTnrm2(handle,
                                      n,
                                      (double *)x, incx,
                                      (double *)result));
            break;
        }
        case 2: {
            gpuBlasErrchk(cublasTnrm2(handle,
                                      n,
                                      (cuComplex *)x, incx,
                                      (float *)result));
            break;
        }
        case 3: {
            gpuBlasErrchk(cublasTnrm2(handle,
                                      n,
                                      (cuDoubleComplex *)x, incx,
                                      (double *)result));
            break;
        }
    }

    return;
}
