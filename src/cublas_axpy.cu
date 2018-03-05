#include <cuda.h>
#include <cublas_v2.h>
#include "cu_errchk.h"
#include "cublas_axpy.h"


template<typename T>
inline cublasStatus_t cublasTaxpy(cublasHandle_t *handle,
        int n,
        const T *alpha,
        const T *x, int incx,
        T *y, int incy)
{
    if (std::is_same<T, float>::value) {
        return cublasSaxpy(*handle, n, (float *)alpha,
                          (float *)x, incx,
                          (float *)y, incy);
    }
    else
    if (std::is_same<T, double>::value) {
        return cublasDaxpy(*handle, n, (double *)alpha,
                          (double *)x, incx,
                          (double *)y, incy);
    }
    else
    if (std::is_same<T, cuComplex>::value) {
        return cublasCaxpy(*handle, n, (cuComplex *)alpha,
                          (cuComplex *)x, incx,
                          (cuComplex *)y, incy);
    }
    else
    if (std::is_same<T, cuDoubleComplex>::value) {
        return cublasZaxpy(*handle, n, (cuDoubleComplex *)alpha,
                          (cuDoubleComplex *)x, incx,
                          (cuDoubleComplex *)y, incy);
    }
    else {
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }
}



/* C compatible version that requires a dtype_id to be converted
to the proper data type. */
void cublas_axpy(cublasHandle_t *handle,
                 int n,
                 const void *alpha,
                 const void *x, int incx,
                 void *y, int incy,
                 int dtype)
{

    switch(dtype) {
        case 0: {
            gpuBlasErrchk(cublasTaxpy(handle,
                                      n,
                                      (float *)alpha,
                                      (float *)x, incx,
                                      (float *)y, incy));
            break;
        }
        case 1: {
            gpuBlasErrchk(cublasTaxpy(handle,
                                      n,
                                      (double *)alpha,
                                      (double *)x, incx,
                                      (double *)y, incy));
            break;
        }
        case 2: {
            gpuBlasErrchk(cublasTaxpy(handle,
                                      n,
                                      (cuComplex *)alpha,
                                      (cuComplex *)x, incx,
                                      (cuComplex *)y, incy));
            break;
        }
        case 3: {
            gpuBlasErrchk(cublasTaxpy(handle,
                                      n,
                                      (cuDoubleComplex *)alpha,
                                      (cuDoubleComplex *)x, incx,
                                      (cuDoubleComplex *)y, incy));
            break;
        }
    }

    return;
}
