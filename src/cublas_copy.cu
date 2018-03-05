#include <cuda.h>
#include <cublas_v2.h>
#include "cu_errchk.h"
#include "cublas_copy.h"


template<typename T>
inline cublasStatus_t cublasTcopy(cublasHandle_t *handle,
                                  int n,
                                  const T *x, int incx,
                                  T *y, int incy)
{
    if (std::is_same<T, float>::value) {
        return cublasScopy(*handle, n,
                          (float *)x, incx,
                          (float *)y, incy);
    }
    else
    if (std::is_same<T, double>::value) {
        return cublasDcopy(*handle, n,
                          (double *)x, incx,
                          (double *)y, incy);
    }
    else
    if (std::is_same<T, cuComplex>::value) {
        return cublasCcopy(*handle, n,
                          (cuComplex *)x, incx,
                          (cuComplex *)y, incy);
    }
    else
    if (std::is_same<T, cuDoubleComplex>::value) {
        return cublasZcopy(*handle, n,
                          (cuDoubleComplex *)x, incx,
                          (cuDoubleComplex *)y, incy);
    }
    else {
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }
}



/* C compatible version that requires a dtype_id to be converted
to the proper data type. */
void cublas_copy(cublasHandle_t *handle,
                 int n,
                 const void *x, int incx,
                 void *y, int incy,
                 int dtype)
{

    switch(dtype) {
        case 0: {
            gpuBlasErrchk(cublasTcopy(handle,
                                      n,
                                      (float*)x, incx,
                                      (float*)y, incy));
            break;
        }
        case 1: {
            gpuBlasErrchk(cublasTcopy(handle,
                                      n,
                                      (double*)x, incx,
                                      (double*)y, incy));
            break;
        }
        case 2: {
            gpuBlasErrchk(cublasTcopy(handle,
                                      n,
                                      (cuComplex*)x, incx,
                                      (cuComplex*)y, incy));
             break;
        }
        case 3: {
            gpuBlasErrchk(cublasTcopy(handle,
                                      n,
                                      (cuDoubleComplex*)x, incx,
                                      (cuDoubleComplex*)y, incy));
            break;
        }
    }

    return;
}
