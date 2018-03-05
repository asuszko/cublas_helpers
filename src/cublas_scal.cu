#include <cuda.h>
#include <cublas_v2.h>
#include "cu_errchk.h"
#include "cublas_scal.h"


template<typename T>
inline cublasStatus_t cublasTscal(cublasHandle_t *handle,
        int n,
        const T *alpha,
        T *x, int incx)
{
    if (std::is_same<T, float>::value) {
        return cublasSscal(*handle, n,
                          (float *)alpha,
                          (float *)x, incx);
    }
    else
    if (std::is_same<T, double>::value) {
        return cublasDscal(*handle, n,
                          (double *)alpha,
                          (double *)x, incx);
    }
    else
    if (std::is_same<T, cuComplex>::value) {
        return cublasCscal(*handle, n,
                          (cuComplex *)alpha,
                          (cuComplex *)x, incx);
    }
    else
    if (std::is_same<T, cuDoubleComplex>::value) {
        return cublasZscal(*handle, n,
                          (cuDoubleComplex *)alpha,
                          (cuDoubleComplex *)x, incx);
    }
    else {
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }
}



/* C compatible version that requires a dtype_id to be converted
to the proper data type. */
void cublas_scal(cublasHandle_t *handle,
                 int n,
                 void *alpha,
                 void *d_x, int incx,
                 int dtype)
{
    switch(dtype) {

        case 0: {
            gpuBlasErrchk(cublasTscal(handle, n,
                                      (float*)alpha,
                                      (float*)d_x, incx));
            break;
        }

        case 1: {
            gpuBlasErrchk(cublasTscal(handle, n,
                                      (double*)alpha,
                                      (double*)d_x, incx));
            break;
        }

        case 2: {
            gpuBlasErrchk(cublasTscal(handle, n,
                                      (cuComplex*)alpha,
                                      (cuComplex*)d_x, incx));
            break;
        }

        case 3: {
            gpuBlasErrchk(cublasTscal(handle, n,
                                      (cuDoubleComplex*)alpha,
                                      (cuDoubleComplex*)d_x, incx));
            break;
        }
    }

    return;

}
