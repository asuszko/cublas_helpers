#include <cuda.h>
#include <cublas_v2.h>
#include "cu_errchk.h"
#include "cublas_axpy.h"


/**
*  http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-axpy
*/
void cublas_axpy(cublasHandle_t *handle,
                 int n,
                 void *alpha,
                 void *x, int incx,
                 void *y, int incy,
                 int dtype)
{

    switch(dtype) {
        case 0:
            gpuBlasErrchk(cublasSaxpy(*handle, n,
                                      static_cast<float*>(alpha),
                                      static_cast<float*>(x), incx,
                                      static_cast<float*>(y), incy));
            break;

        case 1:
            gpuBlasErrchk(cublasDaxpy(*handle, n,
                                      static_cast<double*>(alpha),
                                      static_cast<double*>(x), incx,
                                      static_cast<double*>(y), incy));
            break;

        case 2:
            gpuBlasErrchk(cublasCaxpy(*handle, n,
                                      static_cast<float2*>(alpha),
                                      static_cast<float2*>(x), incx,
                                      static_cast<float2*>(y), incy));
            break;

        case 3:
            gpuBlasErrchk(cublasZaxpy(*handle, n,
                                      static_cast<double2*>(alpha),
                                      static_cast<double2*>(x), incx,
                                      static_cast<double2*>(y), incy));
            break;

    }

    return;
}
