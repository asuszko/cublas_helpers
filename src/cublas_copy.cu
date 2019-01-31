#include <cuda.h>
#include <cublas_v2.h>
#include "cu_errchk.h"
#include "cublas_copy.h"



/**
*  http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-copy
*/
void cublas_copy(cublasHandle_t *handle,
                 int n,
                 void *x, int incx,
                 void *y, int incy,
                 int dtype)
{

    switch(dtype) {
        case 0:
            gpuBlasErrchk(cublasScopy(*handle, n,
                                      static_cast<float*>(x), incx,
                                      static_cast<float*>(y), incy));
            break;

        case 1:
            gpuBlasErrchk(cublasDcopy(*handle, n,
                                      static_cast<double*>(x), incx,
                                      static_cast<double*>(y), incy));
            break;

        case 2:
            gpuBlasErrchk(cublasCcopy(*handle, n,
                                      static_cast<float2*>(x), incx,
                                      static_cast<float2*>(y), incy));
             break;

        case 3:
            gpuBlasErrchk(cublasZcopy(*handle, n,
                                      static_cast<double2*>(x), incx,
                                      static_cast<double2*>(y), incy));
            break;
    }

    return;
}
