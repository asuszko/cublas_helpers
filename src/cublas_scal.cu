#include <cuda.h>
#include <cublas_v2.h>
#include "cu_errchk.h"
#include "cublas_scal.h"


/**
*  http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-scal
*/
void cublas_scal(cublasHandle_t *handle,
                 int n,
                 void *alpha,
                 void *d_x, int incx,
                 int dtype)
{
    switch(dtype) {

        case 0:
            gpuBlasErrchk(cublasSscal(*handle, n,
                                      static_cast<float*>(alpha),
                                      static_cast<float*>(d_x), incx));
            break;

        case 1:
            gpuBlasErrchk(cublasDscal(*handle, n,
                                      static_cast<double*>(alpha),
                                      static_cast<double*>(d_x), incx));
            break;

        case 2:
            gpuBlasErrchk(cublasCscal(*handle, n,
                                      static_cast<float2*>(alpha),
                                      static_cast<float2*>(d_x), incx));
            break;

        case 3:
            gpuBlasErrchk(cublasZscal(*handle, n,
                                      static_cast<double2*>(alpha),
                                      static_cast<double2*>(d_x), incx));
            break;
    }

    return;

}
