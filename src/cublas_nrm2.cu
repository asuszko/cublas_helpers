#include <cuda.h>
#include <cublas_v2.h>
#include "cu_errchk.h"
#include "cublas_nrm2.h"


/**
*  http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-nrm2
*/
void cublas_nrm2(cublasHandle_t *handle,
                 int n,
                 void *x, int incx,
                 void *result,
                 int dtype)
{

    switch(dtype) {
        case 0:
            gpuBlasErrchk(cublasSnrm2(*handle, n,
                                      static_cast<float*>(x), incx,
                                      static_cast<float*>(result)));
            break;

        case 1:
            gpuBlasErrchk(cublasDnrm2(*handle, n,
                                      static_cast<double*>(x), incx,
                                      static_cast<double*>(result)));
            break;

        case 2:
            gpuBlasErrchk(cublasScnrm2(*handle, n,
                                      static_cast<float2*>(x), incx,
                                      static_cast<float*>(result)));
            break;
            
        case 3:
            gpuBlasErrchk(cublasDznrm2(*handle, n,
                                      static_cast<double2*>(x), incx,
                                      static_cast<double*>(result)));
            break;
    }

    return;
}
