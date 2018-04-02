#include <cuda.h>
#include <cublas_v2.h>
#include "cu_errchk.h"
#include "cublas_dgmm.h"


/**
*  http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-dgmm
*/
void cublas_dgmm(cublasHandle_t *handle,
                 cublasSideMode_t mode,
                 int m, int n,
                 void *d_A, int lda,
                 void *d_x, int incx,
                 void *d_C, int ldc,
                 int dtype)
{

    switch(dtype) {

        case 0:
            gpuBlasErrchk(cublasSdgmm(*handle, mode, m, n,
                                      static_cast<float*>(d_A), lda,
                                      static_cast<float*>(d_x), incx,
                                      static_cast<float*>(d_C), ldc));
            break;

        case 1:
            gpuBlasErrchk(cublasDdgmm(*handle, mode, m, n,
                                      static_cast<double*>(d_A), lda,
                                      static_cast<double*>(d_x), incx,
                                      static_cast<double*>(d_C), ldc));
            break;

        case 2:
            gpuBlasErrchk(cublasCdgmm(*handle, mode, m, n,
                                      static_cast<float2*>(d_A), lda,
                                      static_cast<float2*>(d_x), incx,
                                      static_cast<float2*>(d_C), ldc));
            break;

        case 3:
            gpuBlasErrchk(cublasZdgmm(*handle, mode, m, n,
                                      static_cast<double2*>(d_A), lda,
                                      static_cast<double2*>(d_x), incx,
                                      static_cast<double2*>(d_C), ldc));
            break;

    }

    return;
}
