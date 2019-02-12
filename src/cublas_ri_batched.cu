#include <cuda.h>
#include <cublas_v2.h>
#include "cu_errchk.h"
#include "cublas_ri_batched.h"



void cublas_ri_batched(cublasHandle_t *handle,
                       int n,
                       const void *d_A[],
                       int *PivotArray,
                       void *d_C[],
                       int *infoArray,
                       int batchSize,
                       int dtype)
{
    switch(dtype) {

        case 0:
        {
            gpuBlasErrchk(cublasSgetriBatched(*handle, n,
                                              reinterpret_cast<const float**>(d_A), n,
                                              PivotArray,
                                              reinterpret_cast<float**>(d_C), n,
                                              infoArray,
                                              batchSize));
            break;
        }

        case 1:
        {
            gpuBlasErrchk(cublasDgetriBatched(*handle, n,
                                              reinterpret_cast<const double**>(d_A), n,
                                              PivotArray,
                                              reinterpret_cast<double**>(d_C), n,
                                              infoArray,
                                              batchSize));
            break;
        }
        
        case 2:
        {
            gpuBlasErrchk(cublasCgetriBatched(*handle, n,
                                              reinterpret_cast<const float2**>(d_A), n,
                                              PivotArray,
                                              reinterpret_cast<float2**>(d_C), n,
                                              infoArray,
                                              batchSize));
            break;
        }
        
        case 3:
        {
            gpuBlasErrchk(cublasZgetriBatched(*handle, n,
                                              reinterpret_cast<const double2**>(d_A), n,
                                              PivotArray,
                                              reinterpret_cast<double2**>(d_C), n,
                                              infoArray,
                                              batchSize));
            break;
        }
    }

    return;

}
