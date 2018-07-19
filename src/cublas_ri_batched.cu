#include <cuda.h>
#include <cublas_v2.h>
#include "cu_errchk.h"
#include "cublas_ri_batched.h"



void cublas_ri_batched(cublasHandle_t *handle,
                       int n,
                       void *d_A[],
                       int *PivotArray,
                       void *d_C[],
                       int *infoArray,
                       int batchSize,
                       int dtype)
{
    switch(dtype) {

        case 0:
        {
            // float *ASptr = static_cast<float*>(*Aarray);
            // gpuBlasErrchk(cublasSgetrfBatched(*handle, n, 
                                              // &ASptr, n, 
                                              // PivotArray,
                                              // infoArray,
                                              // batchSize));
            // break;
        }

        case 1:
        {
            // double *ADptr = static_cast<double*>(*Aarray);
            // gpuBlasErrchk(cublasDgetrfBatched(*handle, n, 
                                              // &ADptr, n, 
                                              // PivotArray,
                                              // infoArray,
                                              // batchSize));
            // break;
        }
        
        case 2:
        {
            //float2 *ACptr = static_cast<float2*>(*Aarray);
            gpuBlasErrchk(cublasCgetriBatched(*handle, n,
                                              (const float2 **)d_A, n,
                                              PivotArray,
                                              (float2 **)d_C, n,
                                              infoArray,
                                              batchSize));
            break;
        }
        
        case 3:
        {
            // double2 *AZptr = static_cast<double2*>(*Aarray);
            // gpuBlasErrchk(cublasZgetrfBatched(*handle, n, 
                                              // &AZptr, n, 
                                              // PivotArray,
                                              // infoArray,
                                              // batchSize));
            // break;
        }
    }

    return;

}
