#include <cuda.h>
#include <cublas_v2.h>
#include "cu_errchk.h"
#include "cublas_rf_batched.h"



void cublas_rf_batched(cublasHandle_t *handle,
                       int n,
                       void *Aarray[],
                       int *PivotArray,
                       int *infoArray,
                       int batchSize,
                       int dtype)
{
    switch(dtype) {

        case 0:
        {
            // float *ASptr = static_cast<float*>(*Aarray);
            gpuBlasErrchk(cublasSgetrfBatched(*handle, n, 
                                              (float**)Aarray, n, 
                                              PivotArray,
                                              infoArray,
                                              batchSize));
            break;
        }

        case 1:
        {
            // double *ADptr = static_cast<double*>(*Aarray);
            gpuBlasErrchk(cublasDgetrfBatched(*handle, n, 
                                              (double**)Aarray, n, 
                                              PivotArray,
                                              infoArray,
                                              batchSize));
            break;
        }
        
        case 2:
        {
            //float2 *ACptr = static_cast<float2*>(*Aarray);
            gpuBlasErrchk(cublasCgetrfBatched(*handle, n, 
                                              (float2**)Aarray, n, 
                                              PivotArray,
                                              infoArray,
                                              batchSize));
            break;
        }
        
        case 3:
        {
            // double2 *AZptr = static_cast<double2*>(*Aarray);
            gpuBlasErrchk(cublasZgetrfBatched(*handle, n, 
                                              (double2**)Aarray, n, 
                                              PivotArray,
                                              infoArray,
                                              batchSize));
            break;
        }
    }

    return;

}
