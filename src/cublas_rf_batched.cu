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
            gpuBlasErrchk(cublasSgetrfBatched(*handle, n, 
                                              reinterpret_cast<float**>(Aarray), n, 
                                              PivotArray,
                                              infoArray,
                                              batchSize));
            break;
        }

        case 1:
        {
            gpuBlasErrchk(cublasDgetrfBatched(*handle, n, 
                                              reinterpret_cast<double**>(Aarray), n, 
                                              PivotArray,
                                              infoArray,
                                              batchSize));
            break;
        }
        
        case 2:
        {
            gpuBlasErrchk(cublasCgetrfBatched(*handle, n, 
                                              reinterpret_cast<float2**>(Aarray), n, 
                                              PivotArray,
                                              infoArray,
                                              batchSize));
            break;
        }
        
        case 3:
        {
            gpuBlasErrchk(cublasZgetrfBatched(*handle, n, 
                                              reinterpret_cast<double2**>(Aarray), n, 
                                              PivotArray,
                                              infoArray,
                                              batchSize));
            break;
        }
    }

    return;

}
