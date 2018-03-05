
#include <cuda.h>
#include <cublas_v2.h>
#include "cu_errchk.h"
#include "cublas_init.h"


/**
 *  Initialize a cuBLAS library context.
 *  @return handle - [cublasHandle_t*] - cuBLAS handle
 */
cublasHandle_t *cublas_init()
{
    /* Create cuBLAS handle. */
    cublasHandle_t *handle = (cublasHandle_t*)malloc(sizeof(cublasHandle_t));

    /* Initialize cuBLAS library context. */
    gpuBlasErrchk(cublasCreate(handle));

    /* Return pointer to the handle. */
    return handle;
}


void cublas_destroy(cublasHandle_t *handle)
{
    gpuBlasErrchk(cublasDestroy(*handle))
    return;
}
