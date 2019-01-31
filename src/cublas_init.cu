#include <cuda.h>
#include <cublas_v2.h>
#include "cu_errchk.h"
#include "cublas_init.h"

/**
*  Initialize cuBLAS handle.
*  @return [cublasHandle_t *] : cuBLAS handle.
*/
cublasHandle_t *cublas_init()
{
    /* Create cuBLAS handle. */
    cublasHandle_t *handle = new cublasHandle_t;

    /* Initialize cuBLAS library context. */
    gpuBlasErrchk(cublasCreate(handle));

    /* Return pointer to the handle. */
    return handle;
}

/**
*  Destroy cuBLAS handle.
*  @param [cublasHandle_t *] : cuBLAS handle.
*/
void cublas_destroy(cublasHandle_t *handle)
{
    gpuBlasErrchk(cublasDestroy(*handle))
    delete[] handle;
    return;
}
