#include <cuda.h>
#include <cufft.h>
#include "cu_errchk.h"
#include "cublas_setstream.h"

/**
*  Set the stream for the cuBLAS handle.
*  @param handle - [cublasHandle_t*] : The cuBLAS handle.
*  @param stream - [cudaStream_t*] : CUDA stream.
*/
void cublas_setstream(cublasHandle_t *handle, cudaStream_t *stream)
{
    if(stream == NULL) {
        gpuBlasErrchk(cublasSetStream(*handle,NULL));
    }
    else {
        gpuBlasErrchk(cublasSetStream(*handle,*stream));
    }
    return;
}
