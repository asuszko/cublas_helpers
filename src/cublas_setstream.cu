#include <cuda.h>
#include <cufft.h>
#include "cu_errchk.h"
#include "cublas_setstream.h"

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
