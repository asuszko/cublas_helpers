#include <cuda.h>
#include "cu_errchk.h"
#include "cublasx_diag.h"

#define BLOCKSIZE 128
const int bs = 128;

template <typename T>
__global__ void diag_kernel_R(T *data, int nrows, int ncols, int batchsize)
{
    unsigned long long iy = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = gridDim.x * blockDim.x;

    #pragma unroll bs
    for(; iy < nrows; iy += stride) {
        for(int j=0; j < batchsize; ++j) {
            for(int i=0; i < ncols; ++i) {
                if(i != iy) {
                    data[j*ncols*nrows+iy*ncols+i] = (T)0;
                }
            }
        }
    }
}


template <typename T>
__global__ void diag_kernel_C(T *data, int nrows, int ncols, int batchsize)
{
    unsigned long long iy = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = gridDim.x * blockDim.x;

    #pragma unroll bs
    for(; iy < nrows; iy += stride) {
        for(int j=0; j < batchsize; ++j) {
            for(int i=0; i < ncols; ++i) {
                if(i != iy) {
                    data[j*ncols*nrows+iy*ncols+i].x = 0.;
                    data[j*ncols*nrows+iy*ncols+i].y = 0.;
                }
            }
        }
    }
}


void cublasx_diag(void *d_ptr,
                  int m, int n,
                  int batch_size,
                  int dtype,
                  cudaStream_t *stream)
{
    dim3 blockSize(bs);
    dim3 gridSize((((m*n-1)/blockSize.x+1)-1)/blockSize.x+1);
    
    cudaStream_t stream_id;
    (stream == NULL) ? stream_id = NULL : stream_id = *stream;

    switch(dtype) {
        
        case 0:
        {
            diag_kernel_R<<<gridSize,blockSize,0,stream_id>>>(static_cast<float*>(d_ptr),
                                                              m, n, batch_size);
            break;
        }
        
        case 1:
        {
            diag_kernel_R<<<gridSize,blockSize,0,stream_id>>>(static_cast<double*>(d_ptr),
                                                              m, n, batch_size);
            break;
        }
        
        case 2:
        {
            diag_kernel_C<<<gridSize,blockSize,0,stream_id>>>(static_cast<float2*>(d_ptr),
                                                              m, n, batch_size);
            break;
        }
        
        case 3:
        {
            diag_kernel_C<<<gridSize,blockSize,0,stream_id>>>(static_cast<double2*>(d_ptr),
                                                              m, n, batch_size);
            break;
        }
        
    }

    return;
}
