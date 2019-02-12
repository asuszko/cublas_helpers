#include <cuda.h>
#include <cublas_v2.h>
#include "cu_errchk.h"
#include "cublasx_dgmm_vbatched.h"

#define BLOCKSIZE 128
const int bs = 128;


template <typename T>
__global__ void dgmmRbatched_kernel_L(T *data, const T* __restrict__ vec,
                                      int nrows, int ncols, int batchsize)
{
    unsigned long long iy = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = gridDim.x * blockDim.x;

    #pragma unroll bs
    for(; iy < nrows; iy += stride) {
        for(int j=0; j < batchsize; ++j) {
            for(int i=0; i < ncols; ++i) {
                data[j*ncols*nrows+iy*ncols+i] *= vec[iy];
            }
        }
    }
}


template <typename T>
__global__ void dgmmRbatched_kernel_R(T *data, const T* __restrict__ vec,
                                      int nrows, int ncols, int batchsize)
{
    unsigned long long iy = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = gridDim.x * blockDim.x;

    #pragma unroll bs
    for(; iy < nrows; iy += stride) {
        for(int j=0; j < batchsize; ++j) {
            for(int i=0; i < ncols; ++i) {
                data[j*ncols*nrows+iy*ncols+i] *= vec[i];
            }
        }
    }
}


template <typename T>
__global__ void dgmmCbatched_kernel_L(T *data, const T* __restrict__ vec,
                                      int nrows, int ncols, int batchsize)
{
    unsigned long long iy = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = gridDim.x * blockDim.x;

    #pragma unroll bs
    for(; iy < nrows; iy += stride) {
        for(int j=0; j < batchsize; ++j) {
            for(int i=0; i < ncols; ++i) {
                int index = j*ncols*nrows+iy*ncols+i;
                T valy = data[index];
                T valx = vec[iy];

                data[index].x = valy.x*valx.x-valy.y*valx.y;
                data[index].y = valy.x*valx.y+valy.y*valx.x;
            }
        }
    }
}


template <typename T>
__global__ void dgmmCbatched_kernel_R(T *data, const T* __restrict__ vec,
                                      int nrows, int ncols, int batchsize)
{
    unsigned long long iy = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = gridDim.x * blockDim.x;

    #pragma unroll bs
    for(; iy < nrows; iy += stride) {
        for(int j=0; j < batchsize; ++j) {
            for(int i=0; i < ncols; ++i) {
                int index = j*ncols*nrows+iy*ncols+i;
                T valy = data[index];
                T valx = vec[i];

                data[index].x = valy.x*valx.x-valy.y*valx.y;
                data[index].y = valy.x*valx.y+valy.y*valx.x;
            }
        }
    }
}

/**
*  http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-dgmm
* This is a custom written batched version.
*/
void cublasx_dgmm_vbatched(void *d_A,
                           void *d_x,
                           cublasSideMode_t mode,
                           int m, int n,
                           int batch_size,
                           int dtype)
{
    dim3 blockSize(bs);
    int threads;
    if(mode == CUBLAS_SIDE_LEFT) {
        threads = m;
    }
    if(mode == CUBLAS_SIDE_RIGHT) {
        threads = n;
    }
    dim3 gridSize((((threads-1)/blockSize.x+1)-1)/blockSize.x+1);

    switch(dtype) {

        case 0:
        {
            if(mode == CUBLAS_SIDE_LEFT) {
                dgmmRbatched_kernel_L<<<gridSize,blockSize>>>(reinterpret_cast<float*>(d_A),
                                                              static_cast<const float*>(d_x),
                                                              m, n, batch_size);
            }
            if(mode == CUBLAS_SIDE_RIGHT) {
                dgmmRbatched_kernel_R<<<gridSize,blockSize>>>(reinterpret_cast<float*>(d_A),
                                                              static_cast<const float*>(d_x),
                                                              m, n, batch_size);
            }
            break;
        }

        case 1:
        {
            if(mode == CUBLAS_SIDE_LEFT) {
                dgmmRbatched_kernel_L<<<gridSize,blockSize>>>(reinterpret_cast<double*>(d_A),
                                                              static_cast<const double*>(d_x),
                                                              m, n, batch_size);
            }
            if(mode == CUBLAS_SIDE_RIGHT) {
                dgmmRbatched_kernel_R<<<gridSize,blockSize>>>(reinterpret_cast<double*>(d_A),
                                                              static_cast<const double*>(d_x),
                                                              m, n, batch_size);
            }
            break;
        }

        case 2:
        {
            if(mode == CUBLAS_SIDE_LEFT) {
                dgmmCbatched_kernel_L<<<gridSize,blockSize>>>(reinterpret_cast<float2*>(d_A),
                                                              static_cast<const float2*>(d_x),
                                                              m, n, batch_size);
            }
            if(mode == CUBLAS_SIDE_RIGHT) {
                dgmmCbatched_kernel_R<<<gridSize,blockSize>>>(reinterpret_cast<float2*>(d_A),
                                                              static_cast<const float2*>(d_x),
                                                              m, n, batch_size);
            }
            break;
        }

        case 3:
        {
            if(mode == CUBLAS_SIDE_LEFT) {
                dgmmCbatched_kernel_L<<<gridSize,blockSize>>>(reinterpret_cast<double2*>(d_A),
                                                              static_cast<const double2*>(d_x),
                                                              m, n, batch_size);
            }
            if(mode == CUBLAS_SIDE_RIGHT) {
                dgmmCbatched_kernel_R<<<gridSize,blockSize>>>(reinterpret_cast<double2*>(d_A),
                                                              static_cast<const double2*>(d_x),
                                                              m, n, batch_size);
            }
            break;
        }
    }

    return;
}
