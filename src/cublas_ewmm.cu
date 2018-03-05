#include <cuda.h>
#include <cublas_v2.h>
#include "cu_errchk.h"
#include "cublas_ewmm.h"

#define IDX3D(i,j,k,nx,ny) (((k)*(nx)*(ny))+((j)*(nx))+(i))


template<typename T>
__global__ void cublasRmm(T *x, T *y,
                          int nx, int ny, int nz)
{
    int ix = threadIdx.x + blockDim.x*blockIdx.x;
    int iy = threadIdx.y + blockDim.y*blockIdx.y;
    int iz = threadIdx.z + blockDim.z*blockIdx.z;

    if(ix < nx && iy < ny && iz < nz) {
        y[IDX3D(ix,iy,iz,nx,ny)] *= x[IDX3D(ix,iy,iz,nx,ny)];
    }
}


__global__ void cublasCmm(cuComplex *x, cuComplex *y,
                          int nx, int ny, int nz)
{
    int ix = threadIdx.x + blockDim.x*blockIdx.x;
    int iy = threadIdx.y + blockDim.y*blockIdx.y;
    int iz = threadIdx.z + blockDim.z*blockIdx.z;

    if(ix < nx && iy < ny && iz < nz) {
        float a = x[IDX3D(ix,iy,iz,nx,ny)].x;
        float b = x[IDX3D(ix,iy,iz,nx,ny)].y;
        float c = y[IDX3D(ix,iy,iz,nx,ny)].x;
        float d = y[IDX3D(ix,iy,iz,nx,ny)].y;

        y[IDX3D(ix,iy,iz,nx,ny)].x=(a*c-b*d);
        y[IDX3D(ix,iy,iz,nx,ny)].y=(a*d+b*c);
    }
}


__global__ void cublasZmm(cuDoubleComplex *x, cuDoubleComplex *y,
                          int nx, int ny, int nz)
{
    int ix = threadIdx.x + blockDim.x*blockIdx.x;
    int iy = threadIdx.y + blockDim.y*blockIdx.y;
    int iz = threadIdx.z + blockDim.z*blockIdx.z;

    if(ix < nx && iy < ny && iz < nz) {
        double a = x[IDX3D(ix,iy,iz,nx,ny)].x;
        double b = x[IDX3D(ix,iy,iz,nx,ny)].y;
        double c = y[IDX3D(ix,iy,iz,nx,ny)].x;
        double d = y[IDX3D(ix,iy,iz,nx,ny)].y;

        y[IDX3D(ix,iy,iz,nx,ny)].x=(a*c-b*d);
        y[IDX3D(ix,iy,iz,nx,ny)].y=(a*d+b*c);
    }
}

/* C compatible version that requires a dtype_id to be converted
to the proper data type. */
void cublas_ewmm(const void *d_x,
                 void *d_y,
                 dim3 dims,
                 int dtype,
                 cudaStream_t *stream)
{
    int nx = dims.x;
    int ny = dims.y;
    int nz = dims.z;

    dim3 blockSize;
    (nz <= 16) ? blockSize.z = nz : blockSize.z = 16;
    (ny <= 16) ? blockSize.y = ny : blockSize.y = 16;
    (nx <= 16) ? blockSize.x = nx : blockSize.x = 16;

    while (blockSize.x*blockSize.y*blockSize.z > 1024) {
        blockSize.z /= 2;
    }

    dim3 gridSize((nx-1)/blockSize.x+1,
                  (ny-1)/blockSize.y+1,
                  (nz-1)/blockSize.z+1);

    cudaStream_t stream_id;
    (stream == NULL) ? stream_id = NULL : stream_id = *stream;

    switch(dtype) {

        case 0: {
            cublasRmm<<<gridSize,blockSize,0,stream_id>>>((float *)d_x,
                                                          (float *)d_y,
                                                          nx, ny, nz);
            break;
        }

        case 1: {
            cublasRmm<<<gridSize,blockSize,0,stream_id>>>((double *)d_x,
                                                          (double *)d_y,
                                                          nx, ny, nz);
            break;
        }

        case 2: {
            cublasCmm<<<gridSize,blockSize,0,stream_id>>>((cuComplex *)d_x,
                                                          (cuComplex *)d_y,
                                                          nx, ny, nz);
            break;
        }

        case 3: {
            cublasZmm<<<gridSize,blockSize,0,stream_id>>>((cuDoubleComplex *)d_x,
                                                          (cuDoubleComplex *)d_y,
                                                          nx, ny, nz);
            break;
        }
    }

    return;
}
