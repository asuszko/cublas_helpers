#ifndef CUBLAS_AXPY_H
#define CUBLAS_AXPY_H

#ifdef _WIN32
   #define DLL_EXPORT __declspec(dllexport)
#else
   #define DLL_EXPORT
#endif


extern "C" {

    void DLL_EXPORT cublas_axpy(cublasHandle_t *handle,
                                int n,
                                void *alpha,
                                void *x, int incx,
                                void *y, int incy,
                                int dtype);
}


#endif /* ifndef CUBLAS_AXPY_H */
