#ifndef CUBLAS_COPY_H
#define CUBLAS_COPY_H

#ifdef _WIN32
   #define DLL_EXPORT __declspec(dllexport)
#else
   #define DLL_EXPORT
#endif


extern "C" {

  void DLL_EXPORT cublas_copy(cublasHandle_t *handle,
                              int n,
                              void *x, int incx,
                              void *y, int incy,
                              int dtype);
}


#endif /* ifndef CUBLAS_COPY_H */
