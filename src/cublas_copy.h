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
                              const void *x, int incx,
                              void *y, int incy,
                              int dtype);
}


template<typename T>
inline cublasStatus_t cublasTcopy(cublasHandle_t *handle,
                                  int n,
                                  const T *x, int incx,
                                  T *y, int incy);


#endif /* ifndef CUBLAS_COPY_H */
