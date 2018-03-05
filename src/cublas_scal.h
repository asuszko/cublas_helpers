#ifndef CUBLAS_SCAL_H
#define CUBLAS_SCAL_H

#ifdef _WIN32
   #define DLL_EXPORT __declspec(dllexport)
#else
   #define DLL_EXPORT
#endif

extern "C" {

  void DLL_EXPORT cublas_scal(cublasHandle_t *handle,
                              int n,
                              void *alpha,
                              void *d_x, int incx,
                              int dtype);
}

template<typename T>
inline cublasStatus_t cublasTscal(cublasHandle_t *handle,
                                  int n,
                                  const T *alpha,
                                  T *x, int incx);



#endif /* ifndef CUBLAS_SCAL_H */
