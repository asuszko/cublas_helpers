#ifndef CUBLAS_DGMM_H
#define CUBLAS_DGMM_H

#ifdef _WIN32
   #define DLL_EXPORT __declspec(dllexport)
#else
   #define DLL_EXPORT
#endif


extern "C" {

  void DLL_EXPORT cublas_dgmm(cublasHandle_t *handle,
                              cublasSideMode_t mode,
                              int m, int n,
                              const void *d_A, int lda,
                              const void *d_x, int incx,
                              void *d_C, int ldc,
                              int dtype);
}


template<typename T>
inline cublasStatus_t cublasTdgmm(cublasHandle_t *handle,
                                  cublasSideMode_t mode,
                                  int m, int n,
                                  const T *A, int lda,
                                  const T *x, int incx,
                                  T *C, int ldc);


#endif /* ifndef CUBLAS_DGMM_H */
