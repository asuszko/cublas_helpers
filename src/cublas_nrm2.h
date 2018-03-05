#ifndef CUBLAS_NRM2_H
#define CUBLAS_NRM2_H

#ifdef _WIN32
   #define DLL_EXPORT __declspec(dllexport)
#else
   #define DLL_EXPORT
#endif

extern "C" {

  void DLL_EXPORT cublas_nrm2(cublasHandle_t *handle,
                              int n,
                              const void *x, int incx,
                              void *result,
                              int dtype);
}


template<typename T>
inline cublasStatus_t cublasTnrm2(cublasHandle_t *handle,
                                  int n,
                                  const T *x, int incx,
                                  void *result);

#endif /* ifndef CUBLAS_NRM2_H */
