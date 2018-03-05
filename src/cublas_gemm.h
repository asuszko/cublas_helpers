#ifndef CUBLAS_GEMM_H
#define CUBLAS_GEMM_H

#ifdef _WIN32
   #define DLL_EXPORT __declspec(dllexport)
#else
   #define DLL_EXPORT
#endif

extern "C" {

  void DLL_EXPORT cublas_gemm(cublasHandle_t *handle,
                              cublasOperation_t transa,
                              cublasOperation_t transb,
                              int m, int n, int k,
                              const void *alpha,
                              const void *d_A, int lda,
                              const void *d_B, int ldb,
                              const void *beta,
                              void *d_C, int ldc,
                              int dtype);
}


template<typename T>
inline cublasStatus_t cublasTgemm(cublasHandle_t *handle,
                                  cublasOperation_t transa, cublasOperation_t transb,
                                  int m, int n, int k,
                                  const T *alpha,
                                  const T *A, int lda,
                                  const T *B, int ldb,
                                  const T *beta,
                                  T *C, int ldc);


#endif /* ifndef CUBLAS_GEMM_H */
