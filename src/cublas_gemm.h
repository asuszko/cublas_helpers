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
      int dtype,
      bool m3m=false);
               
               
    void DLL_EXPORT cublas_gemm_strided_batched(cublasHandle_t *handle,
      cublasOperation_t transa,
      cublasOperation_t transb,
      int m, int n, int k,
      const void *alpha,
      const void *d_A, int lda, uint64_t strideA,
      const void *d_B, int ldb, uint64_t strideB,
      const void *beta,
      void *d_C, int ldc, uint64_t strideC,
      int batchcount,
      int dtype,
      bool m3m=false);

}


#endif /* ifndef CUBLAS_GEMM_H */
