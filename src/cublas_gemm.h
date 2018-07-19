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
                                void *alpha,
                                void *d_A, int lda,
                                void *d_B, int ldb,
                                void *beta,
                                void *d_C, int ldc,
                                int dtype);
               
               
    void DLL_EXPORT cublas_gemm_batched(cublasHandle_t *handle,
                                        cublasOperation_t transa,
                                        cublasOperation_t transb,
                                        int m, int n, int k,
                                        void *alpha,
                                        void *d_A[], int lda,
                                        void *d_B[], int ldb,
                                        void *beta,
                                        void *d_C[], int ldc,
                                        int batchcount,
                                        int dtype);
                                        
    void DLL_EXPORT cublas_gemm_strided_batched(cublasHandle_t *handle,
                                                cublasOperation_t transa,
                                                cublasOperation_t transb,
                                                int m, int n, int k,
                                                void *alpha,
                                                void *d_A, int lda, long long int strideA,
                                                void *d_B, int ldb, long long int strideB,
                                                void *beta,
                                                void *d_C, int ldc, long long int strideC,
                                                int batchcount,
                                                int dtype);

}


#endif /* ifndef CUBLAS_GEMM_H */
