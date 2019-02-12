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
                                void *d_A, int lda,
                                void *d_x, int incx,
                                void *d_C, int ldc,
                                int dtype);
}


#endif /* ifndef CUBLAS_DGMM_H */
