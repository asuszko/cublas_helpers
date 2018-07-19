#ifndef CUBLAS_RI_BATCHED_H
#define CUBLAS_RI_BATCHED_H

#ifdef _WIN32
   #define DLL_EXPORT __declspec(dllexport)
#else
   #define DLL_EXPORT
#endif


extern "C" {

    void DLL_EXPORT cublas_ri_batched(cublasHandle_t *handle,
                                      int n,
                                      void *d_A[],
                                      int *PivotArray,
                                      void *d_C[],
                                      int *infoArray,
                                      int batchSize,
                                      int dtype);
}


#endif /* ifndef CUBLAS_RI_BATCHED_H */
