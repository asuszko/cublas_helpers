#ifndef CUBLAS_RF_BATCHED_H
#define CUBLAS_RF_BATCHED_H

#ifdef _WIN32
   #define DLL_EXPORT __declspec(dllexport)
#else
   #define DLL_EXPORT
#endif


extern "C" {

    void DLL_EXPORT cublas_rf_batched(cublasHandle_t *handle,
                                      int n,
                                      void *Aarray[],
                                      int *PivotArray,
                                      int *infoArray,
                                      int batchSize,
                                      int dtype);
}


#endif /* ifndef CUBLAS_RF_BATCHED_H */
