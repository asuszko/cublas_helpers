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
                                void *x, int incx,
                                void *result,
                                int dtype);
}


#endif /* ifndef CUBLAS_NRM2_H */
