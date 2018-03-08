#ifndef CUBLAS_INIT_H
#define CUBLAS_INIT_H

#ifdef _WIN32
   #define DLL_EXPORT __declspec(dllexport)
#else
   #define DLL_EXPORT
#endif


extern "C" {

    cublasHandle_t DLL_EXPORT *cublas_init();

    void DLL_EXPORT cublas_destroy(cublasHandle_t *handle);

}


#endif /* ifndef CUBLAS_INIT_H */
