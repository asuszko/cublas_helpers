#ifndef CUBLAS_EWMM_H
#define CUBLAS_EWMM_H

#ifdef _WIN32
   #define DLL_EXPORT __declspec(dllexport)
#else
   #define DLL_EXPORT
#endif

extern "C" {

  void DLL_EXPORT cublas_ewmm(const void *d_x,
                              void *d_y,
                              dim3 dims,
                              int dtype,
                              cudaStream_t *stream=NULL);
}



#endif /* ifndef CUBLAS_EWMM_H */
