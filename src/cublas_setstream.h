#ifndef CUBLAS_SETSTREAM_H
#define CUBLAS_SETSTREAM_H

#ifdef _WIN32
   #define DLL_EXPORT __declspec(dllexport)
#else
   #define DLL_EXPORT
#endif


extern "C" {

  void DLL_EXPORT cublas_setstream(cublasHandle_t *handle,
                                   cudaStream_t *stream);
}


#endif /* ifndef CUBLAS_SETSTREAM_H */
