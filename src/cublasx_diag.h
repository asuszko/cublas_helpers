#ifndef CUBLASX_DIAG_H
#define CUBLASX_DIAG_H

#ifdef _WIN32
   #define DLL_EXPORT __declspec(dllexport)
#else
   #define DLL_EXPORT
#endif


extern "C" {

    void DLL_EXPORT cublasx_diag(void *d_ptr,
                                 int m, int n,
                                 int batch_size,
                                 int dtype,
                                 cudaStream_t *stream);

}


#endif /* ifndef CUBLASX_DIAG_H */
