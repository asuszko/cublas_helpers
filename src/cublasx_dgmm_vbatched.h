#ifndef CUBLASX_DGMM_VBATCHED_H
#define CUBLASX_DGMM_VBATCHED_H

#ifdef _WIN32
   #define DLL_EXPORT __declspec(dllexport)
#else
   #define DLL_EXPORT
#endif


extern "C" {

    void DLL_EXPORT cublasx_dgmm_vbatched(void *d_A,
                                          void *d_x,
                                          cublasSideMode_t mode,
                                          int m, int n,
                                          int batch_size,
                                          int dtype);
}


#endif /* ifndef CUBLASX_DGMM_VBATCHED_H */
