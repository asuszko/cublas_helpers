# -*- coding: utf-8 -*-

__all__ = [
    "cublas",
]

from functools import reduce
from operator import mul
import numpy as np

# Local imports
from cublas_import import (cublasx_dgmm_batched,
                           cublasx_diag)

# Datatype identifier. cuBLAS currently supports these.
_blas_types = {np.dtype('f4'):0,
               np.dtype('f8'):1,
               np.dtype('c8'):2,
               np.dtype('c16'):3}


def check_vectors(x, y):
    try:
        if x.dtype != y.dtype:
            raise TypeError('Dtype mismatch between vectors x and y in axpy.')
    except (TypeError):
        exit('Could not complete request.')


class cublasx(object):

    def __init__(self, stream=None):
        """
        Initialize a blas handle, and tie the stream to it if 
        one is passed in.
        
        Parameters
        ----------
        stream : c_void_p (cudastream_t*), optional
            CUDA stream to initialize the blas handle to.
            
        Attributes
        ----------
        blas_handle : c_void_p (cublasHandle_t *)
            Pointer reference to cuBLAS handle.
        """
        self._stream = stream


    def dgmm_batched(self, a, x, mode='R'):
        """
        cuBLAS-like extension function dgmm.
        
        Parameters
        ---------- 
        a : Device_Ptr object
            Device pointer object with dev_ptr to input matrix a. A 
            should have shape batch_size, nrows, ncols.
            
        x : Device_Ptr object
            Device pointer object with dev_ptr to input vector x.
            The length of vector x should equal the number of 
            rows of matrix a.
        """
        check_vectors(a,x)
        
        # Make sure vector and matrix sizes are valid
        if mode == 'L':
            assert len(x) == a.shape[1]
        if mode == 'R':
            assert len(x) == a.shape[2]
            
        cublasx_dgmm_batched(a.ptr, x.ptr,
                             {'L':0, 'R':1}[mode],
                             a.shape[1], a.shape[2], a.shape[0],
                             _blas_types[a.dtype])


    @property
    def stream(self):
        return self._stream
        
                      
    def __enter__(self):
        return self


    def __exit__(self, *args, **kwargs):
        cublas_destroy(self.handle)