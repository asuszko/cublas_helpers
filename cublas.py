# -*- coding: utf-8 -*-
__all__ = [
    "cublas",
]

import numpy as np

# Local imports
from cublas_import import (cublas_axpy,
                           cublas_copy,
                           cublas_destroy,
                           cublas_ewmm,
                           cublas_init,
                           cublas_nrm2,
                           cublas_scal,
                           cublas_setstream)

# Datatype identifier. cuBLAS currently supports these.
blas_types = {0:np.dtype('f4'),
              1:np.dtype('f8'),
              2:np.dtype('c8'),
              3:np.dtype('c16')}


class cublas(object):

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
        self._blas_handle = cublas_init()
        if self.stream is not None:
            cublas_setstream(self.handle, self.stream)
        

    def axpy(self, n, alpha, x, y, dtype=None, xinc=1, yinc=1):
        """
        cuBLAS function ax plus y.
        
        Parameters
        ----------
        n : int
            Number of elements in the vectors x and y.
        
        alpha : blas_types[dtype]
            Scalar used for multiplication.
            
        x : c_void_p
            Device pointer to vector x.
            
        y : c_void_p
            Device pointer to vector y.
        
        dtype : int, optional
            Data type identifier specified by 'blas_types'.
        
        xinc : int, optional
            Stride between consecutive elements of x.
    
        yinc : int, optional
            Stride between consecutive elements of y.
            
        Notes
        -----
        If a dtype id is not passed in, a dtype will be determined 
        from alpha.
        """
        if dtype is None:
            dtype = [k for k,v in blas_types.items() if v == alpha.dtype][0]

        cublas_axpy(self.handle, n, alpha,
                    x, xinc,
                    y, yinc,
                    dtype)

                           
    def copy(self, n, x, y, dtype, xinc=1, yinc=1):
        """
        cuBLAS function copy vector x to y.
        
        Parameters
        ----------
        n : int
            Number of elements in the vectors x and y.
            
        x : c_void_p
            Device pointer to vector x.
            
        y : c_void_p
            Device pointer to vector y.
        
        dtype : int
            Data type identifier specified by 'blas_types'.
        
        xinc : int, optional
            Stride between consecutive elements of x.
    
        yinc : int, optional
            Stride between consecutive elements of y.
            
        Notes
        -----
        This is the equivalent of doing a cuda_memcpyd2d(...)
        """
        cublas_copy(self.handle, n,
                    x, xinc,
                    y, yinc,
                    dtype)
        
        
    def ewmm(self, x, y, dims, dtype):
        """
        BLAS-like function copy vector x to y. This is not an official 
        function in the cuBLAS library, however may be useful in code that 
        deals with matrix operations.
        
        This code will do y = y*x.
        
        Parameters
        ----------
        x : c_void_p
            Device pointer to vector x.
            
        y : c_void_p
            Device pointer to vector y.
            
        dims : list or np.ndarray
            The dimensions of vectors x and y. Up to three dimensions 
            are currently supported.
            
        dtype : int
            Data type identifier specified by 'blas_types'.
        """
        if type(dims) is list:
            dims = np.array(dims, dtype='i4')
        cublas_ewmm(x, y, dims,
                    dtype,
                    self.stream)


    def nrm2(self, n, x, dtype, xinc=1):
        """
        Computes the Euclidean norm of the vector x, and stores 
        the result on host array y.
        
        Parameters
        ----------
        n : int
            Number of elements in the vectors x
            
        x : c_void_p
            Device pointer to vector x.
          
        dtype : int
            Data type identifier specified by 'blas_types'.
            
        xinc : int, optional
            Stride between consecutive elements of x.
            
        Returns
        -------
        y : blas_types[dtype]
            Euclidean norm of the vector x.
        """
        y = np.empty(1, dtype=blas_types[dtype])
        cublas_nrm2(self.handle, n,
                    x, xinc,
                    y,
                    dtype)
        return y[0]
 
    
    def scal(self, n, alpha, x, dtype=None, xinc=1):
        """
        Scales the vector x by the scalar alpha and overwrites itself 
        with the result.
        
        Parameters
        ----------
        n : int
            Number of elements in the vectors x
        
        alpha : blas_types[dtype]
            Scalar used for multiplication.
            
        x : c_void_p
            Device pointer to vector x.
        
        dtype : int, optional
            Data type identifier specified by 'blas_types'.
        
        xinc : int, optional
            Stride between consecutive elements of x.
            
        Notes
        -----
        If a dtype id is not passed in, a dtype will be determined 
        from alpha.
        """
        if dtype is None:
            dtype = [k for k,v in blas_types.items() if v == alpha.dtype][0]
            
        cublas_scal(self.handle, n, alpha,
                    x, xinc,
                    dtype)
    

    @property
    def handle(self):
        return self._blas_handle


    @property
    def stream(self):
        return self._stream
        
                      
    def __enter__(self):
        return self


    def __exit__(self, *args, **kwargs):
        cublas_destroy(self.blas_handle)
        return