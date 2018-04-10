# -*- coding: utf-8 -*-

__all__ = [
    "cublas",
]

import numpy as np

# Local imports
from cublas_import import (cublas_axpy,
                           cublas_copy,
                           cublas_destroy,
                           cublas_gemm,
                           cublas_init,
                           cublas_nrm2,
                           cublas_scal,
                           cublas_setstream)

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


    def axpy(self, alpha, x, y, xinc=1, yinc=1, n=None):
        """
        cuBLAS function ax plus y.
        
        Parameters
        ----------
        alpha : blas_types[dtype]
            Scalar used for multiplication.
            
        x : Device_Ptr object
            Device pointer object with dev_ptr to vector x.
            
        y : Device_Ptr object
            Device pointer object with dev_ptr to vector x.
        
        xinc : int, optional
            Stride between consecutive elements of x.
    
        yinc : int, optional
            Stride between consecutive elements of y.
            
        n : int, optional
            Number of elements in the vectors x and y.
        """
        if type(alpha) is not np.ndarray:
            alpha = np.array(alpha, dtype=x.dtype)
        n = n or len(x)
        check_vectors(x,y)
        cublas_axpy(self.handle, n, alpha,
                    x.ptr, xinc,
                    y.ptr, yinc,
                    _blas_types[x.dtype])

                           
    def copy(self, x, y, xinc=1, yinc=1, n=None):
        """
        cuBLAS function copy vector x to y.
        
        Parameters
        ----------
        x : Device_Ptr object
            Device pointer object with dev_ptr to vector x.
            
        y : Device_Ptr object
            Device pointer object with dev_ptr to vector y.
        
        xinc : int, optional
            Stride between consecutive elements of x.
    
        yinc : int, optional
            Stride between consecutive elements of y.
            
        n : int
            Number of elements in the vectors x and y.
            
        Notes
        -----
        This is the equivalent of doing a cuda_memcpyd2d(...)
        """
        check_vectors(x,y)
        n = n or len(x)
        cublas_copy(self.handle, n,
                    x.ptr, xinc,
                    y.ptr, yinc,
                    _blas_types[x.dtype])


    def gemm(self, alpha, a, b, c, beta=0, OPA='N', OPB='N'):
        """
        cuBLAS function gemm.
        
        Parameters
        ---------- 
        alpha : blas_types[dtype]
            Scalar used for multiplication.
            
        a : Device_Ptr object
            Device pointer object with dev_ptr to input matrix a.
            
        b : Device_Ptr object
            Device pointer object with dev_ptr to input matrix b.
        
        c : Device_Ptr object
            Device pointer object with dev_ptr to output matrix c.
    
        beta : blas_types[dtype], optional
            Not really sure what beta does. Documentation sets 
            this to zero when doing a gemm.
            
        OPA : str, optional
            CUBLAS_OP_N ('N') or CUBLAS_OP_T ('T')
            
        OPB : str, optional
            CUBLAS_OP_N ('N') or CUBLAS_OP_T ('T')
        
        Notes
        -----
        Dealing with cuBLAS FORTRAN style indexing:
            https://peterwittek.com/cublas-matrix-c-style.html
        """
        check_vectors(a,b)
        check_vectors(b,c)
        
        m = b.shape[1]
        n, k = a.shape
        
        if type(alpha) is not np.ndarray:
            alpha = np.array(alpha, dtype=a.dtype)
            
        beta = np.array(beta, dtype=a.dtype)
        
        cublas_gemm(self.handle,
                    {'N':0, 'T':1}[OPA],
                    {'N':0, 'T':1}[OPB],
                    m, n, k,
                    alpha,
                    b.ptr, m,
                    a.ptr, k,
                    beta,
                    c.ptr, m,
                    _blas_types[a.dtype])


    def nrm2(self, x, xinc=1, n=None):
        """
        Computes the Euclidean norm of the vector x, and stores 
        the result on host array y.
        
        Parameters
        ----------
        x : Device_Ptr object
            Device pointer object with dev_ptr to vector x.
            
        xinc : int, optional
            Stride between consecutive elements of x.
            
        n : int, optional
            Number of elements in the vectors x
            
        Returns
        -------
        y : blas_types[dtype]
            Euclidean norm of the vector x.
        """
        y = np.empty(1, dtype=x.dtype)
        n = n or len(x)
        cublas_nrm2(self.handle, n,
                    x.ptr, xinc,
                    y,
                    _blas_types[x.dtype])
        return y[0]
 
    
    def scal(self, alpha, x, xinc=1, n=None):
        """
        Scales the vector x by the scalar alpha and overwrites itself 
        with the result.
        
        Parameters
        ----------       
        alpha : blas_types[dtype]
            Scalar used for multiplication.
            
        x : Device_Ptr object
            Device pointer object with dev_ptr to vector x.
            
        xinc : int, optional
            Stride between consecutive elements of x.
            
        n : int, optional
            Number of elements in the vectors x
        """
        if type(alpha) is not np.ndarray:
            alpha = np.array(alpha, dtype=x.dtype)
        n = n or len(x)
        cublas_scal(self.handle, n, alpha,
                    x.ptr, xinc,
                    _blas_types[x.dtype])
    

    @property
    def handle(self):
        return self._blas_handle


    @property
    def stream(self):
        return self._stream
        
                      
    def __enter__(self):
        return self


    def __exit__(self, *args, **kwargs):
        cublas_destroy(self.handle)