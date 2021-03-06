# -*- coding: utf-8 -*-

__all__ = [
    "cublas_axpy",
    "cublas_copy",
    "cublas_destroy",
    "cublas_dgmm",
    "cublas_gemm",
    "cublas_gemm_strided_batched",
    "cublas_init",
    "cublas_nrm2",
    "cublas_rf_batched",
    "cublas_ri_batched",
    "cublas_scal",
    "cublas_setstream",
    "cublasx_dgmm_batched",
    "cublasx_diag",
]

import os
from numpy.ctypeslib import ndpointer
from ctypes import (c_int,
                    c_uint64,
                    c_void_p,
                    c_bool)

# Load the shared library
from shared_utils import load_lib
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "lib"))
cublas_lib = load_lib(lib_path,"cublas")


# Define argtypes for all functions to import
argtype_defs = {

    "cublas_axpy" : [
            c_void_p,              #Pointer to cuBLAS handle
            c_int,                 #Number of elements in array
            ndpointer(),           #Scalar used for multplication
            c_void_p,              #Device pointer to x
            c_int,                 #Increment of x
            c_void_p,              #Device pointer to y
            c_int,                 #Increment of y
            c_int],                #Data type identifier
                     
                     
    "cublas_copy" : [
            c_void_p,              #Pointer to cuBLAS handle
            c_int,                 #Number of elements in array
            c_void_p,              #Device pointer to x
            c_int,                 #Increment of x
            c_void_p,              #Device pointer to y
            c_int,                 #Increment of y
            c_int],                #Data type identifier

                     
    "cublas_destroy" : [
            c_void_p],             #Pointer to cuBLAS handle


    "cublas_dgmm" : [
            c_void_p,              #Pointer to cuBLAS handle
            c_int,                 #Operation side mode
            c_int,                 #m (Number of rows of matrix A and C)
            c_int,                 #n (Number of cols of matrix A and C)
            c_void_p,              #Device pointer to matirx A
            c_int,                 #Leading dimension length of A
            c_void_p,              #Device pointer to 1d array x
            c_int,                 #Stride of x
            c_void_p,              #Device pointer to matrix c
            c_int,                 #Leading dimension length of C
            c_int],                #Data type identifier

    "cublasx_diag" : [
            c_void_p,
            c_int,
            c_int,
            c_int,
            c_int,
            c_void_p],

    "cublas_gemm" : [
            c_void_p,              #Pointer to cuBLAS handle
            c_int,                 #Operation OP(A) (normal or transpose)
            c_int,                 #Operation OP(B) (normal or transpose)
            c_int,                 #Number of rows of matrix OP(A) and C
            c_int,                 #Number of cols of matrix OP(B) and C
            c_int,                 #Number of cols of matrix OP(A) and rows of OP(B)
            ndpointer(),           #Scalar used for multiplication
            c_void_p,              #Device pointer to matirx A
            c_int,                 #Leading dimension length of A
            c_void_p,              #Device pointer to matirx B
            c_int,                 #Leading dimension length of B
            ndpointer(),           #Ccalar used for multiplication
            c_void_p,              #Device pointer to matirx C
            c_int,                 #Leading dimension length of C
            c_int,                 #Data type identifier
            c_bool],               #3m optimization flag


    "cublas_gemm_strided_batched" : [
            c_void_p,              #Pointer to cuBLAS handle
            c_int,                 #Operation OP(A) (normal or transpose)
            c_int,                 #Operation OP(B) (normal or transpose)
            c_int,                 #Number of rows of matrix OP(A) and C
            c_int,                 #Number of cols of matrix OP(B) and C
            c_int,                 #Number of cols of matrix OP(A) and rows of OP(B)
            ndpointer(),           #Scalar used for multiplication
            c_void_p,              #Device pointer to matirx A
            c_int,                 #Leading dimension length of A
            c_uint64,              #Stride of matrix A
            c_void_p,              #Device pointer to matirx B
            c_int,                 #Leading dimension length of B
            c_uint64,              #Stride of matrix B
            ndpointer(),           #Ccalar used for multiplication
            c_void_p,              #Device pointer to matirx C
            c_int,                 #Leading dimension length of C
            c_uint64,              #Stride of matrix C
            c_int,                 #Batch size
            c_int,                 #Data type identifier
            c_bool],               #3m optimization flag


    "cublas_init" : [],


    "cublas_nrm2" : [
            c_void_p,              #Pointer to cuBLAS handle
            c_int,                 #Number of elements in the array
            c_void_p,              #Device pointer to x
            c_int,                 #Increment of x
            ndpointer(),           #Host pointer to the result
            c_int],                #Data type identifier


    "cublas_rf_batched" : [
            c_void_p,              #Pointer to cuBLAS handle
            c_int,                 #Number of rows of each matrix in A_array
            c_void_p,              #Device double pointer to A_array
            c_void_p,              #Device pivot array info
            c_void_p,              #Device info array
            c_int,                 #Batch size
            c_int],                #Data type identifier


    "cublas_ri_batched" : [
            c_void_p,              #Pointer to cuBLAS handle
            c_int,                 #Number of rows of each matrix in A_array
            c_void_p,              #Device double pointer to A_array
            c_void_p,              #Device pivot array info
            c_void_p,              #Device double pointer to C_array
            c_void_p,              #Device info array
            c_int,                 #Batch size
            c_int],                #Data type identifier
 
    
    "cublas_setstream" : [
            c_void_p,              #Pointer to cuBLAS handle
            c_void_p],             #Pointer to CUDA stream


    "cublas_scal" : [
            c_void_p,              #Pointer to cuBLAS handle
            c_int,                 #Number of elements in the array
            ndpointer(),           #Scalar used for multiplication
            c_void_p,              #Device pointer to x
            c_int,                 #Increment of x
            c_int],                #Data type identifier
            
    
    "cublasx_dgmm_batched" : [
            c_void_p,
            c_void_p,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            ]
}


restype_defs = {

    "cublas_init" :       c_void_p,         #Pointer to cuBLAS handle

}


## Import functions from DLL
for func, argtypes in argtype_defs.items():
    restype = restype_defs.get(func)
    vars().update({func: cublas_lib[func]})
    vars()[func].argtypes = argtypes
    vars()[func].restype = restype