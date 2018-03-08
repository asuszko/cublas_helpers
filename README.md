# cublas_helpers

Part of a Python framework that allows a user to **GPU accelerate their Python code with cuBLAS**. An object is created that handles the interface between Python and cuBLAS. This object is part of a larger container within the [cuda_manager](https://github.com/asuszko/cuda_manager). Thus, while this repo may be used standalone, it is recommended it be downloaded with [cuda_manager](https://github.com/asuszko/cuda_manager). Doing a recursive clone on [cuda_manager](https://github.com/asuszko/cuda_manager) will also clone [cublas_helpers](https://github.com/asuszko/cublas_helpers).


# cuBLAS Support

cuBLAS functions are supported as they become needed by the user. Thus, currently not all functions are available out of the box. When a user needs a new function, it may be added to the source in the same manner others are. When doing this, start a new branch, add the function, compile, test, and then merge request after verifying the added function works. Available cuBLAS functions are listed below.

For reference to official Nvidia documentation:
- [cuBLAS Documentation](http://docs.nvidia.com/cuda/cublas/index.html)

## Setup

To compile the shared library, run the **setup.py** file found in the root folder from the command line, with optional argument(s) -arch, and -cc_bin if on Windows. On Windows, the NVCC compiler looks for cl.exe to compile the C/C++, which comes with Visual Studio. On Linux, it uses the built in GCC compiler. An example of a command line run to compile the code is given below:
> python setup.py -arch=sm_50 -cc_bin="C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin"

If you are unable to compile, you may download precompiled libraries for your latest hardware supported shader architecture. 

## Compiler Requirements

- Python 3.6.x (2.7 compatibility not yet tested) 
- The latest  version of the [Nvidia CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- (Windows) Visual Studio 2013 Professional, or any compatible version with your version of CUDA Toolkit. Note: You can download a trial of Professional to obtain cl.exe. Compilation via the command line will still work after the trial period has ended.

## Supported cuBLAS Functions

- [cublas<t<t>>axpy](http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-axpy)
- [cublas<t<t>>copy](http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-copy)
- [cublas<t<t>>dgmm](http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-dgmm)
- [cublas<t<t>>gemm](http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemm)
- [cublas<t<t>>nrm2](http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-nrm2)
- [cublas<t<t>>scal](http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-scal)

### Additional Functions
-  cublas<t<t>>ewmm : Vector element-wise multiply. 

## Samples & Notes

For sample scripts or further documentation on how to use this framework, view [sample scripts](https://github.com/asuszko/cuda_manager/tree/master/samples) that import and utilize cublas_helpers, and/or view the PowerPoint presentation [here](https://github.com/asuszko/cuda_manager/blob/master/link).