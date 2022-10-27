#ifndef __error_check_hpp
#define __error_check_hpp

#include <cstdio>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>

inline __host__ __device__ void error_check( cudaError_t error, const char* loc ="unknown location" )
{ 
    if ( error != 0 )
    {
        #ifndef __CUDA_ARCH__
            printf( "error: %s, ( %s )", cudaGetErrorString( error ), loc );
            cudaDeviceSynchronize();
            exit( -1 );
        #else
            printf( "error: %s", loc );
        #endif
    }
}

inline __host__ __device__ void error_check( const char* loc ="unknown location" )
{ 
    #ifndef __CUDA_ARCH__
    {
        #ifndef NDEBUG
            cudaDeviceSynchronize();
        #endif

        cudaError_t error = cudaGetLastError();
        if ( error != 0 )
        {
                printf( "error: %s, ( %s )\n", cudaGetErrorString( error ), loc );
                cudaDeviceSynchronize();
                exit( -1 );
        }
    }
    #endif
}

#endif
