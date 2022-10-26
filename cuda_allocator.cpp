#include <cstdlib>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>

void cuda_set_device( size_t i )
{
    cudaDeviceSynchronize();
    cudaSetDevice( i );
} 

void cuda_device_reset()
{
    cudaDeviceSynchronize();
    cudaDeviceReset();
} 

void* cuda_malloc_host( size_t n ) noexcept
{
    void* p = nullptr;
    cudaMallocHost( (void**) &p, n ); 
    cudaError_t error = cudaGetLastError();
    if ( error != 0 )
    {
        std::cout << "could not allocated pinned memorey" << std::endl;
        p = std::malloc( n ); 
    }
    
    return p;
}

void cuda_free( void* p ) noexcept 
{ 
    cudaFreeHost( p ); 
        
    cudaError_t error = cudaGetLastError();
    if ( error != 0 )
        std::free( p );
}

