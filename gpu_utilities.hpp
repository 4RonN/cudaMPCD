#ifndef __GPU_UTILITIES_HPP
#define __GPU_UTILITIES_HPP

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>

#include "vektor_type.hpp"

namespace gpu_utilities
{
    using vektor     = math::vektor;
    using float_type = math::float_type;

#if ( defined __CUDACC__ ) or ( defined __NVCC__ ) 

    template< typename T > 
    __device__ __inline__ T warp_prefix_sum( T t ) // upward
    {
        T prefix_sum = t;

        for ( int i = 1; i < 32; i <<= 1 )
        {
            T lower = __shfl_up_sync( 0xFFFFFFFF, prefix_sum, i,  32 );
            if ( threadIdx.x >= i )
                prefix_sum += lower;
        }
        
        return prefix_sum - t;
    }

    template< typename T > 
    __device__ __inline__ T block_prefix_sum( T t, T* shared_mem, T& sum ) // upward
    {
        T prefix_sum = t;

        for ( int i = 1; i < 32; i <<= 1 )
        {
            T lower = __shfl_up_sync( 0xFFFFFFFF, prefix_sum, i,  32 );
            if ( ( threadIdx.x % 32 ) >= i )
                prefix_sum += lower;
        }

        __syncthreads();

        if ( ( threadIdx.x % 32 ) == 31 )
            shared_mem[ threadIdx.x / 32 ]  = prefix_sum;
        
        __syncthreads();
        
        T block_prefix = 0;
        if ( ( threadIdx.x % 32 ) < ( blockDim.x / 32 ) )
        {
            block_prefix  = shared_mem[ threadIdx.x % 32 ];
            uint32_t mask = uint32_t( -1 ) >> ( 32 - ( blockDim.x / 32 ) );
            
            for ( int i = 1; i < ( blockDim.x / 32 ); i <<= 1 )
            {
                T lower = __shfl_up_sync( mask, block_prefix, i,  32 );
                if ( ( threadIdx.x % 32 ) >= i )
                    block_prefix += lower;
            }
        }

        sum          = __shfl_sync( 0xFFFFFFFF, block_prefix,               31, 32 );  
        block_prefix = __shfl_sync( 0xFFFFFFFF, block_prefix, threadIdx.x / 32, 32 ) - __shfl_sync( 0xFFFFFFFF, prefix_sum, 31 ); 

        return prefix_sum + block_prefix - t;
    }
    
    template< typename T > 
    __device__ __inline__ T group_bitwise_or( T t, unsigned mask, unsigned group_size ) // upward
    {
        if ( group_size == 1 )
            return t;

        if ( group_size == 2 )
            return t | __shfl_xor_sync( mask, t, 0x1 );  

        unsigned half = group_size / 2;
        unsigned step = group_size;

        while ( half )
        {
            step = half + step % 2;

            T add = __shfl_down_sync( mask, t, step );

            if ( ( threadIdx.x % group_size ) < half )
                t |= add;
            
            half = step / 2;
        }

        return __shfl_sync( mask, t, ( threadIdx.x / group_size ) * group_size ); // distribute 
    }
    
    template< typename T > 
    __device__ __inline__ T group_sum( T t, unsigned mask, unsigned group_size ) // upward
    {
        if ( group_size == 1 )
            return t;

        if ( group_size == 2 )
            return t + __shfl_xor_sync( mask, t, 0x1 );  

        unsigned half = group_size / 2;
        unsigned step = group_size;

        while ( half )
        {
            step = half + step % 2;

            T add = __shfl_down_sync( mask, t, step );

            if ( ( threadIdx.x % group_size ) < half )
                t += add;
            
            half = step / 2;
        }

        return __shfl_sync( mask, t, ( threadIdx.x / group_size ) * group_size );   
    }

    __device__ __inline__ vektor group_sum( vektor v, unsigned mask, unsigned group_size )
    {
       return { group_sum( v.x, mask, group_size ), group_sum( v.y, mask, group_size ), group_sum( v.z, mask, group_size ) };
    } 
   
    template< typename T > 
    __device__ __inline__ T group_share( T t, unsigned mask, unsigned group_size )
    {
       if ( group_size > 1 )
           return __shfl_sync( mask, t, ( threadIdx.x / group_size ) * group_size );
       else 
           return t;
    }

    __device__ __inline__ vektor group_share( vektor v, unsigned mask, unsigned group_size )
    {
       if ( group_size > 1 )
           return { __shfl_sync( mask, v.x, ( threadIdx.x / group_size ) * group_size ),
                    __shfl_sync( mask, v.y, ( threadIdx.x / group_size ) * group_size ),
                    __shfl_sync( mask, v.z, ( threadIdx.x / group_size ) * group_size ) };
       else 
           return v;
    } 

    template< typename T > 
    __device__ __inline__ T const texture_load( T const* t ) 
    {
        T loaded;
        
        for ( int i = 0; i < sizeof( T ) / 16; ++ i )
           reinterpret_cast< float4* >( & loaded )[ i ] = __ldg( reinterpret_cast< float4 const* >( t ) + i ); 
        
        
        for ( int i = 0; i < ( sizeof( T ) % 16 ) / 8; ++ i )
           reinterpret_cast< float2* >( reinterpret_cast< float4* >( & loaded ) + sizeof( T ) / 16 )[ i ] 
                                                        = __ldg( reinterpret_cast< float2 const* >( reinterpret_cast< float4 const* >( t ) + sizeof( T ) / 16 ) + i ); 
        
        for ( int i = 0; i < ( sizeof( T ) % 8 ) / 4; ++ i )
           reinterpret_cast< float* >( reinterpret_cast< float2* >( reinterpret_cast< float4* >( & loaded ) + sizeof( T ) / 16 ) + ( sizeof( T ) % 16 ) / 8 )[ i ] 
                                                        = __ldg( reinterpret_cast< float const* >( reinterpret_cast< float2 const* >( reinterpret_cast< float4 const* >( t ) + sizeof( T ) / 16 ) + ( sizeof( T ) % 16 ) / 8 ) + i ); 
        
        return loaded;
    }

#endif 

} // namespace gpu_utilities

#endif // __GPU_UTILITIES_HPP
