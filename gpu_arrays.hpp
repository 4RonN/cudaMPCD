#ifndef __GPU_ARRAYS_HPP 
#define __GPU_ARRAYS_HPP 

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>

#include <limits>
#include <iostream>
#include <fstream>
#include <functional>
#include <vector>

#include "gpu_error_check.hpp"

template< class T >
struct async_vector 
{
    typedef       T  value_type;              
    typedef       T& reference;             
    typedef const T& const_reference;
    typedef       T* iterator;        
    typedef const T* const_iterator;

    typedef std::reverse_iterator< iterator >       reverse_iterator;
    typedef std::reverse_iterator< const_iterator > const_reverse_iterator;
    
    typedef size_t    size_type; 
    typedef ptrdiff_t difference_type;

    T*         host_store;
    T*         device_store;
    size_type  count;
    bool       copy;

    // ctors:
        
    #ifndef __CUDA_ARCH__ 
        async_vector() : count( 0 ), copy( 0 ) {};
    #else
        async_vector() = default;
    #endif 

                        async_vector( async_vector && ) = default;
    __host__ __device__ async_vector( async_vector const& rhs ) : host_store( rhs.host_store ), device_store( rhs.device_store ), count( rhs.count ), copy( 1 ) { }  

    __host__ async_vector( size_type c ) : count( c ), copy( 0 ) 
    { 
        cudaMallocHost( (void**) &host_store,   count * sizeof( T ) ); error_check( "alloc async_vector host" ); 
        cudaMalloc    ( (void**) &device_store, count * sizeof( T ) ); error_check( "alloc async_vector device" ); 
    }
    
    __host__ __device__ ~async_vector()  
    { 
        #ifndef __CUDA_ARCH__ 
            if ( !copy and count != 0 )
            {
                cudaFree( device_store ); 
                error_check( ( std::string( "delete/free async_vector device type: " ) + typeid( T ).name() ).c_str() ); 
                cudaFreeHost( host_store ); 
                error_check( ( std::string( "delete/free async_vector host type: " ) + typeid( T ).name() ).c_str() ); 
            }
        #endif 
    }
    
    void push()
    {
        cudaMemcpy( device_store, host_store, count * sizeof( T ), cudaMemcpyHostToDevice );
        error_check( "async_vector::push" );
    }

    void pull()
    {
        cudaMemcpy( host_store, device_store, count * sizeof( T ), cudaMemcpyDeviceToHost );
        error_check( "async_vector::pull" );
    }

    __host__ __device__ iterator       begin()        
    {
        #ifdef __CUDA_ARCH__ 
            return device_store;
        #else 
            return host_store;
        #endif 
    }

    __host__ __device__ const_iterator begin()  const { return begin(); } 
    __host__ __device__ const_iterator cbegin() const { return begin(); } 
    __host__ __device__ iterator       end()          
    {
        #ifdef __CUDA_ARCH__ 
            return device_store + count;
        #else 
            return host_store   + count;
        #endif 
    }
    __host__ __device__ const_iterator end()    const { return end(); }
    __host__ __device__ const_iterator cend()   const { return end(); } 

    __host__ __device__ reverse_iterator       rbegin()
    { return reverse_iterator( end() ); }
    __host__ __device__ const_reverse_iterator rbegin()  const
    { return reverse_iterator( end() ); }
    __host__ __device__ reverse_iterator       rend()
    { return reverse_iterator( begin() ); }
    __host__ __device__ const_reverse_iterator rend()    const
    { return reverse_iterator( begin() ); }

    __host__ __device__ size_type element_size()       { return sizeof( T ); }
    __host__ __device__ size_type element_size() const { return sizeof( T ); }
    __host__ __device__ size_type size()               { return count; }
    __host__ __device__ size_type size()         const { return count; }

    size_type max_size()     const { return count; }
    bool      empty()        const { return count == 0; }

    async_vector< T >* ptr() { return this; }

    __host__ __device__ reference       operator[]( size_type n )
    {
        #ifdef __CUDA_ARCH__ 
            return device_store[ n ];
        #else 
            return host_store[ n ];
        #endif 
    }
    __host__ __device__ const_reference operator[]( size_type n ) const { return operator[] ( n ); }

    __host__ __device__ reference       front()      
    {
        #ifdef __CUDA_ARCH__ 
            return device_store[ 0 ];
        #else 
            return host_store[ 0 ];
        #endif 
    }
    __host__ __device__ const_reference front() const { return front(); } 

    __host__ __device__ reference       back()        
    {
        #ifdef __CUDA_ARCH__ 
            return device_store[ count-1 ];
        #else 
            return host_store[ count-1 ];
        #endif 
    }
    __host__ __device__ const_reference back()  const { return back(); } 

    __host__ __device__ T*              data()
    { 
        #ifdef __CUDA_ARCH__ 
            return device_store;
        #else 
            return host_store;
        #endif 
    }
    __host__ __device__ const T*        data()  const { return data(); }

    void fill( T const value )
    {   for( size_t i = 0; i < count; ++i ) host_store[i] = value;   }
    
    void set( int i ) 
    { error_check( cudaMemset( device_store, i, sizeof( T ) * count ), "gpu_vector set" ); }

    bool            operator== ( async_vector const& o ) const
    {   if ( o.count != count ) return false;
        for ( int i = 0; i < count; ++i )
            if ( !( host_store[i] == o.host_store[i] ) ) return false;
        return true;    }

    // io:
    void write_binary( std::ofstream &stream )
    {
        for ( size_t i = 0; i < count; ++i )
            host_store[i].write_binary( stream );
    }
    void read_binary( std::ifstream &stream )
    {
        for ( size_t i = 0; i < count; ++i )
            host_store[i].read_binary( stream );
    }
};

template< typename T >
struct gpu_vector
{
    using value_type             =       T;
    using reference              =       T&;
    using const_reference        = const T&;
    using iterator               =       T*;
    using const_iterator         = const T*;
    using reverse_iterator       =       std::reverse_iterator< iterator >;
    using const_reverse_iterator =       std::reverse_iterator< const_iterator >;
    using size_type              =       size_t;
    using difference_type        =       ptrdiff_t;

    // data storages:
    T*        store;
    size_type count;
    bool      copy;

    // ctors:
        
    #ifndef __CUDA_ARCH__ 
        __host__ gpu_vector() : count( 0 ), copy( 0 ) {};
    #else
        gpu_vector() = default;
    #endif 

                        gpu_vector( gpu_vector && ) = default;
    __host__ __device__ gpu_vector( gpu_vector const& rhs ) : store( rhs.store ), count( rhs.count ), copy( 1 ) { }  

    __host__ gpu_vector( size_type c )           : count( c ) { cudaMalloc( (void**) &store, count * sizeof( T ) ); error_check( "alloc gpu_vector" ); }
    __host__ gpu_vector( size_type c, int init ) : count( c ) { cudaMalloc( (void**) &store, count * sizeof( T ) ); error_check( "alloc gpu_vector" ); set( init );  }
    
    __host__ __device__ ~gpu_vector()  
    { 
        #ifndef __CUDA_ARCH__ 
            if ( !copy and count != 0 )
            {
                cudaFree( store );
                error_check( ( std::string( "delete/free gpu_vector type: " ) + typeid( T ).name() ).c_str() ); 
            }
        #endif 
    }

    __host__ void alloc( size_type c ) 
    {
        if ( count != 0 )
            cudaFree( store ); 

        count = c; 
        error_check( cudaMalloc( (void**) &store, count * sizeof( T ) ), "alloc gpu_vector" ); 
    }

    // capacity:
    __device__ __host__ size_type size()     const { return count; }

    // element access:
    __device__ reference  operator[]( size_type n ) const { return store[n]; }
               
    __device__ iterator       begin()        { return store; }
    __device__ const_iterator begin()  const { return store; }
    __device__ const_iterator cbegin() const { return store; }
    __device__ iterator       end()          { return store + count; }
    __device__ const_iterator end()    const { return store + count; }
    __device__ const_iterator cend()   const { return store + count; }
    
    __device__ reference       front()        
    {
            return store[ 0 ];
    }
    __device__ const_reference front() const { return front(); } 

    __device__ reference       back()        
    {
            return store[ count-1 ];
    }
    __device__ const_reference back()  const { return back(); } 

    // data access:
    __device__ __host__ value_type*  data() const { return store; }

    // modify:
    __host__ void set( int i ) { error_check( cudaMemset( store, i, sizeof( T ) * count ), "gpu_vector set" ); }

    T get()
    {
        T retval;
        cudaMemcpy( &retval, store, sizeof( T ), cudaMemcpyDeviceToHost );
        cudaError_t err = cudaGetLastError();
        if ( err != 0 ) printf( "%s\n", cudaGetErrorString( err ) );
        return retval;
    }
};

template< typename T >
gpu_vector< T > push( std::vector< T > const& source )
{
    gpu_vector< T > destination( source.size() );
    cudaMemcpy( destination.data(), source.data(), sizeof( T ) * source.size(), cudaMemcpyHostToDevice );
}

template< typename T >
std::vector< T > pull( gpu_vector< T > const& source )
{
    std::vector< T > destination( source.size() );
    cudaMemcpy( destination.data(), source.data(), sizeof( T ) * source.size(), cudaMemcpyDeviceToHost );
}

#endif // __GPU_ARRAYS_HPP 
