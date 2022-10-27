#ifndef __CUDA_ALLOCATOR_HPP
#define __CUDA_ALLOCATOR_HPP

//#if ( defined __CUDACC__ ) || ( defined __NVCC__ )

void  cuda_set_device( size_t );
void  cuda_device_reset();

void* cuda_malloc_host( size_t );
void  cuda_free( void* );

template< class T >
struct cuda_host_allocator 
{
    typedef T value_type;
   
    cuda_host_allocator() = default;
    template< class U > constexpr 
    cuda_host_allocator( cuda_host_allocator< U > const& ) noexcept {}

    T* allocate( std::size_t n ) 
    {
        if ( n > 0 )
            return static_cast< T* >( cuda_malloc_host( n * sizeof( T ) ) );
        else return nullptr;
    }

    void deallocate( T* p, std::size_t n ) noexcept 
    {
        if ( n > 0 ) 
            cuda_free( p ); 
    }
};

template< class T, class U >
bool operator==( const cuda_host_allocator< T >&, const cuda_host_allocator< U >& ) { return true; }
template< class T, class U >
bool operator!=( const cuda_host_allocator< T >&, const cuda_host_allocator< U >& ) { return false; }

/*#else //////

#include <memory>

template< typename T >
using cuda_host_allocator = std::allocator< T >;
void  cuda_set_device( size_t ) {}

#endif // CUDA or not 
*/
#endif // __CUDA_ALLOCATOR_HPP
