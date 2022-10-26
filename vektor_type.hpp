#ifndef __VEKTOR_TYPE_HPP
#define __VEKTOR_TYPE_HPP

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <array>
#include <iostream>
#include <fstream>
#include <tuple>
#include <type_traits>

#if defined __CUDA_ARCH__  or defined __NVCC__ 
        #define __target__ __host__ __device__ 
#else 
        #define __target__ 
#endif 

namespace math 
{
    // :::::::::::::::::::::::::::::: types and constants:

    #if defined( __CUDA_ARCH__ ) and __CUDA_ARCH__ < 600 
    __device__ double atomicAdd( double* address, double const val )
    {
        unsigned long long int* address_as_ull = ( unsigned long long int* ) address;
        unsigned long long int old = *address_as_ull, assumed;
       
        do 
        {
            assumed = old;
            old = atomicCAS( address_as_ull, assumed, __double_as_longlong( val + __longlong_as_double( assumed ) ) );
        } 
        while (assumed != old);
                        
        return __longlong_as_double( old );
    }
    #endif

    template< typename base_type >
    struct vektor_type
    {
        base_type x, y, z;
            
        inline vektor_type()                     = default;
        inline vektor_type( vektor_type const& ) = default;
        inline vektor_type( vektor_type&& )      = default;

        __target__ inline vektor_type( base_type _x, base_type _y, base_type _z ) : x(_x ), y(_y ), z(_z ) {}  
            
        template< typename T > 
        __target__ inline vektor_type( vektor_type< T > const& v )
        {
            x = static_cast< base_type >( v.x ); 
            y = static_cast< base_type >( v.y ); 
            z = static_cast< base_type >( v.z ); 
        }
      
        inline vektor_type& operator = ( vektor_type const& ) = default;
        inline vektor_type& operator = ( vektor_type&& )      = default;
  
        template< typename T >
        __target__ inline vektor_type& operator = ( vektor_type< T > const& v )
        {
            x = static_cast< base_type >( v.x ); 
            y = static_cast< base_type >( v.y ); 
            z = static_cast< base_type >( v.z ); 

            return *this;
        }
        
        __target__ inline bool sane() const 
        { 
            return isfinite( x ) and isfinite( y ) and isfinite( z );
        }

        __target__ inline vektor_type& operator=  ( base_type const& d )         { x = d; y = d; z = d; return *this; }
        __target__ inline bool         operator== ( vektor_type const& o ) const { return ( x == o.x ) && ( y == o.y ) && ( z == o.z ); }
        __target__ inline vektor_type& operator+= ( vektor_type const& o )       { x += o.x; y += o.y; z += o.z; return *this; }
        __target__ inline vektor_type& operator-= ( vektor_type const& o )       { x -= o.x; y -= o.y; z -= o.z; return *this; }
        __target__ inline vektor_type& operator+= ( base_type const& d )         { x += d; y += d; z += d; return *this; }
        __target__ inline vektor_type& operator-= ( base_type const& d )         { x -= d; y -= d; z -= d; return *this; }
        __target__ inline vektor_type& operator*= ( base_type const& d )         { x *= d; y *= d; z *= d; return *this; }
        __target__ inline vektor_type& operator/= ( base_type const& d )         { x *= ( base_type( 1 ) / d ); y *= ( base_type( 1 ) / d ); z *= ( base_type( 1 ) / d ); return *this; }
        __target__ inline vektor_type& operator/= ( vektor_type const& o )       { x /= o.x; y /= o.y; z /= o.z; return *this; }
        
        __target__ inline vektor_type< int > operator%  ( vektor_type< int > const& o ) const { return { static_cast< int >( x ) % o.x, static_cast< int >( y ) % o.y, static_cast< int >( z ) % o.z }; }
        __target__ inline vektor_type< int > operator%= ( vektor_type< int > const& o )       { x = x % o.x; y = y % o.y; z = z % o.z; return *this; }

        __target__ inline vektor_type operator-  ( vektor_type const& o ) const { vektor_type tmp = { x - o.x , y - o.y , z - o.z }; return tmp; }
        __target__ inline vektor_type operator-  ( base_type const& d )   const { vektor_type tmp = { x - d , y - d , z - d }; return tmp; }
        __target__ inline vektor_type operator+  ( vektor_type const& o ) const { vektor_type tmp = { x + o.x , y + o.y , z + o.z }; return tmp; }
        __target__ inline vektor_type operator+  ( base_type const& d )   const { vektor_type tmp = { x + d , y + d , z + d }; return tmp; }
        __target__ inline vektor_type operator*  ( base_type const& d )   const { vektor_type tmp = { x*d , y*d , z*d }; return tmp; }
        __target__ inline vektor_type operator/  ( base_type const& i )   const { vektor_type tmp = { x * ( base_type( 1 ) / i ) , y * ( base_type( 1 ) / i ) , z * ( base_type( 1 ) / i ) }; return tmp; }
        __target__ inline base_type&  operator[] ( size_t const& i )            { return (&x)[i]; }
        __target__ inline vektor_type operator - ()                       const { return { -x, -y, -z }; }

        //template< typename T > 
        //__target__ inline vektor_type operator* ( T const& t )          const { return operator *( static_cast< base_type >( t ) ); }
            
        __target__ inline bool operator== ( base_type const& p )   const { return ( x == p ) && ( y == p ) && ( z == p ); }
        __target__ inline bool operator!= ( base_type const& p )   const { return ( x != p ) || ( y != p ) || ( z != p ); }
        __target__ inline bool operator!= ( vektor_type const& p ) const { return ( x != p.x ) || ( y != p.y ) || ( z != p.z ); }
        
        __target__ inline vektor_type< int > operator<  ( base_type const& p )   const { return { x < p, y < p, z < p }; }
        __target__ inline vektor_type< int > operator>  ( base_type const& p )   const { return { x > p, y > p, z > p }; }
        __target__ inline vektor_type< int > operator<  ( vektor_type const& p ) const { return { x < p.x, y < p.y, z < p.z }; }
        __target__ inline vektor_type< int > operator>  ( vektor_type const& p ) const { return { x > p.x, y > p.y, z > p.z }; }
           
        __target__ vektor_type< int > operator&& ( vektor_type const& o ) const { return { x and o.x, y and o.y, z and o.z }; }
        __target__ inline bool any() const { return x or  y or  z; }
        __target__ inline bool all() const { return x and y and z; }
        __target__ vektor_type< int > elementwise_equal( vektor_type const& o ) const { return { x == o.x, y == o.y, z == o.z }; } 

        __target__ inline void   set( base_type xx, base_type yy, base_type zz )    { x = xx; y = yy; z = zz; }
        __target__ inline base_type trace()                                   const { return x + y + z; }
        __target__ inline base_type diagonal_product()                        const { return x * y * z; }
        __target__ inline base_type squared()                                 const { return x*x + y*y + z*z; }
        __target__ inline base_type    dot_product  ( vektor_type const& o )  const { return  x*o.x + y*o.y + z*o.z; }
        __target__ inline base_type    dot          ( vektor_type const& o )  const { return  dot_product( o ); }
        __target__ inline vektor_type  cross_product( vektor_type const& o )  const { vektor_type tmp = { y*o.z - z*o.y , z*o.x - x*o.z , x*o.y - y*o.x }; return tmp; }
        __target__ inline vektor_type  cross        ( vektor_type const& o )  const { return  cross_product( o ); }

        __target__ inline base_type length() const 
        { 
            return sqrt( squared() );
/*            #ifdef __CUDA_ARCH__
                if ( std::is_same< base_type, double >::value )  
                    return norm3d( x, y, z );
                else if ( std::is_same< base_type, float >::value )  
                    return norm3df( x, y, z );
                else 
                    return sqrt( squared() );
            #else
                return sqrt( squared() );
            #endif */
        }
        
        __target__ inline vektor_type periodic() const 
        { 
            return { x - static_cast< base_type >( rintf( x ) ),
                     y - static_cast< base_type >( rintf( y ) ),
                     z - static_cast< base_type >( rintf( z ) ) };
        }
                                                                                
        __target__ inline vektor_type inverse()                               const { vektor_type tmp = { base_type( 1 ) / x, base_type( 1 ) / y, base_type( 1 ) / z }; return tmp; }
        __target__ inline vektor_type scaled_with    ( vektor_type const& o ) const { vektor_type tmp = { x*o.x, y*o.y, z*o.z }; return tmp; }
        __target__ inline vektor_type xy_scaled_with ( vektor_type const& o ) const { vektor_type tmp = { x*o.x, y*o.y, z     }; return tmp; }
        __target__ inline base_type angle_inbetween( vektor_type const& o )   const { return acos( dot_product( o ) / ( o.length() * length() ) ); }
        __target__ inline vektor_type unit_vektor() const 
        { 
            base_type il = 1 / length(); 
            return { x*il, y*il, z*il }; 
        }
       
        #if defined __CUDA_ARCH__ or defined __NVCC__ 
        __device__ inline void atomic_add( vektor_type const& v ) 
        {
            atomicAdd( &x, v.x ); 
            atomicAdd( &y, v.y ); 
            atomicAdd( &z, v.z ); 
        } 

        __device__ inline void atomic_add( base_type const& a, base_type const& b, base_type const& c ) 
        {
            atomicAdd( &x, a );
            atomicAdd( &y, b );
            atomicAdd( &z, c );
        }
        #endif

        template< typename F >
        __target__ void element_wise( F&& f ) { f(x); f(y); f(z);  }

        inline void write_binary( ::std::ofstream &stream ) const
        {
            stream.write( (char*) &x , 3 * sizeof( base_type ) );
        }

        inline void read_binary(  ::std::ifstream &stream )
        {
            stream.read(  (char*) &x , 3 * sizeof( base_type ) );
        }

        __target__ inline void print() const
        {
           printf( "%f, %f, %f\n", x, y, z );
        }
    };

    template< typename T >
    __target__ inline vektor_type< T >      operator - ( T t,      vektor_type< T > v ) { return -( v - t ); }
    template< typename T >
    __target__ inline vektor_type< T >      operator + ( T t,      vektor_type< T > v ) { return v + t; }
    template< typename T >
    __target__ inline vektor_type< T >      operator * ( T t,      vektor_type< T > v ) { return v * t; }
    __target__ inline vektor_type< double > operator * ( float t,  vektor_type< double > v ) { return v * static_cast< double >( t ); }
    __target__ inline vektor_type< float >  operator * ( double t, vektor_type< float > v )  { return v * static_cast< float >( t ); }
    template< typename T >
    __target__ inline vektor_type< T > operator / ( T d, vektor_type< T > v ) { return v / d; }

    template< typename base_type >
    static ::std::ostream& operator<< ( ::std::ostream& os, vektor_type< base_type > const& v )
    {
        os << v.x << " " << v.y << " " << v.z;
        return os;
    }

    template< typename base_type >
    static ::std::istream& operator>> ( ::std::istream& is, vektor_type< base_type > &v )
    {
        is >> v.x >> v.y >> v.z;
        return is;
    }

    using float_type    = float;
    using vektor        = vektor_type< float_type >;
    using float_vektor  = vektor_type< float >;
    using double_vektor = vektor_type< double >;
    using int_vektor    = vektor_type< int >;
    using uint_vektor   = vektor_type< uint32_t >;

    /**
     *  component wise !!!
     */                                                       
    __target__ __inline__
    vektor  max( vektor v, vektor w ) { return { fmaxf( v.x, w.x ), 
                                                 fmaxf( v.y, w.y ), 
                                                 fmaxf( v.z, w.z ) }; } 
    __target__ __inline__                           
    vektor  min( vektor v, vektor w ) { return { fminf( v.x, w.x ), 
                                                 fminf( v.y, w.y ), 
                                                 fminf( v.z, w.z ) }; }

    __target__ __inline__
    vektor floor( vektor v ) { return {  floorf(v.x), 
                                         floorf(v.y), 
                                         floorf(v.z) }; } 
    __target__ __inline__
    vektor round( vektor v ) { return {  roundf(v.x), 
                                         roundf(v.y), 
                                         roundf(v.z) }; }
    __target__ __inline__
    vektor abs( vektor v ) { return {  fabsf(v.x), 
                                       fabsf(v.y), 
                                       fabsf(v.z) }; }

    namespace constants 
    {
        vektor const x_axis = { 1, 0, 0 },
                     y_axis = { 0, 1, 0 },
                     z_axis = { 0, 0, 1 };
    }
}

#undef __target__ 
#endif // __VEKTOR_TYPE_HPP
