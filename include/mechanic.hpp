#ifndef __MECHANIK_HPP
#define __MECHANIK_HPP

#include "vektor_type.hpp"

template< typename T > 
struct symmetric_matrix
{
    T xx = {},
      yy = {},
      zz = {}, 
      xy = {},
      xz = {},
      yz = {};
   
    __device__ symmetric_matrix() {} 
    __device__ symmetric_matrix( T _xx, T _yy, T _zz, T _xy, T _xz, T _yz ) : xx {_xx }, yy {_yy }, zz {_zz }, xy {_xy }, xz {_xz }, yz {_yz } {}

    __device__ T trace()       const { return xx + yy + zz; }
    __device__ T determinant() const { return xx * ( yy * zz -  yz * yz ) + xz * ( 2 * xy * yz - yy * xz ) - ( xy * xy * zz ); }
    
    __device__ math::vektor_type< T > operator* ( math::vektor_type< T > const& v ) const 
    { 
        return { ( xx * v.x ) + ( xy * v.y ) + ( xz * v.z ), 
                 ( xy * v.x ) + ( yy * v.y ) + ( yz * v.z ),
                 ( xz * v.x ) + ( yz * v.y ) + ( zz * v.z ) }; 
    } 

    __device__ symmetric_matrix operator* ( T const& scalar ) const { return { xx * scalar, yy * scalar, zz * scalar, xy * scalar, xz * scalar, yz * scalar }; } 
    
    __device__ symmetric_matrix inverse_no_check( T det )
    {
        if ( det < 0 )
            det = 0;

        T inverse_det = 1.0f / det;

        return { ( yy * zz - yz * yz ) * inverse_det, // xx
                 ( xx * zz - xz * xz ) * inverse_det, // yy
                 ( xx * yy - xy * xy ) * inverse_det, // ...
                 ( xz * yz - zz * xy ) * inverse_det, 
                 ( xy * yz - yy * xz ) * inverse_det,
                 ( xy * xz - xx * yz ) * inverse_det };
    }

    __device__ symmetric_matrix inverse( T const tolerance=1.0e-5 )
    {
        T det = determinant();

        if ( det < 0 )
            det = 0;
        
        if ( det > tolerance )
        {
            T inverse_det = 1 / det;

            return { ( yy * zz - yz * yz ) * inverse_det, // xx
                     ( xx * zz - xz * xz ) * inverse_det, // yy
                     ( xx * yy - xy * xy ) * inverse_det, // ...
                     ( xz * yz - zz * xy ) * inverse_det, 
                     ( xy * yz - yy * xz ) * inverse_det,
                     ( xy * xz - xx * yz ) * inverse_det };
        }
        else
        {
            T tr = trace(); // If all particles lie on a line, the remaining eigenvalues must be equal! -> iversion not neccesary

            if ( tr > tolerance )
                return ( *this ) * ( 4 / ( tr * tr ) );
            else 
                return {};
        }
    }
    __device__ symmetric_matrix invert()
    {
        *this = inverse();
        return *this;
    }

    __device__ void print()
    {
        printf( "%f %f %f %f %f %f\n", xx, yy, zz, xy, xz, yz ); 
    }
};

template< typename T >
struct traegheitsmoment : public symmetric_matrix< T >
{
    using base_class = symmetric_matrix< T >;
    using base_class::xx;
    using base_class::yy;
    using base_class::zz;
    using base_class::xy;
    using base_class::xz;
    using base_class::yz;
    
    // ~~~ ctors: 
    __device__ traegheitsmoment() {} 

    // ~~~ functions:
    __device__ void shift_frame( math::vektor shift, T const& mass = 1 ) // Steinerscher Statz
    {
        auto squares = shift.scaled_with( shift );

        xx += ( squares.y + squares.z ) * mass; 
        yy += ( squares.x + squares.z ) * mass;
        zz += ( squares.x + squares.y ) * mass;
        xy -= ( shift.x * shift.y     ) * mass;
        xz -= ( shift.x * shift.z     ) * mass;
        yz -= ( shift.y * shift.z     ) * mass;
    }
    
    __device__ void unshift_frame( math::vektor_type< T > shift, T const& mass = 1 ) // Steinerscher Statz
    {
        auto squares = shift.scaled_with( shift );

        xx -= ( squares.y + squares.z ) * mass; 
        yy -= ( squares.x + squares.z ) * mass;
        zz -= ( squares.x + squares.y ) * mass;
        xy += ( shift.x * shift.y     ) * mass;
        xz += ( shift.x * shift.z     ) * mass;
        yz += ( shift.y * shift.z     ) * mass;
    }
    
    __device__ T diagonal_to_orientation( math::vektor_type< T > const& axis ) const 
    {
        return axis.x * axis.x * xx + axis.y * axis.y * yy, axis.z * axis.z * zz;
    }

    __device__ traegheitsmoment & operator+= ( base_class const& contribution )
    { 
        xx += contribution.xx; 
        yy += contribution.yy; 
        zz += contribution.zz; 
        xy += contribution.xy; 
        xz += contribution.xz; 
        yz += contribution.yz; 
        return *this; 
    } 
    
    __device__ void atomic_add( base_class const& contribution )
    { 
        atomicAdd( &xx, contribution.xx ); 
        atomicAdd( &yy, contribution.yy ); 
        atomicAdd( &zz, contribution.zz ); 
        atomicAdd( &xy, contribution.xy ); 
        atomicAdd( &xz, contribution.xz ); 
        atomicAdd( &yz, contribution.yz ); 
    }

    __device__ traegheitsmoment& operator=  ( base_class const& o ) 
    { 
        xx = o.xx; yy = o.yy; zz = o.zz; 
        xy = o.xy; yz = o.yz; xz = o.xz; 
        return *this; 
    } 
    __device__ traegheitsmoment& operator=  ( T p ) { xx = p; yy = p; zz = p; xy = p; xz = p; yz = p; return *this; }
};


#endif // __MECHANIK_HPP
