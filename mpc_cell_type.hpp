#ifndef __MPC_CELL_TYPE_HPP
#define __MPC_CELL_TYPE_HPP

#include "particle_type.hpp"
#include "vektor_type.hpp"
#include "gpu_utilities.hpp"

/**
 *  @brief Object for data and functions related to the MPC algorithm. Additionally, the
 *         class holds pointers to obstacles. 
 */
struct mpc_cell_type 
{
    using vektor     = math::vektor;
    using float_type = math::float_type;

    unsigned density;       
    vektor   mean_velocity, 
             centre_of_mass;

#if ( defined __CUDACC__ ) or ( defined __NVCC__ ) // hide device functions from gcc 
    
    __device__ void atomic_add( mpc_cell_type const& rhs ) 
    {
        atomicAdd( &density, rhs.density );

        mean_velocity .atomic_add( rhs.mean_velocity );
        centre_of_mass.atomic_add( rhs.centre_of_mass );
    }
    
    /**
     *  @brief increment counter and return last value.
     */
    __device__ unsigned get_particle_index()
    {
        return atomicAdd( &density, 1 );
    }
    
    __device__ unsigned add( particle_type const& particle )
    {
        mean_velocity .atomic_add( particle.velocity );
        centre_of_mass.atomic_add( particle.position ); 
        return atomicAdd( &density, 1 );
    }
    
    __device__ void unlocked_add( particle_type const& particle )
    {
        density += 1;
        mean_velocity  += particle.velocity;
        centre_of_mass += particle.position; 
    }
    
    __device__ void unlocked_subtract_velocity( particle_type const& particle )
    {
        mean_velocity  -= particle.velocity;
    }

    __device__ void average()
    {
        centre_of_mass  *= float_type( 1.0 ) / density;
        mean_velocity   *= float_type( 1.0 ) / density;
    }

    __device__ vektor const get_correction( vektor const& position ) const
    {
        return mean_velocity; 
    }
    

    __device__ void add_reduce_only( vektor const& velocity )
    {
        atomicAdd( &density, 1 );
        mean_velocity.atomic_add( velocity ); 
    }
   
    __device__ void average_reduce_only() { if( density > 0 ) { mean_velocity = mean_velocity / density; } }

    __device__ void clear()
    {
        density        = {};
        mean_velocity  = {};
        centre_of_mass = {};
    }
    
    __device__ void group_reduce( unsigned group_size )
    {
        if ( group_size > 1 )
        {
            density          = gpu_utilities::group_sum( density,         -1u, group_size );  
            mean_velocity.x  = gpu_utilities::group_sum( mean_velocity.x,  -1u, group_size );  
            mean_velocity.y  = gpu_utilities::group_sum( mean_velocity.y,  -1u, group_size );  
            mean_velocity.z  = gpu_utilities::group_sum( mean_velocity.z,  -1u, group_size );  
            centre_of_mass.x = gpu_utilities::group_sum( centre_of_mass.x, -1u, group_size );  
            centre_of_mass.y = gpu_utilities::group_sum( centre_of_mass.y, -1u, group_size );  
            centre_of_mass.z = gpu_utilities::group_sum( centre_of_mass.z, -1u, group_size );  
        } 
    }

#endif // hide from gcc
};
    
template< typename T >
static std::ostream& operator<< ( std::ostream& os, mpc_cell_type const& c )
{
    os << c.density << ' ' << c.centre_of_mass << ' ' << c.mean_velocity;
    return os;
}

/**
 *  @brief Minimal fluid state struct. The 3rd moment is not neccesary because there is a thermostat.
 */
struct fluid_state_type
{
    math::float_type density;      // 1st moment 
    math::vektor     mean_velocity; // 2nd moment
};

#endif // __MPC_CELL_TYPE_HPP
