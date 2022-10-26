
#include "parameter_set.hpp"
#include "vektor_type.hpp"
#include "gpu_container_classes.hpp"
#include "probing.hpp"
#include "gpu_constants.hpp"

namespace initialize 
{
    /**
     *  @brief Initialize GPU random number generators
     */
    __global__ void random_number_generators( gpu_vector< xoshiro128plus > generator, async_vector< uint64_t > seed )
    {
        size_t idx    = blockIdx.x * blockDim.x + threadIdx.x,
               stride = blockDim.x * gridDim.x; 

        for ( ; idx < generator.size(); idx += stride )
            generator[ idx ].seed( seed[ idx ], seed[ idx + generator.size() ] );
    }
     
    /**
     *  @brief Initialize the fluid by distributing the SRD fluid particles in the simulation volume
     */
    __global__ void distribute_particles( async_vector< particle_type > particles, gpu_volumetric_container< mpc_cell_type > mpc_cells,
                                          gpu_vector< xoshiro128plus > generator, parameter_set parameters, 
                                          math::vektor grid_shift, uint32_t start )
    {
        size_t idx    = blockIdx.x * blockDim.x + threadIdx.x,
               stride = blockDim.x * gridDim.x;
        auto   random = generator[ idx ];
        auto   scale  = parameters.volume_size - 2 * ( 1 - parameters.periodicity );
       
        float channel_radius2 = ( parameters.volume_size.z - 2 ) * ( parameters.volume_size.z - 2 ) * 0.25f; 

        idx += start; 
        for ( auto end = particles.size(); idx < end; idx += stride )
        {
            particle_type particle = {};
            bool          replace;
           
            do 
            { 
                replace  = false;
                particle.position = { random.uniform_float(), random.uniform_float(), random.uniform_float() }; // uniform on the unit cube.
                particle.position = ( particle.position - math::float_type( 0.5 ) ).scaled_with( scale );  // rescale to the simulation volume
        
                if ( parameters.experiment == channel )
                    replace = replace or ( ( particle.position.z * particle.position.z + particle.position.y * particle.position.y ) > channel_radius2 );

            } 
            while ( replace );

            particle.velocity = random.maxwell_boltzmann() * parameters.thermal_velocity;  
            particle.cell_idx = mpc_cells.get_index( particle.position );

            particles[ idx ] = particle;
        }

        generator[ idx % stride ] = random; // store new state of the random number generator
    }

}  //  namespace initialize

/**
 *  @brief This function applies the SRD streaming step to the fluid particles 
 */
__global__ void translate_particles( async_vector< particle_type > particles, gpu_volumetric_container< mpc_cell_type > mpc_cells,
                                     gpu_vector< xoshiro128plus > generator, parameter_set parameters, math::vektor grid_shift, 
                                     gpu_vector< uint32_t > uniform_counter, gpu_vector< uint32_t > uniform_list )
{
    size_t idx    = blockIdx.x * blockDim.x + threadIdx.x,
           stride = blockDim.x * gridDim.x; 
    auto   random = generator[ idx ];

    uint32_t cell_lookup_size = uniform_list.size() / uniform_counter.size(); 
    float channel_radius2 = ( parameters.volume_size.z - 2 ) * ( parameters.volume_size.z - 2 ) * 0.25f; 

    auto apply_periodic_boundaries = [&] ( auto r ) 
    { 
        r.x = fmodf( r.x + 1.5f * parameters.volume_size.x, parameters.volume_size.x ) - parameters.volume_size.x * 0.5f;   
        r.y = fmodf( r.y + 1.5f * parameters.volume_size.y, parameters.volume_size.y ) - parameters.volume_size.y * 0.5f;   
        r.z = fmodf( r.z + 1.5f * parameters.volume_size.z, parameters.volume_size.z ) - parameters.volume_size.z * 0.5f; // does not interfere with using walls...  
        return r; 
    };
    
    for ( auto end = particles.size(); idx < end; idx += stride )
    {
        auto particle = gpu_utilities::texture_load( particles.data() + idx ); // load via texture load path
        
        if ( parameters.experiment != channel )
        { 
            if ( not parameters.periodicity.z ) // if walls are present, calculate collisions
            {
                auto z_wall = ( 0.5f * parameters.volume_size.z - 1 ); // distance of the walls, remove one layer for ghost particles
                auto next_z = particle.position.z + particle.velocity.z * parameters.delta_t;
                
                if ( fabsf( next_z ) > z_wall ) // this is more safe than just calcualating the time 
                {
                    auto time_left = ( next_z > 0 ? z_wall - particle.position.z : -z_wall - particle.position.z ) 
                                       / particle.velocity.z;
                
                    particle.position += particle.velocity * time_left;
                    particle.velocity = -particle.velocity; // bounce back roule, creates no-slip boundary condition 
                    particle.position += particle.velocity * ( parameters.delta_t - time_left );
                }
                else 
                    particle.position += particle.velocity * parameters.delta_t;
            }
            else 
                particle.position += particle.velocity * parameters.delta_t;
        }
        else // channel:
        {
            particle.position += particle.velocity * parameters.delta_t; // advance particle

            if ( ( particle.position.z * particle.position.z + particle.position.y * particle.position.y ) > channel_radius2 )
            {
                particle.position -= particle.velocity * parameters.delta_t; // apply correction if it left the channel
                particle.velocity = -particle.velocity;
            } 
        }

        particle.position = apply_periodic_boundaries( particle.position + grid_shift );
        particle.cell_idx = mpc_cells.get_index( particle.position );
     
        // make an entry in the index lookup for the cell in which the particle lies 
        int slot = atomicAdd( uniform_counter.data() + particle.cell_idx, 1 );
        if ( slot < cell_lookup_size ) uniform_list[ particle.cell_idx + slot * mpc_cells.size() ] = idx;
        else particle.position = apply_periodic_boundaries( particle.position - grid_shift ); 
        // if the lookup is full, the particle gets no shift because after the collision step particle are not shifted 
        
        assert( particle.position.sane() ); // check for error in floating point math
        assert( particle.velocity.sane() );

        particles[ idx ] = particle;
    }
    
    generator[ idx % stride ] = random;
}

/**
 *  @brief This function applies the SRD collision step to the fluid particles 
 */
__global__ void __launch_bounds__( 32, 8 ) srd_collision( async_vector< particle_type > particles, gpu_volumetric_container< mpc_cell_type > mpc_cells,
                                                          xoshiro128plus* generator, math::vektor grid_shift, gpu_vector< uint32_t > uniform_counter, 
                                                          gpu_vector< uint32_t > uniform_list, uint32_t const shared_bytes )
{
    extern __shared__ uint32_t shared_mem [];
   
    // asign shared memory for particle positions and velocities 
    uint32_t const max_particles      = shared_bytes / ( sizeof( uint32_t ) + 2 * sizeof( math::vektor ) ); // this is the per particle memory
    uint32_t     * particle_idx       = shared_mem;                                                         // storing the indices of loaded particles
    math::vektor * particle_position  = reinterpret_cast< math::vektor* >( particle_idx + max_particles );  // 1st vector
    math::vektor * particle_velocity  = particle_position + max_particles;                                  // 2nd vector 

    uint32_t       cell_idx           = blockIdx.x * blockDim.x + threadIdx.x,
                   stride             = blockDim.x * gridDim.x;

    auto           random             = generator[ cell_idx ];
    uint32_t const cell_lookup_size   = uniform_list.size() / uniform_counter.size(); 
    auto     const regular_particles  = particles.size();

    auto const     shift              = fabs( grid_shift.z ); // create wall's ghost particles on the fly  
    unsigned const sign               = grid_shift.z > 0;
    float const    drag               = gpu_const::parameters.drag * gpu_const::parameters.delta_t, // pressure gradient that accelerates the flow
                   sin_alpha          = sinf( ( M_PI * 120 ) / 180 ), // these are used to represent the SRD rotation matrix
                   cos_alpha          = cosf( ( M_PI * 120 ) / 180 );
    
    auto apply_periodic_boundaries = [&] ( auto r ) // does not interfere with using walls...  
    { 
        r.x = fmodf( r.x + 1.5f * gpu_const::parameters.volume_size.x, gpu_const::parameters.volume_size.x ) - gpu_const::parameters.volume_size.x * 0.5f;   
        r.y = fmodf( r.y + 1.5f * gpu_const::parameters.volume_size.y, gpu_const::parameters.volume_size.y ) - gpu_const::parameters.volume_size.y * 0.5f;   
        r.z = fmodf( r.z + 1.5f * gpu_const::parameters.volume_size.z, gpu_const::parameters.volume_size.z ) - gpu_const::parameters.volume_size.z * 0.5f;
        return r; 
    };
    
    for ( uint32_t end = mpc_cells.size(); __any_sync( 0xFFFFFFFF, cell_idx < end ); cell_idx += stride ) 
    {
        random.sync_phase();

        mpc_cell_type cell         = {};
        uint32_t      n_particles  = ( cell_idx < mpc_cells.size() ) ? min( cell_lookup_size, uniform_counter[ cell_idx ] ) : 0; // load the table size
        bool const    layer        = ( mpc_cells.get_z_idx( cell_idx ) == ( gpu_const::parameters.volume_size.z - ( sign ? 1 : 2 ) ) ); // wall layer? 
        bool const    add_ghosts   = ( not gpu_const::parameters.periodicity.z ) and ( ( mpc_cells.get_z_idx( cell_idx ) == sign ) or layer ); // wall layer?
        uint32_t      added_ghosts = {};

        if ( add_ghosts  ) // create wall's ghost particles on the fly; prepare number of ghosts 
            for ( int i = 0; i < gpu_const::parameters.n; ++i )
                if ( ( ( random.uniform_float() > shift ) xor layer ) xor sign ) 
                    ++added_ghosts;

        n_particles += added_ghosts;

        // arrange shared memory and decides how many cells can be fit into memory and handeled at once 
        uint32_t const prefix        = gpu_utilities::warp_prefix_sum( n_particles );
        uint32_t const active_cells  = __popc( __ballot_sync( 0xFFFFFFFF, prefix + n_particles < max_particles and cell_idx < mpc_cells.size() ) ); 
        uint32_t const sum           = __shfl_sync( 0xFFFFFFFF, prefix + n_particles, active_cells - 1 ); 
        bool     const thread_active = prefix + n_particles < max_particles and cell_idx < mpc_cells.size();
        
        if ( thread_active )
        {
            for ( uint32_t i = 0; i < n_particles - added_ghosts; ++i ) 
                particle_idx[ prefix + i ] = cell_idx + i * mpc_cells.size(); // write to shared mem which lookup table positions need to be loaded.    
            
            if ( add_ghosts ) // create wall's ghost particles on the fly
            {
                auto pos = mpc_cells.get_position( cell_idx );
            
                for ( int i = n_particles - added_ghosts; i < n_particles; ++i )
                {
                    float z;
                    
                    do
                    {
                        z = random.uniform_float();
                    }
                    while ( ( ( z < shift ) xor layer ) xor sign );
                
                    particle_idx     [ prefix + i ] = -1u; // this means that this particle should not be loaded 
                    particle_position[ prefix + i ] = math::vektor( random.uniform_float() - 0.5f, random.uniform_float() - 0.5f, z - 0.5f ) + pos;
                    particle_velocity[ prefix + i ] = random.maxwell_boltzmann() * gpu_const::parameters.thermal_velocity;
                }
            }
        }
        
        __syncwarp(); 

        for ( uint32_t i = threadIdx.x; i < sum; i += 32 ) // load the entries of the lookup table uniformly without binding threads to SRD cells
            if ( particle_idx[ i ] != -1u )
                particle_idx[ i ] = gpu_utilities::texture_load( uniform_list.data() + particle_idx[ i ] );

        __syncwarp(); 

        for ( uint32_t i = threadIdx.x; i < sum; i += 32 ) // load the SRD fluid particles uniformly without binding threads to SRD cells
        {
            if ( particle_idx[ i ] != -1u )
            {
                if ( particle_idx[ i ] < regular_particles )
                {
                    auto particle = gpu_utilities::texture_load( particles.data() + particle_idx[ i ] ); 
                   
                    particle_position[ i ] = particle.position;
                    particle_velocity[ i ] = particle.velocity;
                }
            }
        } 
        
        if ( thread_active ) // SRD collision step, each thread one SRD cell 
        {
            if ( n_particles > 1 ) 
            {
                for ( uint32_t i = 0; i < n_particles; ++i )
                    cell.unlocked_add( { 0, 0, particle_position[ prefix + i ], particle_velocity[ prefix + i ] } );
            
                cell.average();
                random.sync_phase();

                auto  axis = random.unit_vektor();
            
                for ( uint32_t i = 0; i < n_particles; ++i ) // rotation step:
                {
                    auto v      = particle_velocity[ prefix + i ] - cell.mean_velocity;
                    auto v_para = axis * ( v.dot_product( axis ) );
                    auto v_perp = v - v_para;

                    particle_velocity[ prefix + i ] = v_para + cos_alpha * v_perp + sin_alpha * v_perp.cross_product( axis );
                }
           
                for ( uint32_t i = 0; i < n_particles; ++i ) // finilize step:
                {
                    particle_velocity[ prefix + i ] += cell.get_correction( particle_position[ prefix + i ] );  
                    particle_position[ prefix + i ]  = apply_periodic_boundaries( particle_position[ prefix + i ] - grid_shift );
                    particle_velocity[ prefix + i ].x += drag;
                }
            } 
        }
        else
            cell_idx -= stride;
            
        cell_idx = __shfl_sync( 0xFFFFFFFF, cell_idx, threadIdx.x + active_cells ); // cyclic shift so that theads' cell_index remains uniform
       
        for ( uint32_t i = threadIdx.x; i < sum; i += 32 ) // write the SRD fluid particles to memory uniformly without binding threads to SRD cells 
        {
            if ( particle_idx[ i ] != -1u )
            {
                if ( particle_idx [ i ] < regular_particles )
                {
                    auto cidx = mpc_cells.get_index( particle_position[ i ] );
                    particles[ particle_idx[ i ] ] = { 0, 0, particle_position[ i ], particle_velocity[ i ], cidx };
                }
            }
        } 
    } 
    
    generator[ blockIdx.x * blockDim.x + threadIdx.x ] = random;
}

namespace sampling
{
    /**
     *  @brief 1st step to compute the fluid state on the grid of the SRD collision cell 
     */
    __global__ void add_particles( parameter_set parameters, gpu_volumetric_container< mpc_cell_type > mpc_cells, async_vector< particle_type > particles )
    {
        size_t idx       = blockIdx.x * blockDim.x + threadIdx.x,
               stride    = blockDim.x * gridDim.x;

        for ( auto end = particles.size(); idx < end; idx += stride )
        {
            auto particle = particles[ idx ];
            mpc_cells[ particle.position ].add_reduce_only( particle.velocity );
        }
    }

    /**
     *  @brief 2st step to compute the fluid state on the grid of the SRD collision cell 
     */
    __global__ void average_cells( gpu_volumetric_container< mpc_cell_type > mpc_cells )
    {
        size_t idx    = blockIdx.x * blockDim.x + threadIdx.x,
               stride = blockDim.x * gridDim.x; 
        
        for ( ; idx < mpc_cells.size(); idx += stride )
            mpc_cells[ idx ].average_reduce_only(); 
    }

    /**
     *  @brief Store one timestep, either to initialize a time average or to store just one time step
     */
    __global__ void snapshot( parameter_set parameters, gpu_volumetric_container< mpc_cell_type > mpc_cells, async_vector< fluid_state_type > cell_states )
    {
        size_t idx    = blockIdx.x * blockDim.x + threadIdx.x,
               stride = blockDim.x * gridDim.x; 

        for ( ; idx < mpc_cells.size(); idx += stride )
        {
            auto cell_centre = mpc_cells.get_position( idx ); 
            cell_states[ idx ].density       = mpc_cells[ cell_centre ].density;
            cell_states[ idx ].mean_velocity = mpc_cells[ cell_centre ].mean_velocity;
        }
    }
    
    /**
     *  @brief Add more data when performing an average  
     */
    __global__ void accumulate( parameter_set parameters, gpu_volumetric_container< mpc_cell_type > mpc_cells, async_vector< fluid_state_type > cell_states )
    {
        size_t idx    = blockIdx.x * blockDim.x + threadIdx.x,
               stride = blockDim.x * gridDim.x; 
        
        for ( ; idx < mpc_cells.size(); idx += stride )
        {
            auto cell_centre = mpc_cells.get_position( idx ); 
            cell_states[ idx ].density       += mpc_cells[ cell_centre ].density;
            cell_states[ idx ].mean_velocity += mpc_cells[ cell_centre ].mean_velocity;
        }
    }
    
    /**
     *  @brief Finish performing the average  
     */
    __global__ void finish( parameter_set parameters, gpu_volumetric_container< mpc_cell_type > mpc_cells, async_vector< fluid_state_type > cell_states )
    {
        size_t idx    = blockIdx.x * blockDim.x + threadIdx.x,
               stride = blockDim.x * gridDim.x; 
        
        math::float_type inverse = 1.0 / parameters.average_samples; 

        for ( ; idx < mpc_cells.size(); idx += stride )
        {
            auto cell_centre = mpc_cells.get_position( idx );

            cell_states[ idx ].density       += mpc_cells[ cell_centre ].density;
            cell_states[ idx ].mean_velocity += mpc_cells[ cell_centre ].mean_velocity;
            cell_states[ idx ].density       *= inverse;
            cell_states[ idx ].mean_velocity *= inverse;
        }
    }
}
