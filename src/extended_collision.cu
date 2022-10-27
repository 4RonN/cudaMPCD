
#include "extended_collision.hpp"
#include "parameter_set.hpp"
#include "mechanic.hpp"
#include "probing.hpp"
#include "gpu_constants.hpp"

/**
 *  @brief grouping into sub-groups by random plane:
 *          (i)   momentum transpher between groups
 *          (ii)  thrermalised in soubgroups
 *          (iii) angular momentum conservation
 */
__global__ __launch_bounds__( 32, 8 ) void extended_collision( math::vektor grid_shift, gpu_vector< uint32_t > uniform_counter, 
                                                               gpu_vector< uint32_t > uniform_list, uint32_t const shared_bytes )
{
    extern __shared__ uint32_t shared_mem [];
    
    // asign shared memory for particle positions and velocities 
    uint32_t const max_particles      = shared_bytes / ( sizeof( uint32_t ) + 2 * sizeof( math::vektor ) ); // this is the per particle memory
    uint32_t     * particle_idx       = shared_mem;                                                         // storing the indices of loaded particles
    math::vektor * particle_position  = reinterpret_cast< math::vektor* >( particle_idx + max_particles );  // 1st vector
    math::vektor * particle_velocity  = particle_position + max_particles;                                  // 2nd vector 
                                       
    auto           random              = gpu_const::generator[ blockIdx.x * blockDim.x + threadIdx.x ];
    particle_type* ghost_particles     = nullptr; // override this to add ghost particles!
    uint32_t const cell_lookup_size    = uniform_list.size() / uniform_counter.size(); 
    auto     const regular_particles   = gpu_const::particles.size();
    auto           shift               = fabs( grid_shift.z );
    unsigned       sign                = grid_shift.z > 0;
    auto           drag                = gpu_const::parameters.drag * gpu_const::parameters.delta_t;
    float const    scale               = 0.01f; // collision probability scale

    for ( int cell_idx = blockIdx.x * blockDim.x + threadIdx.x,
              stride   = blockDim.x * gridDim.x,
              end      = gpu_const::mpc_cells.size(); 
              __any_sync( 0xFFFFFFFF, cell_idx < end ); cell_idx += stride ) // iterate in complete warps, overhanging threads still need to join. 
    {
        // ~~~ setup & load particles:

        int  n_particles  = ( cell_idx < end ) ? min( cell_lookup_size, uniform_counter[ cell_idx ] ) : 0; // read lookup size
        bool layer        = ( gpu_const::mpc_cells.get_z_idx( cell_idx ) == ( gpu_const::parameters.volume_size.z - ( sign ? 1 : 2 ) ) ); // wall layer?
        bool add_ghosts   = ( not gpu_const::parameters.periodicity.z ) 
                                 and ( ( gpu_const::mpc_cells.get_z_idx( cell_idx ) == sign ) or layer ); // wall layer?
        int  added_ghosts = {};

        random.sync_phase();
        if ( add_ghosts  )
            for ( int i = 0; i < gpu_const::parameters.n; ++i )
                if ( ( ( random.uniform_float() > shift ) xor layer ) xor sign ) 
                    ++added_ghosts;

        n_particles += added_ghosts;

        // arrange thread groups to maximise shared memmory usage. if only particles of a few cells fit in memory, work in groups.  

        int       prefix         = gpu_utilities::warp_prefix_sum( n_particles ); // where does the threads storage start?
        // how many cells' particles fit into shared memory?
        int const active_cells   = min( 8, __popc( __ballot_sync( -1u, prefix + n_particles < max_particles and cell_idx < gpu_const::mpc_cells.size() ) ) );  
        int const group_size     = 32 / active_cells;
        int const sum            = __shfl_sync( -1u, prefix + n_particles, active_cells - 1 ); // total number of particles of the used number of cells.  
        int       group_cell_idx = cell_idx;
      
        if ( group_size > 1 ) // communicate group variables betweeen grouped threads.
        {
            auto group_root = threadIdx.x / group_size;

            n_particles    = __shfl_sync( 0xFFFFFFFF, n_particles,  group_root ); 
            layer          = __shfl_sync( 0xFFFFFFFF, layer,        group_root );
            added_ghosts   = __shfl_sync( 0xFFFFFFFF, added_ghosts, group_root ); 
            prefix         = __shfl_sync( 0xFFFFFFFF, prefix,       group_root ); 
            group_cell_idx = __shfl_sync( 0xFFFFFFFF, cell_idx,     group_root ); 

            add_ghosts = ( added_ghosts != 0 );
        }

        // decide if threads are left over and deactivate them:
        bool     const thread_active = prefix + n_particles < max_particles and group_cell_idx < gpu_const::mpc_cells.size(); 
        uint32_t const mask          = __ballot_sync( 0xFFFFFFFF, thread_active ); // mask of participating threads for following __shfl operations.
        
        // define initial coordinate offset to improve calculation of the moment of intertia tensor as the cell centre.
        auto offset = gpu_const::mpc_cells.get_position( group_cell_idx ); 
        
        if ( thread_active )
        {
            for ( int i = threadIdx.x % group_size, end = n_particles - added_ghosts; i < end; i += group_size ) // prepare lookup table: which indices will be loaded?
                particle_idx[ prefix + i ] = group_cell_idx + i * gpu_const::mpc_cells.size();
        
            if ( add_ghosts ) // in wall layers: add random "ghost" particles
            {    
                auto z_scale = sign ? ( layer ? 1 - shift : shift ) : ( layer ? shift : 1 - shift );

                for ( int i = n_particles - added_ghosts + threadIdx.x % group_size; i < n_particles; i += group_size )
                {
                    float z = z_scale * random.uniform_float();
                    
                    particle_idx     [ prefix + i ] = -1u;
                    particle_velocity[ prefix + i ] = random.maxwell_boltzmann() * gpu_const::parameters.thermal_velocity; 
                    particle_position[ prefix + i ] = math::vektor( random.uniform_float() - 0.5f, random.uniform_float() - 0.5f, layer ? 0.5f - z : z - 0.5f ) + offset;
                }
            }
        }
       
        __syncwarp(); // dependencies in accesses to shared mem have to be synced -> memory fence...

        for ( int i = threadIdx.x; i < sum; i += 32 ) // read the lookup table of the MPCD cells in use.
            if ( particle_idx[ i ] != -1u )
                particle_idx[ i ] = __ldg( uniform_list.data() + particle_idx[ i ] ); // using texture load path 
       
        __syncwarp();

        for ( int i = threadIdx.x; i < sum; i += 32 ) // now transfer the particles into shared mem based on the lookup table
        {
            if ( particle_idx[ i ] != -1u )
            {
                // using texture load path __ldg() 
                auto particle = gpu_utilities::texture_load( ( particle_idx[ i ] < gpu_const::particles.size() ? gpu_const::particles.data() : 
                                                                                                                 ghost_particles ) + particle_idx[ i ] ); 
                particle_position[ i ] = particle.position;
                particle_velocity[ i ] = particle.velocity;
            }
        }

        __syncwarp(); // ~~~ apply collision rule:

        if ( thread_active ) 
        {
            random.sync_phase();

            #if 1 // discrete axis set or continuous random vector

                int constexpr steps = 4; // discretization. careful: avoid overweight of theta = 0 pole with step phi cases...
               
                float theta, phi; // chose discretized random direction:
                {
                    int select  = gpu_utilities::group_share( random.uniform_int( 0, steps * ( steps - 1 ) ), mask, group_size );
                    int phi_i   = select % steps + 1,
                        theta_i = select / steps + 1; 
               
                    if ( ( theta_i % 2 ) and ( phi_i % 2 ) ) // edges
                        theta = theta_i == 1 ? 0.95531661812450927816f : float( M_PI ) - 0.95531661812450927816f; 
                    else 
                        theta = theta_i * ( float( M_PI ) / steps );

                    phi = phi_i * ( float( M_PI ) / steps );
                }
                math::vektor axis  = { __sinf( theta ) * __cosf( phi ), // transfer from spherical coordinates to cartesian.
                                       __sinf( theta ) * __sinf( phi ),
                                       __cosf( theta ) };

                if ( gpu_utilities::group_share( random.uniform_int( 0, 1 ), mask, group_size ) )
                    axis = -axis;
            #else           
                math::vektor axis = gpu_utilities::group_share( random.unit_vektor(), mask, group_size );
            #endif

            float z_centre = gpu_const::mpc_cells.get_position( group_cell_idx ).z;
            math::vektor centre_of_mass = {},
                         mean_velocity  = {};

            bool constexpr conserve_L = true;
           
            // calculate cells' center of mass and mean velocity, iterate in thread groups: 
            for ( math::vektor* position = particle_position + prefix + threadIdx.x % group_size,
                              * velocity = particle_velocity + prefix + threadIdx.x % group_size,
                              * end      = particle_position + prefix + n_particles; 
                        position < end; 
                        position += group_size, velocity += group_size )
            { 
                centre_of_mass += ( *position - offset );
                mean_velocity  += *velocity;
            }
            
            // reduce in thread groups using the __shfl shuffle operations:
            centre_of_mass = gpu_utilities::group_sum( centre_of_mass, mask, group_size ) * ( float( 1 ) / n_particles ) + offset; 
            mean_velocity  = gpu_utilities::group_sum( mean_velocity,  mask, group_size ) * ( float( 1 ) / n_particles );

            // --------------- collision:

            // devide particles into the groups defined by the random axis and center of mass. 
            float    projection_0 = {}, // projection of the mean velocities in group0 along the axis
                     projection_1 = {};
            uint64_t group        = {}; // bitstring storing on which side particles lie. 
                
            for ( int i = threadIdx.x % group_size; i < n_particles; i += group_size ) 
            { 
                particle_position[ prefix + i ] -= centre_of_mass; // tranfer into local comoving coordinate system
                particle_velocity[ prefix + i ] -= mean_velocity;
                    
                uint64_t const side = static_cast< uint64_t >( particle_position[ prefix + i ].dot_product( axis ) < 0 ); 
                group   |= ( side << ( i / group_size ) ); // store on which side partile i lies.
                
                if ( side ) // we only need to consider one side, the other one can be derived from it.  
                    projection_0 += particle_velocity[ prefix + i ].dot_product( axis ); 
            }
            
            int size_0 = gpu_utilities::group_sum( __popc( group ), mask, group_size ); // size of group0
            int size_1 = n_particles - size_0;

            projection_0 = gpu_utilities::group_sum( projection_0, mask, group_size );
            projection_1 = -projection_0 / size_1;
            projection_0 =  projection_0 / size_0;

            // calculate the collision probability based on the dynamics of the particles
            #if 1 // saturate:
                float probability = 1 - __expf( scale * ( projection_1 - projection_0 ) * size_0 * size_1 );
            #else // cutoff: 
                float probability = scale * ( projection_0 - projection_1 ) * size_0 * size_1;
            #endif
            
            math::vektor   delta_L        = {}; // cells' change in angular momentum  
            bool           collide        = gpu_utilities::group_share( random.uniform_float(), mask, group_size ) < probability;  
            unsigned const collision_mask = __ballot_sync( mask, collide ); // which groups participate in calculating the collision?
         
            if ( collide )
            {
                float transfer_0 = projection_1 * float( size_1 ) / size_0, // momentum trasferred to the other side of the plane.
                      transfer_1 = projection_0 * float( size_0 ) / size_1;
                
                float mean_0 = {}, // we assign new random velocities in the groups, but have to remove their mean to assure momentum conservation. 
                      mean_1 = {};
                
                traegheitsmoment< float > I = {}; // moment of inertia tensor

                random.sync_phase();
                for ( int i = threadIdx.x % group_size; i < n_particles; i += group_size )
                {
                    bool const side     = ( group >> ( i / group_size ) ) & 0x1; 
                    auto const v_random = random.gaussianf() * gpu_const::parameters.thermal_velocity;
                    
                    ( side ? mean_0 : mean_1 )      += v_random;
                    delta_L                         += particle_position[ prefix + i ].cross_product( particle_velocity[ prefix + i ] );
                    particle_velocity[ prefix + i ] += axis * ( v_random - particle_velocity[ prefix + i ].dot_product( axis ) + ( side ? transfer_0 : transfer_1 ) ); 
                    
                    auto squares  = particle_position[ prefix + i ].scaled_with( particle_position[ prefix + i ] );
                    I            += symmetric_matrix< float > ( { squares.y + squares.z, squares.x + squares.z, squares.x + squares.y,
                                                                  -particle_position[ prefix + i ].x * particle_position[ prefix + i ].y,
                                                                  -particle_position[ prefix + i ].x * particle_position[ prefix + i ].z, 
                                                                  -particle_position[ prefix + i ].y * particle_position[ prefix + i ].z } );
                }
                                    
                auto v_mean_0 = axis * ( gpu_utilities::group_sum( mean_0, collision_mask, group_size ) / size_0 ), // random velocities' mean
                     v_mean_1 = axis * ( gpu_utilities::group_sum( mean_1, collision_mask, group_size ) / size_1 );
                
                for ( int i = threadIdx.x % group_size; i < n_particles; i += group_size )
                { 
                    particle_velocity[ prefix + i ] -= ( ( group >> ( i / group_size ) ) & 0x1 ) ? v_mean_0 : v_mean_1; // restore groups momentum conservation
                    delta_L -= particle_position[ prefix + i ].cross_product( particle_velocity[ prefix + i ] ); // sum up angular momentum change 
                }
            
                I       =     gpu_utilities::group_sum( I,       collision_mask, group_size ).inverse( 1e-5f );
                delta_L = I * gpu_utilities::group_sum( delta_L, collision_mask, group_size );
            }

            // --------------- end collision.
       
            for ( math::vektor* position = particle_position + prefix + threadIdx.x % group_size,
                              * velocity = particle_velocity + prefix + threadIdx.x % group_size,
                              * end      = particle_position + prefix + n_particles; 
                        position < end; 
                        position += group_size, velocity += group_size )
            {
                *velocity += mean_velocity; // restore momentum conservation
                if ( conserve_L )
                    *velocity -= position -> cross_product( delta_L ); // restore angular momentum conservation
                *position += centre_of_mass; // convert coordinate back into global coordinate system
            }
        }
        __syncwarp(); // ~~~ write particles back to ram:
       
        for ( int i = threadIdx.x; i < sum; i += 32 )
        {
            if ( particle_idx[ i ] != -1u )
            {
                particle_velocity[ i ].x += drag;

                auto apply_periodic_boundaries = [&] ( auto r ) // does not interfere with using walls...  
                { 
                    r.x = fmodf( r.x + 1.5f * gpu_const::parameters.volume_size.x, gpu_const::parameters.volume_size.x ) - gpu_const::parameters.volume_size.x * 0.5f;   
                    r.y = fmodf( r.y + 1.5f * gpu_const::parameters.volume_size.y, gpu_const::parameters.volume_size.y ) - gpu_const::parameters.volume_size.y * 0.5f;   
                    r.z = fmodf( r.z + 1.5f * gpu_const::parameters.volume_size.z, gpu_const::parameters.volume_size.z ) - gpu_const::parameters.volume_size.z * 0.5f;
                    return r; 
                };
                particle_position[ i ] = apply_periodic_boundaries( particle_position[ i ] - grid_shift );
                
                assert( particle_position[ i ].sane() );
                assert( particle_velocity[ i ].sane() );
               
                // ~~~ unified write back by switching pointer for usual / ghost particles:

                *( ( particle_idx[ i ] < gpu_const::particles.size() ? gpu_const::particles.data() : ghost_particles ) + particle_idx[ i ] ) 
                    = { static_cast< uint16_t >( 0 ), static_cast< uint16_t >( particle_idx [ i ] < gpu_const::particles.size() ? 0u : 1u ), 
                        particle_position[ i ], particle_velocity[ i ], }; 
            }
        }
        __syncwarp();

        if ( threadIdx.x >= active_cells ) // rewind skipped cells
            cell_idx -= stride;

        cell_idx = __shfl_sync( 0xFFFFFFFF, cell_idx, threadIdx.x + active_cells ); // uniform interation, "shift" processed cells out of the warp iteration.
    }

    gpu_const::generator[ blockIdx.x * blockDim.x + threadIdx.x ] = random; // save new state of the generators
} 
