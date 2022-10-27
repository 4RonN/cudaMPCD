
#include <string> 

#include "simulation_context.hpp"
#include "gpu_functions.hpp"
#include "h5cpp.hpp"
#include "gpu_constants.hpp"

/**
 *   @brief Initialize the GPU and the SRD fluid 
 */
simulation_context::simulation_context( parameter_set const& parameters ) :
                                                                       particles( static_cast< size_t >( parameters.volume_size.diagonal_product() * parameters.n ) ),
                                                                       particles_sorted( 10 ), 
                                                                       mpc_cells( parameters.volume_size ),
                                                                       cell_states( static_cast< size_t >( parameters.volume_size.diagonal_product() ) ),
                                                                       uniform_list( mpc_cells.size() * 4 * parameters.n ),
                                                                       uniform_counter( mpc_cells.size() ),
                                                                       internal_step_counter( 0 ),
                                                                       resort_rate( 100 )
{

    // seed the parallel random number generators
    generator.alloc( parameters.block_size * parameters.sharing_blocks );
    {    
        async_vector< uint64_t > seed( 2 * generator.size() );                                                                                         

        for ( auto& item : seed )                                                                                                                      
            item = std::hash< uint64_t >()( random() );

        seed.push();                                                                                                                                   
        initialize::random_number_generators <<< parameters.block_count, parameters.block_size >>> ( generator, seed );                                                     
        error_check( "initialise_generators" );
    }

    // load members into GPU's constant memory:
    cudaMemcpyToSymbol( gpu_const::parameters, &( parameters ), sizeof( parameter_set ) ); 
    error_check( "symbol parameters" );
    cudaMemcpyToSymbol( gpu_const::particles, &( particles ), sizeof( decltype( particles ) ) ); 
    error_check( "symbol particles" );
    cudaMemcpyToSymbol( gpu_const::generator, &( generator ), sizeof( decltype( generator ) ) ); 
    error_check( "symbol particles" );
    cudaMemcpyToSymbol( gpu_const::mpc_cells, &mpc_cells, sizeof( decltype( mpc_cells ) ) ); 
    error_check( "symbol mpc_cells" );
    cudaMemcpyToSymbol( gpu_const::uniform_list, &uniform_list, sizeof( decltype( uniform_list ) ) ); 
    error_check( "symbol uniform_list" );
    cudaMemcpyToSymbol( gpu_const::uniform_counter, &uniform_counter, sizeof( decltype( uniform_counter ) ) ); 
    error_check( "symbol uniform_counter" );
    
    // initialize SRD fluid particles 
    grid_shift = { random.uniform_float() - float_type( 0.5 ), 
                   random.uniform_float() - float_type( 0.5 ), 
                   random.uniform_float() - float_type( 0.5 )  };

    initialize::distribute_particles <<< parameters.block_count, parameters.block_size >>> ( particles, mpc_cells, generator, parameters, grid_shift, 0 ); 
    error_check( "distribute_particles" );

    cudaDeviceSynchronize();
    std::cout << "gpu initialized ..." << std::endl;

    // create output file           
    h5::file data( "simulation_data.h5", "rw" );
    data.create_group( "fluid" );
}

/**
 *   @brief Perform entire SRD simulation step. 
 */
void simulation_context::perform_step( parameter_set const& parameters )
{
    // SRD steps: 
    translation_step( parameters );
    collision_step( parameters );
   
    // sort praticles array to improve memory access times 
    if ( ( internal_step_counter++ % resort_rate ) == 0 )
    {
        particles.pull();
        std::sort( particles.begin(), particles.end(), [] ( auto a, auto b ) { return a.cell_idx < b.cell_idx; } ); 
        particles.push();
    }
}

/**
 *   @brief Data io, either creating snapshots or averaging and writing to disk. 
 */
void simulation_context::write_sample( size_t step, parameter_set parameters )
{
    probing_type probe = what_to_do( step, parameters );
    
    mpc_cells.set( 0 ); // clear cells.
    sampling::add_particles <<< parameters.block_count, parameters.block_size >>> ( parameters, mpc_cells, particles ); 
    error_check( "add_particles_reduce_only" );
        
    sampling::average_cells <<< parameters.block_count, parameters.block_size >>> ( mpc_cells );
    error_check( "average_cells_reduce_only" );

    switch ( probe )
    {
        case snapshots_only:
            sampling::snapshot   <<< parameters.block_count, parameters.block_size >>> ( parameters, mpc_cells, cell_states );
            error_check( "snapshot" );
            break;
        case start_accumulating:
            sampling::snapshot   <<< parameters.block_count, parameters.block_size >>> ( parameters, mpc_cells, cell_states );               
            error_check( "snapshot" );
            break;                                                                                
        case accumulate:                                                                          
            sampling::accumulate <<< parameters.block_count, parameters.block_size >>> ( parameters, mpc_cells, cell_states );               
            error_check( "accumulate" );
            break;
        case finish_accumulation:
            sampling::finish     <<< parameters.block_count, parameters.block_size >>> ( parameters, mpc_cells, cell_states );               
            error_check( "finish" );
            break;
    }
        
    if ( probe == finish_accumulation or probe == snapshots_only )
    { 
        cell_states.pull();
        h5::file data( "simulation_data.h5", "a" ); // append 
        data.write_float_data( std::string( "fluid/" ) + std::to_string( step ), reinterpret_cast< float* >( cell_states.data() ), 
                        { 4, static_cast< size_t >( parameters.volume_size.x ), static_cast< size_t >( parameters.volume_size.y ), 
                             static_cast< size_t >( parameters.volume_size.z ) } );
    } 
}

void simulation_context::write_backup_file()
{
    // TODO
}

/**
 *   @brief Perform SRD streaming step
 */
void simulation_context::translation_step( parameter_set const& parameters )
{
    grid_shift = { random.uniform_float() - math::float_type( 0.5 ), 
                   random.uniform_float() - math::float_type( 0.5 ), 
                   random.uniform_float() - math::float_type( 0.5 )  };

    uniform_counter.set( 0 );
    translate_particles <<< parameters.block_count, parameters.block_size >>> ( particles, mpc_cells, generator, parameters, grid_shift, 
                                                                                uniform_counter, uniform_list );
    error_check( "translate_particles" );
}

/**
 *   @brief Perform SRD collision step
 */
void simulation_context::collision_step( parameter_set const& parameters )
{
    mpc_cells.set( 0 );
    
    switch ( parameters.algorithm )
    {
        case srd: 
            srd_collision <<< parameters.sharing_blocks, 32, parameters.shared_bytes >>> ( particles, mpc_cells, generator.data(), grid_shift, 
                                                                                           uniform_counter, uniform_list, parameters.shared_bytes );
            break;
        case extended:
            extended_collision <<< parameters.sharing_blocks, 32, parameters.shared_bytes >>> ( grid_shift, uniform_counter, uniform_list, parameters.shared_bytes );
            break;
        default:
            break;
    }

    error_check( "collision_step" );
}
