
#include "parameter_set.hpp"
#include "vektor_type.hpp"
#include "gpu_container_classes.hpp"

namespace initialize 
{
    __global__ void srd_cells( parameter_set parameters, gpu_volumetric_container< mpc_cell_type > mpc_cells  );
    
    __global__ void random_number_generators( gpu_vector< xoshiro128plus > generator, async_vector< uint64_t > seed );
     
    __global__ void distribute_particles( async_vector< particle_type > particles, gpu_volumetric_container< mpc_cell_type > mpc_cells,
                                          gpu_vector< xoshiro128plus > generator, parameter_set parameters, 
                                          math::vektor grid_shift, uint32_t start );
}  //  namespace initialize

__global__ void translate_particles( async_vector< particle_type > particles, gpu_volumetric_container< mpc_cell_type > mpc_cells,
                                     gpu_vector< xoshiro128plus > generator, parameter_set parameters, math::vektor grid_shift,
                                     gpu_vector< uint32_t > uniform_counter, gpu_vector< uint32_t > uniform_list );

__global__ void srd_collision( async_vector< particle_type > particles, gpu_volumetric_container< mpc_cell_type > mpc_cells,
                               xoshiro128plus* generator, math::vektor grid_shift, gpu_vector< uint32_t > uniform_counter, 
                               gpu_vector< uint32_t > uniform_list, uint32_t const shared_bytes );

__global__ void extended_collision( math::vektor grid_shift, gpu_vector< uint32_t > uniform_counter, 
                                    gpu_vector< uint32_t > uniform_list, uint32_t const shared_bytes );

namespace sampling
{
    __global__ void add_particles( parameter_set parameters, gpu_volumetric_container< mpc_cell_type > mpc_cells, async_vector< particle_type > particles );
    __global__ void average_cells( gpu_volumetric_container< mpc_cell_type > mpc_cells );
    __global__ void snapshot( parameter_set parameters, gpu_volumetric_container< mpc_cell_type > mpc_cells, async_vector< fluid_state_type > cell_states );
    __global__ void accumulate( parameter_set parameters, gpu_volumetric_container< mpc_cell_type > mpc_cells, async_vector< fluid_state_type > cell_states );
    __global__ void finish( parameter_set parameters, gpu_volumetric_container< mpc_cell_type > mpc_cells, async_vector< fluid_state_type > cell_states );
}
