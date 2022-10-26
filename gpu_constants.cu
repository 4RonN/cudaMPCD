
#include "parameter_set.hpp"
#include "gpu_container_classes.hpp"
#include "particle_type.hpp"
#include "mpc_cell_type.hpp"
#include "gpu_random.hpp"

namespace gpu_const
{
    __constant__ parameter_set                             parameters;
    __constant__ gpu_volumetric_container< mpc_cell_type > mpc_cells;
    __constant__ async_vector< particle_type >             particles;
    __constant__ gpu_vector< xoshiro128plus >          generator;
    __constant__ gpu_vector< uint32_t >                    uniform_list,
                                                           uniform_counter; 
}
