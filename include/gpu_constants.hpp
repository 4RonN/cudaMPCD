#ifndef __GPU_CONSTANTS_HPP
#define __GPU_CONSTANTS_HPP

#include "parameter_set.hpp"
#include "gpu_container_classes.hpp"
#include "mpc_cell_type.hpp"
#include "particle_type.hpp"
#include "gpu_random.hpp"

// To avoid filling the kernel's stack with function arguments, large structs are placed in the GPU's constant memorgy.
// This is especially true for the parameter_set 

namespace gpu_const
{
    extern __constant__ parameter_set                             parameters;
    extern __constant__ gpu_volumetric_container< mpc_cell_type > mpc_cells;
    extern __constant__ async_vector< particle_type >             particles;
    extern __constant__ gpu_vector< xoshiro128plus >          generator;
    extern __constant__ gpu_vector< uint32_t >                    uniform_list,
                                                                  uniform_counter; 
}

#endif // __GPU_CONSTANTS_HPP
