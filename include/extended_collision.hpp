#ifndef __EXTENDED_COLLISION_HPP
#define __EXTENDED_COLLISION_HPP

#include "vektor_type.hpp"
#include "gpu_container_classes.hpp"

__global__ void extended_collision( math::vektor grid_shift, gpu_vector< uint32_t > uniform_counter, 
                                    gpu_vector< uint32_t > uniform_list, uint32_t const shared_bytes );

#endif // __EXTENDED_COLLISION_HPP
