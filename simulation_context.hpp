#ifndef __SIMULATION_CONTEXT_HPP
#define __SIMULATION_CONTEXT_HPP

#include "mpc_cell_type.hpp"
#include "gpu_arrays.hpp"
#include "gpu_container_classes.hpp"
#include "gpu_random.hpp"
#include "parameter_set.hpp"
#include "particle_type.hpp"
#include "probing.hpp"

struct simulation_context 
{
    using vektor     = math::vektor;
    using float_type = math::float_type;
        
    async_vector< particle_type >                   particles;   // SRD fluid particles
    gpu_vector< particle_type >                     particles_sorted; // use for gpu sorting later
        
    vektor                                          grid_shift;  // SRD grid shift
        
    gpu_volumetric_container< mpc_cell_type >       mpc_cells;   // SRD cell storage
    async_vector< fluid_state_type >                cell_states; // for averaging over the fluid state

    // The indices for fluid particles are stored in a lookup table for the collision step.
    // This optimizes the data througput, because particles can be stored in shared memory 
    // and only need to be loaded once: 
    gpu_vector< uint32_t >                          uniform_list,    // the index lookup 
                                                    uniform_counter; // next free table entry, used with atomicAdd. 
        
    gpu_vector< xoshiro128plus >                    generator;  // random number generators for the gpu
    xorshift1024star                                random;     // random number generatofor the cpu

    // routines:
    simulation_context( parameter_set const& );  // initialization
    void perform_step( parameter_set const& );   // perform one entire simulation step. 

    // data io:
    void write_sample( size_t step, parameter_set parameters );
    void write_backup_file();

    private:
   
    // To furthe optimize memory loading, the particle array is sorted according to the SRD cell-index.
    // This enables array striding, ie. coalesced memory loading:
    size_t internal_step_counter, resort_rate;

    void translation_step( parameter_set const& );  // SRD streaming step
    void collision_step( parameter_set const& );    // SRD collision step
};

#endif // __SIMULATION_CONTEXT_HPP
