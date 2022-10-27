#ifndef __PROBING_HPP
#define __PROBING_HPP

#include "parameter_set.hpp"

enum probing_type { snapshots_only, start_accumulating, accumulate, finish_accumulation };

inline bool do_sampling( size_t const& time_step, parameter_set const& parameters )
{
    return !( time_step % parameters.sample_every );
} 

inline probing_type what_to_do( int const& time_step, parameter_set const& parameters )
{
    if ( parameters.average_samples == 1 or time_step == -1 ) 
        return snapshots_only;

    size_t at = ( ( time_step / parameters.sample_every ) % parameters.average_samples );

    if ( at == 1 )
        return start_accumulating;
    if ( at == 0 )
        return finish_accumulation;
    
    return accumulate;
}

#endif // __PROBING_HPP
