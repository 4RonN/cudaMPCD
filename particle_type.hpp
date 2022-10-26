#ifndef __PARTICLE_TYPE_HPP
#define __PARTICLE_TYPE_HPP

#include "vektor_type.hpp"

struct alignas( 16 ) particle_type
{
    using vektor     = math::vektor;
    
    uint16_t flags,
             cidx;
    vektor   position,
             velocity;
    uint32_t cell_idx;
    
    void write_binary( std::ofstream &stream ) const
    {
        stream.write( (char*) &position, sizeof( math::vektor ) * 2 );
    }

    void read_binary(  std::ifstream &stream )
    {
        stream.read( (char*) &position, sizeof( math::vektor ) * 2 );
    }
};

#endif // __PARTICLE_TYPE_HPP
