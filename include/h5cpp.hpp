#ifndef _H5CPP_HPP
#define _H5CPP_HPP

#include <string>
#include <vector>
#include <memory>
#include <initializer_list>

namespace h5
{
    struct file
    {
        std::string file_name;

        file( std::string file_name_, std::string open_mode );
        ~file();
        
        void create_group( std::string path );
        void flush();

        template< typename T=float >
        void write_float_data( std::string path, std::vector< T > const& data );
        
        template< typename T >
        void write_float_data( std::string path, T const* data, std::initializer_list< size_t > size );

        template< typename T=float >
        void read_float_data( std::string path, std::vector< T > &data );

        template< typename T=float >
        std::vector< T > read_float_data( std::string path );
       
        private: 

        struct hidden;
        std::unique_ptr< hidden > impl; 
    };
}

#endif // _H5CPP_HPP
