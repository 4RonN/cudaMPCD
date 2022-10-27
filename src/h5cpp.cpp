
#include <H5Cpp.h>
#include <vector>

#include "h5cpp.hpp"

namespace h5
{
    struct file::hidden
    {
        H5::H5File data_file;
    }; 

    file::file( std::string file_name_, std::string open_mode ) : file_name( file_name_), impl( new hidden{ H5::H5File( file_name_.c_str(), open_mode == "rw" ? 
                                                     H5F_ACC_TRUNC : open_mode == "a" ? H5F_ACC_RDWR : 1 ) } ) { }
    
    file::~file() { impl->data_file.close(); } 

    void file::create_group( std::string path )
    {
        impl->data_file.createGroup( path.c_str() );
    }
        
    void file::flush()
    {
        impl->data_file.close(); 
        impl->data_file.openFile( file_name, H5F_ACC_RDWR );
    }
        
    // hier muss man jeweils entsprechend eine instanziierung fordern!!
    template< typename T >
    void file::write_float_data( std::string path, T const* data, std::initializer_list< size_t > size )
    {
        std::vector< hsize_t > dim( std::rbegin(size), std::rend(size) );
        H5::DataSet dataset = impl->data_file.createDataSet( path.c_str(), H5::PredType::NATIVE_FLOAT, H5::DataSpace( dim.size(), dim.data() ) );
      
        dataset.write( data, H5::PredType::NATIVE_FLOAT );
    }
    template
    void file::write_float_data<float>( std::string, float const*, std::initializer_list< size_t > size ); // template instance

    template< typename T >
    void file::write_float_data( std::string path, std::vector< T > const& data )
    {
        write_float_data< T >( path, data.data(), { data.size() } );
    }
    template
    void file::write_float_data<float>( std::string, std::vector< float > const& ); // template instance
        
    template< typename T >
    void file::read_float_data( std::string path, std::vector< T > &data )
    {
        auto dataset = impl->data_file.openDataSet( path.c_str() );

        auto dataspace = dataset.getSpace();
        hsize_t dims_out[2];
        dataspace.getSimpleExtentDims( dims_out, NULL );

        hsize_t dim[] = { dims_out[0] };
        data.resize( dim[0] );
        dataset.read( data.data(), H5::PredType::NATIVE_FLOAT, H5::DataSpace( 1, dim ), H5::DataSpace( 1, dim ) );
    }
    template
    void file::read_float_data< float >( std::string path, std::vector< float > &data );
    
    template< typename T >
    std::vector< T > file::read_float_data( std::string path )
    {
        std::vector< T > data;
        read_float_data( path, data );
        return data; 
    }
    template
    std::vector< float > file::read_float_data< float >( std::string path  );
}
