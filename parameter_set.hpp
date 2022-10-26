#ifndef __PARAMETER_SET_HPP
#define __PARAMETER_SET_HPP

#include <cuda_runtime_api.h>

#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>

#include <cassert>
#include <cmath>

#include <algorithm>
#include <array>
#include <functional>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "cuda_allocator.hpp" 
#include "vektor_type.hpp"

enum experiment_type 
{
    standart,
    channel,
};
    
/**
 *  @brief parameter_set and settings that are handed through the program.
 */
struct parameter_set 
{
    using int_vektor = math::int_vektor;
    using vektor     = math::vektor;
    
    experiment_type experiment;

    int_vektor   periodicity,
                 domain_periodicity, 
                 direction_is_split;
    vektor       volume_size;

    unsigned int N, // number of particles
                 n; // number of particles per cell
                 
    float        delta_t, // SRD time step 
                 temperature,
                 thermal_velocity,
                 drag; 
                       
    unsigned int equilibration_steps,
                 steps,
                 sample_every,
                 average_samples;
    bool         sample_fluid;
    bool         write_backup,
                 read_backup;

    size_t       cuda_device,
                 block_size,
                 block_count,
                 multiprocessors,
                 shared_bytes,
                 sharing_blocks,
                 internal_step_counter,
                 resort_rate;

    float        thermal_sigma;

#if ( defined __CUDA_ARCH__ )
    char         placeholder[ 2 * sizeof( std::string ) ];
#else
    std::string  output_directory,
                 load_directory;

    /**
     *  @brief Reads the program start arguments and depending on them, an input file. 
     */
    static parameter_set read_parameter_file( std::string const& file_name, bool read_only=false )
    {
        std::string   parameter_name, 
                      branch;

        //std::unordered_map< std::string, std::any > parameter_name_map;
       
        std::ifstream input_file( file_name ); 
        
        if( !input_file ) 
        { 
            std::cout << "no such input_file: " << file_name << std::endl;
            exit(0); 
        }

        parameter_set parameters;
        
        while ( parameter_name != "general" )  
            input_file >> parameter_name;

        std::getline( input_file, parameter_name );

        input_file >> parameter_name >> parameters.output_directory;
        
        input_file >> parameter_name >> branch;
        if ( branch == "standart" )
            parameters.experiment = standart;    

        input_file >> parameter_name >> parameters.volume_size.x >> parameters.volume_size.y >> parameters.volume_size.z;
        input_file >> parameter_name >> parameters.periodicity.x >> parameters.periodicity.y >> parameters.periodicity.z;

        parameters.volume_size -= ( parameters.periodicity - 1 ) * 2; // add wall layer
       
        // skip 3 lines: 
        std::getline( input_file, parameter_name ); std::getline( input_file, parameter_name ); std::getline( input_file, parameter_name ); 
       
        // fluid parameters:

        input_file >> parameter_name >> parameters.n; 
        input_file >> parameter_name >> parameters.temperature;
        input_file >> parameter_name >> parameters.delta_t;
        input_file >> parameter_name >> parameters.drag;
        
        std::getline( input_file, parameter_name ); std::getline( input_file, parameter_name ); std::getline( input_file, parameter_name ); 
        
        // simulation parameters:

        input_file >> parameter_name >> parameters.equilibration_steps;
        input_file >> parameter_name >> parameters.steps;
        input_file >> parameter_name >> parameters.sample_every;
        input_file >> parameter_name >> parameters.average_samples;
        input_file >> parameter_name >> branch; parameters.write_backup             = ( branch == "yes" );
        input_file >> parameter_name >> branch; parameters.read_backup              = ( branch == "yes" );

        if ( branch != "yes" and branch != "no" ) // load status from directory
        {
           parameters.read_backup = true;
           parameters.load_directory = branch;
        }
        else 
           parameters.load_directory = "./";

        input_file >> parameter_name >> parameters.cuda_device;

        // ----------- implementation hints etc:

        if ( parameters.experiment == channel )
        {
            std::cout << "implementation of ghost particles not complete for this geometry. program end..." << std::endl;
            std::exit( 0 );
        }           

        // -----------

        if ( not read_only ) // create simulation output folder given in the input file
        {
            std::string code_id = "";

            std::ifstream commit_info;
            commit_info.open( "commit_info" );

            auto ptr = get_current_dir_name();
            auto dp  = opendir( ( ptr + std::string("/") + parameters.output_directory ).c_str() );
            free( ptr );

            if ( dp )
            {
                std::cout << "using directory " << parameters.output_directory << " ..." << std::endl;

                closedir( dp );  
                if ( false )
                {
                    std::cerr << "overwrite existing data? y/n: " << std::flush;
                    std::string s;
                    std::cin >> s;
                    if ( s[0] == 'n' ) 
                        exit( 1 );
                }   

                if ( 0 != chdir( ( get_current_dir_name() + std::string("/") +  parameters.output_directory ).c_str() ) )
                {
                    std::cout << "problem changinng into the requested directory... exiting..." << std::endl;
                    std::exit( 0 );
                }
            }
            else
            {
                std::cout << "creating new directory " << parameters.output_directory << " ..." << std::endl;

                int pos; 
                for (;;) 
                {
                    pos = parameters.output_directory.find( "/" );
                    if( pos == -1 ) 
                       break;

                    std::string folder = parameters.output_directory.substr( 0, pos );
                    if ( 0 != mkdir( ( get_current_dir_name() + std::string("/") + folder ).c_str(), 0777 ) ) { }
                    if ( 0 != chdir( ( get_current_dir_name() + std::string("/") + folder ).c_str() ) )
                    {
                        std::cout << "problem cd-ing into the requested directory... exiting..." << std::endl;
                        std::exit( 0 );
                    }
                    parameters.output_directory = parameters.output_directory.substr( pos+1 );
                }
                
                if ( 0 != mkdir( ( get_current_dir_name() + std::string("/") + parameters.output_directory ).c_str(), 0777 ) ) { }
                if ( 0 != chdir( ( get_current_dir_name() + std::string("/") + parameters.output_directory ).c_str() ) )
                {
                    std::cout << "problem cd-ing into the requested directory... exiting..." << std::endl;
                    std::exit( 0 );
                }
            }
            
            input_file.clear();
            input_file.seekg( 0 );

            // add info file to the new directory

            std::ofstream info_file( "simulation.info" );
            
            /*        
            if ( not commit_info )
            {
                std::cout << "commit_info file missing! terminating here!" << std::endl;
                exit( 0 );
            }

            while ( !commit_info.eof() )
            {
                char c;
                commit_info.get( c );
                info_file << c;
            }*/

            info_file << "simulation parameters where: \n";

            while ( !input_file.eof() )
            {
                char c;
                input_file.get( c );
                info_file << c;
            }
        }
            
        parameters.N = parameters.n * parameters.volume_size.diagonal_product();
        parameters.thermal_velocity = std::sqrt( parameters.temperature );
    
        cuda_set_device( parameters.cuda_device );

        // query device properties to decide kernel launch layouts

        cudaDeviceProp properties;
        cudaGetDeviceProperties( &properties, parameters.cuda_device );
        parameters.multiprocessors = properties.multiProcessorCount;
        parameters.block_count     = properties.multiProcessorCount;
        parameters.shared_bytes    = properties.sharedMemPerMultiprocessor;
        cudaDeviceSetSharedMemConfig( cudaSharedMemBankSizeFourByte );

        if ( properties.major < 7 ) 
        { 
            std::cout << "cuda: pascal architecture ... " << std::endl;
            
            if ( parameters.block_count > 28 )    // p100
            {
                parameters.block_size = 64;            
                parameters.block_count *= 4;  // concurrent kernels  
                
                parameters.shared_bytes   = properties.sharedMemPerMultiprocessor / ( 2 * 4 );
                parameters.sharing_blocks = properties.multiProcessorCount * 2 * 4;  // 2 warps per SM, 4x occupancy.
            }
            else if ( parameters.block_count > 14 )         // gtx 1070 etc.
            {
                parameters.block_size = 128;             
                parameters.block_count *= 4;
                
                parameters.shared_bytes   = properties.sharedMemPerMultiprocessor / ( 4 * 4 );
                parameters.sharing_blocks = properties.multiProcessorCount * 4 * 4;  // 4 warps per SM, 4x occupancy.
            }
            else                            // gtx 960
            {
                parameters.block_size = 64;
                parameters.block_count *= 2; 
                
                parameters.shared_bytes   = properties.sharedMemPerMultiprocessor / ( 2 * 2 );
                parameters.sharing_blocks = properties.multiProcessorCount * 2 * 2;  // 2 warps per SM, 2x occupancy.
            }
        }
        else // Turing:
        {
            std::cout << "cuda: turing architecture ... " << std::endl;

            parameters.block_size = 64;
            parameters.block_count *= 4; 
            
            parameters.shared_bytes   = properties.sharedMemPerMultiprocessor / ( 2 * 4 );
            parameters.sharing_blocks = properties.multiProcessorCount * 2 * 4;  // 4 warps per SM, 4x occupancy.
        }

        return parameters;
    }
#endif
};
    
//parameter_set* parameter_set::parameters;

#endif // __PARAMETER_SET_HPP
