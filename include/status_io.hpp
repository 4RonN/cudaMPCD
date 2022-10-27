#ifndef __STATUS_IO_HPP
#define __STATUS_IO_HPP

#include <csignal>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <functional>
#include <fstream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>

namespace status
{
    void write_time_file( int time_step )
    {
        static std::mutex safety_firscht; 
        std::lock_guard< std::mutex > lock( safety_firscht );
        std::ofstream( "last_timestep" ) << time_step;
    }
    
    int read_time_file( std::string dir="" )
    {
        int time_step = -1;
        
        auto time_file = std::ifstream( dir + "last_timestep" );
        if ( time_file )
            time_file >> time_step;
        else
        {
            std::cout << "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b"
                      << "no time file found! exiting..." << std::endl;
        }
        
        if ( time_step < 0 )
            std::exit( -1 );

        return time_step;
    }

    void report( char const *status )
    {
        printf( "%s\t%3.0f%%", status, 0.0f );
        std::cout << std::flush;
    }

    void update( size_t time_step, size_t of )
    {
        printf( "\b\b\b\b%3.0f%%", time_step / ( of / 100.0f ) );
        std::cout << std::flush;
    }

    void report_done()
    {
        printf( "\b\b\b\b%3.0f%%, done.\n", 100.0f );
        std::cout << std::flush;
    }

    volatile bool time_out = false;

    void set_flag( int )
    {
        time_out = true;
    }

    struct init
    {
        init() 
        { 
            signal( SIGUSR2, set_flag ); 
        }
    } 
    it; 
}

namespace clean_up 
{
    std::vector< std::function< void() >> actions = {};
    
    void register_action( std::function< void() >&& f )
    {
        actions.push_back( f );
    }
    
    void perform_actions()
    {
        std::cout << "\ntime out! performing backup..." << std::flush; 
       
        for ( auto& f : actions )
            f();
        
        std::cout << " bye!" << std::endl;
        
#ifdef __PARALLEL_HPP
        parallel::synchronize();      
#endif 

        std::this_thread::sleep_for( std::chrono::seconds( 5 ) );
        std::exit( 0 );
    }
}

#endif // __STATUS_IO_HPP
