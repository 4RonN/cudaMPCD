
#include "parameter_set.hpp"
#include "simulation_context.hpp"
#include "probing.hpp"
#include "status_io.hpp"

int main( int argc, char **argv )
{
    // --- initialize:
    if ( argc != 2 )
    {
        std::cout << "Please start the program with the name of an input file as parameter. The program is now exiting." << std::endl;
        return -1;
    }
   
    parameter_set const& parameters = parameter_set::read_parameter_file( std::string( argv[1] )); 

    simulation_context simulation( parameters );

    // --- simulation:
    if ( not parameters.read_backup )
    {
        status::report( "equilibration" );
        for ( size_t equilibration_step = 0; equilibration_step < parameters.equilibration_steps; ++equilibration_step )
        {
            simulation.perform_step( parameters );
            
            if ( !( equilibration_step % 100 ) )
                status::update( equilibration_step, parameters.equilibration_steps ); 
        }
        status::report_done();
    } 
        
    status::report( "sampling" );
    size_t start = ( parameters.read_backup and ( parameters.load_directory == "./" ) ) ? status::read_time_file( parameters.load_directory ) + 1: 0; 
    for ( size_t step = start; step < parameters.steps; ++step )
    {
        simulation.perform_step( parameters );

        if ( do_sampling( step, parameters ) )
            simulation.write_sample( step, parameters );

        if ( !( step % 100 ) )
            status::update( step, parameters.steps ); 

        if ( status::time_out or ( parameters.write_backup and !( step % ( parameters.volume_size.diagonal_product() * parameters.n > 10e7 ? 100'000 : 500'000 ) ) and step != 0 ) )
        {
            status::write_time_file( step );
            simulation.write_backup_file();
        
            if ( status::time_out )
                std::exit( 0x0 );
        }
    }
    status::report_done(); 
    status::write_time_file( parameters.steps );
   
    // --- final data output:
    if ( parameters.steps and ( start < parameters.steps ) )
        simulation.write_sample( parameters.steps, parameters );
     
    if ( parameters.write_backup )
        simulation.write_backup_file();

    return 0x0; 
}
