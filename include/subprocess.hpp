#ifndef __SUBPROCESS_HPP 
#define __SUBPROCESS_HPP

#include <algorithm>
#include <functional>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <unistd.h>
#include <sys/wait.h>

struct command_execution_exception : public std::exception
{
    std::string error_message;
    command_execution_exception( std::string e ) : error_message( e ) {}
    virtual const char* what() const throw()
    {
        return error_message.c_str();
    }
};

inline std::string make_pipe() { return ""; }
template< typename ...Ts >
std::string make_pipe( std::string first, Ts ...tail )
{
    auto s = make_pipe( tail... );
    if( s != "" ) s = " | " + s;
    return first + " 2>&1" + s ;
}

class output_of
{
    int  fp[2],
         child_pid;
    char separator = ' ';

    public:

    template< typename ...Ts >
    output_of( char const* t, Ts&&... args ) 
    {
        if( pipe( fp ) < 0 )
            throw command_execution_exception( "pipe error!" );

        switch ( ( child_pid = fork() ) )
        {
            case -1:
                throw command_execution_exception( "fork error!" );
                break;
            case 0:
                close( 1 );
                if( dup( fp[ 1 ] ) < 0 )
                    throw command_execution_exception( "dup error!" );
                close( fp[ 0 ] );
                if ( execl( t, t, args... , 0 ) < 0 )
                    write( fp[ 1 ], "error on exec!\0", 15 ); 
                exit( 0 );
                break;
            default:
                close( fp[ 1 ] );
                break;
        }
    }
    
    ~output_of() 
    {
        waitpid( child_pid, NULL, 0 );
    }

    output_of& operator >> ( std::string& word )
    {
        word = {};
        char c = {};

        if( read( fp[ 0 ], &c, sizeof(c) ) > 0 and c != ' ' and c != '\n' )
            word.push_back( c );

        while( read( fp[ 0 ], &c, sizeof(c) ) > 0 )
        {
            if( c == separator or c == '\n' ) 
                break;
            word.push_back( c );
        }
        return *this;
    }
};

inline std::string date()
{
    output_of o( "/bin/date" );
    std::string buffer, line;

    do 
    {
        o >> buffer;
        line += buffer + ' ';
    } while ( buffer.length() > 0 );
    line.pop_back();
    
    return line;
}

template< typename ...Ts >
std::vector< std::string > get_output( Ts ...commands )
{
    output_of o( commands... );
    std::string buffer; std::vector< std::string > retval;

    for( ;; )
    {
        o >> buffer;
        if( buffer == "" ) break;
        retval.push_back( buffer );
    }

    return retval;
}

template< typename ...Ts >
std::vector< std::string > list_directory( Ts ...commands )
{
    output_of o( "ls " + commands... );
    std::string buffer; std::vector< std::string > retval;
    for(;;)
    {
        o >> buffer;
        if( buffer == "" ) break;
        retval.push_back( buffer );
    }

    if(  retval[0].find( "ls: cannot acces", 0 ) != std::string::npos )
        return {};
    else
        return retval;
}

inline size_t get_pid()
{
    return static_cast< size_t >( getpid() );
}


#endif // __SUBPROCESS_HPP
