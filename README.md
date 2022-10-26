# cudaMPCD
Fast hydrodynamic solver for GPUS using the method of multi-particle collision dynamics in C++/CUDA.
For configuring the simulation after compilation using the make command, modify the input_file.     
In the current state, this code can be used to simulate a Poiseuille flow, i.e. a flow between to parallel plates.

The Poiseuille flow is a geometry for which the Navier-Stokes equations can be solved analytically. 
Thus, we can use this geometry for testing the code.
After the simulation, we may use the flow field data as follows (in Julia code):
```
using HDF5
using PyPlot
using Statistics

file = h5open( "data/poiseuille_flow/simulation_data.h5" )
data = read( file[ "fluid/100000" ] )
close( file );

density    = data[1,:,:,:];
x_velocity = data[2,:,:,:]; # and so on...
profile = reshape( mean( x_velocity, dims=(1,2) ), : ); # make one-dimensional

plot( profile ) # will show a parabola

#simulation parameters:
L = 100
g = 0.0001
n = 10
ν = η / 10
Δt = 0.02

# viscosity measurement:
η = L * L * n * g / ( 8 * max( profile... ) )

# viscosity theoretical:
η_theo = ( 1 - cos(120/180*π)) / (6*3*Δt) * ( n - 1 + exp(-n) )

println( "theoretical: " η_theo, ", measured:", η ) 
```
The last line prints the two values of the fluid viscosity "theoretical: 37.50, measured: 37.27"

Dependencies: CUDA, HDF5

TODO: add references (papers)

