CC := h5c++
BUILDDIR := build

CPP_FLAGS := -std=c++17 -Wall -Wpedantic -Wextra -Werror=shadow
CUDA_FLAGS := -std=c++17 -ccbin g++ --compiler-options -Wall,-Wextra -arch=sm_61 --expt-extended-lambda -rdc=true
LIB := -L $(CUDA)/lib64 -lcudart 
INC := -I $(CUDA)/include 

OBJECTS := main.o cuda_allocator.o simulation_context.o gpu_functions.o gpu_constants.o h5cpp.o
OBJECTS := $(addprefix $(BUILDDIR)/,$(OBJECTS))
all: CPP_FLAGS += -O3 -DNDEBUG
all: CUDA_FLAGS += -O3 -DNDEBUG
all: $(OBJECTS)	
	nvcc $(CUDA_FLAGS) $(OBJECTS) -dlink -o $(BUILDDIR)/device_linked.o -lcudadevrt 
	$(CC) $(CPP_FLAGS) $(OBJECTS) $(BUILDDIR)/device_linked.o -o main $(LIB) 

debug: CPP_FLAGS += -g -DNDEBUG
debug: CUDA_FLAGS += -G -g -DNDEBUG
debug: $(OBJECTS)	
	nvcc $(CUDA_FLAGS) $(OBJECTS) -dlink -o device_linked.o -lcudadevrt 
	$(CC) $(CPP_FLAGS) $(OBJECTS) device_linked.o -o main $(LIB) 

$(BUILDDIR)/main.o: main.cpp 
	$(CC) $(CPP_FLAGS) main.cpp -c $(INC) -o $@

$(BUILDDIR)/cuda_allocator.o: cuda_allocator.cpp cuda_allocator.hpp 
	nvcc $(CUDA_FLAGS) cuda_allocator.cpp -c -o $@

$(BUILDDIR)/simulation_context.o: simulation_context.cu simulation_context.hpp 
	nvcc $(CUDA_FLAGS) simulation_context.cu -c -o $@

$(BUILDDIR)/gpu_functions.o: gpu_functions.cu gpu_functions.hpp
	nvcc $(CUDA_FLAGS) gpu_functions.cu -c -o $@
	
$(BUILDDIR)/gpu_constants.o: gpu_constants.cu gpu_constants.hpp
	nvcc $(CUDA_FLAGS) gpu_constants.cu -c -o $@

$(BUILDDIR)/h5cpp.o: h5cpp.cpp h5cpp.hpp 
	$(CC) $(CPP_FLAGS) -Wno-unused-parameter h5cpp.cpp -c -o $@ 

clean: 
	rm -f $(BUILDDIR)/*.o

