CC := h5c++
BUILDDIR := build
SRCDIR := include

CPP_FLAGS := -std=c++17 -Wall -Wpedantic -Wextra -Werror=shadow
CUDA_FLAGS := -std=c++17 -ccbin g++ --compiler-options -Wall,-Wextra -arch=sm_61 --expt-extended-lambda -rdc=true
LIB := -L $(CUDA)/lib64 -lcudart 
INC := -I $(CUDA)/include -I $(SRCDIR) 

OBJECTS := main.o cuda_allocator.o simulation_context.o gpu_functions.o extended_collision.o gpu_constants.o h5cpp.o 
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
	@mkdir -p $(BUILDDIR)	
	@echo $(SRCDIR)/cuda_allocator.hpp
	$(CC) $(CPP_FLAGS) main.cpp -c $(INC) -o $@

$(BUILDDIR)/cuda_allocator.o: cuda_allocator.cpp $(SRCDIR)/cuda_allocator.hpp 
	nvcc $(CUDA_FLAGS) cuda_allocator.cpp -c $(INC) -o $@

$(BUILDDIR)/simulation_context.o: simulation_context.cu $(SRCDIR)/simulation_context.hpp 
	nvcc $(CUDA_FLAGS) simulation_context.cu -c $(INC) -o $@

$(BUILDDIR)/gpu_functions.o: gpu_functions.cu $(SRCDIR)/gpu_functions.hpp
	nvcc $(CUDA_FLAGS) gpu_functions.cu -c $(INC) -o $@

$(BUILDDIR)/extended_collision.o: extended_collision.cu $(SRCDIR)/extended_collision.hpp 
	nvcc $(CUDA_FLAGS) extended_collision.cu -c $(INC) -o $@
	
$(BUILDDIR)/gpu_constants.o: gpu_constants.cu $(SRCDIR)/gpu_constants.hpp
	nvcc $(CUDA_FLAGS) gpu_constants.cu -c $(INC) -o $@

$(BUILDDIR)/h5cpp.o: h5cpp.cpp $(SRCDIR)/h5cpp.hpp 
	$(CC) $(CPP_FLAGS) -Wno-unused-parameter h5cpp.cpp -c $(INC) -o $@ 

clean: 
	rm -f $(BUILDDIR)/*.o

