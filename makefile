CC := h5c++
BUILDDIR := build
SRCDIR := src

CPP_FLAGS := -std=c++17 -Wall -Wpedantic -Wextra -Werror=shadow
CUDA_FLAGS := -std=c++17 -ccbin g++ --compiler-options -Wall,-Wextra -arch=sm_61 --expt-extended-lambda -rdc=true
LIB := -L $(CUDA)/lib64 -lcudart 
INC := -I $(CUDA)/include -I include 

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
	nvcc $(CUDA_FLAGS) $(OBJECTS) -dlink -o $(SRCDIR)/device_linked.o -lcudadevrt 
	$(CC) $(CPP_FLAGS) $(OBJECTS) $(SRCDIR)/device_linked.o -o main $(LIB) 

$(BUILDDIR)/main.o: $(SRCDIR)/main.cpp 
	@mkdir -p $(BUILDDIR)	
	$(CC) $(CPP_FLAGS) $(SRCDIR)/main.cpp -c $(INC) -o $@

$(BUILDDIR)/cuda_allocator.o: $(SRCDIR)/cuda_allocator.cpp 
	nvcc $(CUDA_FLAGS) $(SRCDIR)/cuda_allocator.cpp -c $(INC) -o $@

$(BUILDDIR)/simulation_context.o: $(SRCDIR)/simulation_context.cu 
	nvcc $(CUDA_FLAGS) $(SRCDIR)/simulation_context.cu -c $(INC) -o $@

$(BUILDDIR)/gpu_functions.o: $(SRCDIR)/gpu_functions.cu
	nvcc $(CUDA_FLAGS) $(SRCDIR)/gpu_functions.cu -c $(INC) -o $@

$(BUILDDIR)/extended_collision.o: $(SRCDIR)/extended_collision.cu 
	nvcc $(CUDA_FLAGS) $(SRCDIR)/extended_collision.cu -c $(INC) -o $@
	
$(BUILDDIR)/gpu_constants.o: $(SRCDIR)/gpu_constants.cu
	nvcc $(CUDA_FLAGS) $(SRCDIR)/gpu_constants.cu -c $(INC) -o $@

$(BUILDDIR)/h5cpp.o: $(SRCDIR)/h5cpp.cpp 
	$(CC) $(CPP_FLAGS) -Wno-unused-parameter $(SRCDIR)/h5cpp.cpp -c $(INC) -o $@ 

clean: 
	rm -f $(BUILDDIR)/*.o

