DEPS = $(wildcard include/*.h)
OBJ = obj/parallelcalls.o obj/main.o obj/serialcalls.o
MAKE = make
CFLAGS = -I/opt/nvidia-cuda/include -I./include -c

all: $(OBJ)
	nvcc $(OBJ) -o a.out -lmpi -lm
	mpirun  -np 2 ./a.out
	
obj/parallelcalls.o: src/parallelcalls.cu $(DEPS)
	nvcc $(CFLAGS) src/parallelcalls.cu -o obj/parallelcalls.o

obj/serialcalls.o: src/serialcalls.cu $(DEPS)
	nvcc $(CFLAGS) src/serialcalls.cu -o obj/serialcalls.o

obj/main.o: $(DEPS) src/main.c
	mpicc $(CFLAGS)  -std=c99 src/main.c -o obj/main.o


clean:
	@rm -rf $(OBJ)
	@rm -rf field1.bin field2.bin field3.bin res.bin gcx.bin gdx.bin gcz.bin
	@rm -rf a.out
.PHONY: clean