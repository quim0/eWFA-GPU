CC=nvcc
SRC=main.cpp utils.cpp wavefront.cu kernels.cu

all:
	$(CC) $(SRC) -o wfa.edit.distance.gpu
debug:
	$(CC) $(SRC) -DDEBUG_MODE -g -o wfa.edit.distance.gpu
clean:
	rm -f wfa.edit.distance.gpu
