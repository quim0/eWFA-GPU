CC=nvcc
SRC=main.cu utils.cu wavefront.cu kernels.cu
ARGS=-gencode arch=compute_75,code=sm_75

all:
	$(CC) $(SRC) $(ARGS) -O3 -o wfa.edit.distance.gpu
debug:
	$(CC) $(SRC) $(ARGS) -DDEBUG_MODE -g -G -o wfa.edit.distance.gpu
profile:
	$(CC) $(SRC) $(ARGS) -lineinfo -o wfa.edit.distance.gpu
clean:
	rm -f wfa.edit.distance.gpu
