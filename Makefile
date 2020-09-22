CC=nvcc
SRC=main.cu utils.cu wavefront.cu kernels.cu
ARGS=-gencode arch=compute_70,code=sm_70 -default-stream per-thread

all:
	$(CC) $(SRC) $(ARGS) -O3 -o wfa.edit.distance.gpu
debug:
	$(CC) $(SRC) $(ARGS) -DDEBUG_MODE -g -G -o wfa.edit.distance.gpu
profile:
	$(CC) $(SRC) $(ARGS) -DUSE_NVTX -lnvToolsExt -lineinfo -o wfa.edit.distance.gpu
clean:
	rm -f wfa.edit.distance.gpu
