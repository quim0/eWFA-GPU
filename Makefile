COMPUTE=70
SM=70
NVCC=nvcc
CC=gcc
SRC=src/main.cu src/wavefront.cu src/kernels.cu utils/sequence_reader.cu build/*.o
ARGS=-Isrc/ -I. -gencode arch=compute_$(COMPUTE),code=sm_$(SM) -default-stream per-thread
BIN=wfe.aligner

all: utils-o
	$(NVCC) $(SRC) $(ARGS) -O3 -o $(BIN)
debug: utils-debug
	$(NVCC) $(SRC) $(ARGS) -DDEBUG_MODE -g -G -o $(BIN)
profile: utils-o
	$(NVCC) $(SRC) $(ARGS) -DUSE_NVTX -lnvToolsExt -lineinfo -o $(BIN)
utils-o: utils/arg_handler.c
	mkdir -p build
	$(CC) -Ofast -c $^
	mv *.o build/
utils-debug: utils/arg_handler.c
	mkdir -p build
	$(CC) -g -c $^
	mv *.o build/
clean:
	rm -rf $(BIN) build/
