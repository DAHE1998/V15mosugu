# change_cuda — Standalone CUDA change detection (Phase 1)
# Build: make
# Run:   ./change_cuda_main <video.mp4> [output.csv]

NVCC    = /home/dahe/miniconda3/envs/amaterasu/lib/python3.10/site-packages/nvidia/cu13/bin/nvcc
PKG     = pkg-config

CUDA_ARCH = -gencode arch=compute_89,code=sm_89
NVCCFLAGS = -O2 $(CUDA_ARCH) -lineinfo

LDFLAGS = $(shell $(PKG) --libs libavformat libavcodec libavutil libswscale) -lcuda -lcudart
CFLAGS  = $(shell $(PKG) --cflags libavformat libavcodec libavutil libswscale) -I$(CUDA_HOME)/include

TARGET  = change_cuda_main
SRC     = change_cuda_main.cu

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)

test: $(TARGET)
	./$(TARGET) /home/dahe/VideoCenter/test_videos/street/shibuya_last_train.mp4 /tmp/test_scores.csv
