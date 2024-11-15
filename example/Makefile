#NVCC=nvcc --generate-line-info
#NVCC=/usr/local/cuda-11.0/bin/nvcc --generate-line-info
# include ../../../../utility/config.mk

all: ex

ex:
	nvcc --generate-line-info ex.cu -o ex

exf:
	nvcc --generate-line-info --use_fast_math ex.cu -o exf

run:ex
	./ex

runf:exf
	./exf

detect:ex
	LD_PRELOAD=../nvbit_release/tools/GPU-FPX/detector/detector.so ./ex

detectf:exf
	LD_PRELOAD=../nvbit_release/tools/GPU-FPX/detector/detector.so ./exf

analyze:ex
	LD_PRELOAD=../nvbit_release/tools/GPU-FPX/analyzer//analyzer.so ./ex

analyzef:exf
	LD_PRELOAD=../nvbit_release/tools/GPU-FPX/analyzer//analyzer.so ./exf

clean:
	rm -rf ex exf

