PHONY:
	all clean

all : 
	nvcc resize.cu -o resize
	# nvcc resize_original.cu -o resize_original
	
clean :
	rm -rf resize
	# rm -rf resize_original
