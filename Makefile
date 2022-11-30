PHONY:
	all clean

all : 
	nvcc resize.cu  -o resize
	
clean :
	rm -rf resize
