#Makefile


SOURCE =  main.cu

CC = nvcc

exe: $(SOURCE)
	$(CC)    $(SOURCE)  -o exe


clean:
	$(RM) -rf exe 

run:
	./exe




