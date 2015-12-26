/* 

	C file for CUDA helper functions
	Include this file in your code if you want to use the function checkCudaError() 
	checkCudaError() function checks if the last cuda function call or kernel launch caused an error
	and if yes it will print the error message and will abort the program

	For example to check any errors due to memcpy use it as
	cudaMemcpy(arguments...);
	checkCudaError();

	For example to check any errors due to a kernel launch use it as		
	kernel<<<parameters>>>(arguments...);
	checkCudaError();
	
	Hasindu Gamaarachchi <hasindu2008@live.com>
	22 Dec 2015

*/

#include <stdio.h>
#include "helpers.cuh"
	
/* check whether the last CUDA function or CUDA kernel launch is erroneous and if yes an error message will be printed
and then the program will be aborted*/

void gpuAssert(const char *file, int line){

	cudaError_t code = cudaGetLastError();
	if (code != cudaSuccess) {
        fprintf(stderr, "Cuda error: %s \n in file : %s line number : %d", cudaGetErrorString(code), file, line );
        exit(-1);
   }
}


/* Check whether a previous memory allocation was successful. If RAM is full usually the returned value is a NULL pointer.
For example if you allocate memory by doing 
int *mem = malloc(sizeof(int)*SIZE)
check whether it was successful by calling
checkAllocRAM(mem) afterwards */

void checkAllocRAM(void *ptr){
	if (ptr==NULL){
		fprintf(stderr, "Memory Full.\nYour array is too large. Please try a smaller array.\n");
		exit(EXIT_FAILURE);
	}
}

/* This checks whether a file has been opened corrected. If a file opening failed the returned value is a NULL pointer
FOr example if you open a file using
FILE *file=fopen("file.txt","r");
check by calling isFileValid(file); */

void isFileValid(FILE *fp){
	if (fp==NULL){
		perror("A file access error occurred\n");
		exit(EXIT_FAILURE);
	}
}