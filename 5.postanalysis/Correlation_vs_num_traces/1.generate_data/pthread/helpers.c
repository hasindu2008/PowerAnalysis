/************************************************************************************************
 * 																								*
 * Helper functions for file input,output,fork,exec,signals and fifos							*		
 * 																								*
 ************************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/**fopen() and do error checking. This opens the file name specified in 
 * file argument with permission specified in mode argument */
FILE* openFile(char* file, char *mode){
	FILE* filep=fopen(file,mode);
	if(filep==NULL){
		perror("Cannot open file");
		exit(EXIT_FAILURE);
	}
	return filep;
}

/**fclose() and do error checking. This closes the file specified by file arg**/
void closeFile(FILE * file){
	int ret=fclose(file);
	if(ret==-1){
		perror("Cannot close file");
		exit(EXIT_FAILURE);
	}
}


/**Check whether malloc failed*/
void checkMalloc(void *ptr){
	if(ptr==NULL){					
		perror("Malloc failed");
		exit(EXIT_FAILURE);
	}
}
	

/**Print wavedata*/
void printWavedata(int **array,int length,int height){
	int i,j;
	for(i=0;i<height;i++){
		for(j=0;j<length;j++){
			printf("%d ",array[i][j]);
		}
		printf("\n");
	}
}

/**Print cipher*/
void printcipher(unsigned int **array,int length,int height){
	int i,j;
	for(i=0;i<height;i++){
		for(j=0;j<length;j++){
			printf("%x ",array[i][j]);
		}
		printf("\n");
	}
}
