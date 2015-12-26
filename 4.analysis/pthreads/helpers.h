/************************************************************************************************
 * 																								*
 * Helper functions for file input,output,fork,exec,signals and fifos							*		
 * 																								*
 ************************************************************************************************/

/**fopen() and do error checking. This opens the file name specified in 
 * file argument with permission specified in mode argument */
FILE *openFile(char* file, char *mode);

/**fclose() and do error checking. This closes the file specified by file arg**/
void closeFile(FILE * file);

/**Check whether malloc failed*/
void checkMalloc(void *ptr);

void printWavedata(int **array,int length,int height);
void printcipher(unsigned int **array,int length,int height);
