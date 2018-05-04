#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

void stringToBlock(char *str,uint8_t block[16]){
	int i;
	for(i=0;i<16;i++){
		int temp;
		sscanf(&str[2*i],"%02X",&temp);
		block[i]=(uint8_t)temp;
	}
}

void blockToSTring(uint8_t block[16],char *str){
	int i;
	for(i=0;i<16;i++){
		sprintf(&str[2*i],"%02X",(int)block[i]);
	}
	str[16*2]=0;
	
}


int main(int argc, char** argv)
{
	if(argc!=3){
		perror("Not enough args. eg ./program <no of plain text> <output file>");
		exit(1);
	}
	
	int seed=time(NULL);
	srand(seed);

	uint8_t plain[16];
	char cplain[100];

	FILE *file=fopen(argv[2],"w");
	int count=atoi(argv[1]);
	
	int j=0;
	for(j=0;j<count;j++){
	
		int i=0;
		for(i=0;i<16;i++){
			plain[i]=rand()%256;
		}
		blockToSTring(plain,cplain);
		fprintf(file,"%s\n",cplain);
		
	}
        
    return 0;
}
