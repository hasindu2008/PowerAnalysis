/*
	The A to Z of Building a Testbed for Power Analysis Attacks
	C source code for generating random 128 bit plain text samples

    Authors : Hasindu Gamaarachchi, Harsha Ganegoda and Roshan Ragel, 
    Department of Computer Engineering, 
    Faculty of Engineering, University of Peradeniya, 22 Dec 2015
 
    For more information read 
    Hasindu Gamaarachchi, Harsha Ganegoda and Roshan Ragel, 
    "The A to Z of Building a Testbed for Power Analysis Attacks", 
    10th IEEE International Conference on Industrial and Information Systems 2015 (ICIIS)]
 
    Any bugs, issues or suggestions please email to hasindu2008@live.com
	
*/


#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

//convert a hexedecimal format ascci character string to a numerical byte array
void stringToBlock(char *str,uint8_t block[16]){
	int i;
	for(i=0;i<16;i++){
		int temp;
		sscanf(&str[2*i],"%02X",&temp);
		block[i]=(uint8_t)temp;
	}
}

//convert a numerical byte array to hexedecimal format ascci character string
void blockToSTring(uint8_t block[16],char *str){
	int i;
	for(i=0;i<16;i++){
		sprintf(&str[2*i],"%02X",(int)block[i]);
	}
	str[16*2]=0;
	
}


int main(int argc, char** argv)
{
	//check for args
	if(argc!=3){
		perror("Not enough args. eg ./program <no of plain text> <output file>");
		exit(1);
	}
	
	//take a seed for random number generator
	int seed=time(NULL);
	srand(seed);

	//space for plaintext
	uint8_t plain[16]; //numeric byte array
	char cplain[100];  //character string

	//open file
	FILE *file=fopen(argv[2],"w");
	
	//get number of plain text required to build
	int count=atoi(argv[1]);
	
	
	//generate them and writeto file
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
