/*
	The A to Z of Building a Testbed for Power Analysis Attacks
	C source code for calculating the cipher text for a given set of plain text using 128 bit AES
 
    Any bugs, issues or suggestions please email to hasindu2008@live.com
*/

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include "aes.h"

//convert a hexedecimal format ascci character string to a numerical byte array
void stringToBlock(char *str,uint8_t block[16]){
	int i;
	for(i=0;i<16;i++){
		int temp;
		sscanf(&str[2*i],"%02X",&temp);
		block[i]=(uint8_t)temp;
	}
}

//convert a numerical byte array to hexedecimal format assci character string
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
	if(argc!=4){
		perror("Not enough args. eg : ./program <no of samples> <plain text> <cipher text>");
		exit(1);
	}

	int count=atoi(argv[1]);	
	FILE *input=fopen(argv[2],"r");
	FILE *output=fopen(argv[3],"w");
	
	//space for plain text, key and cipher text to store as numeric byte arrays
	uint8_t inp[16];
	uint8_t key[16];
	uint8_t out[16];

	//space for plain text, key and cipher text to store as character arrays
	char cinp[100];
	char cout[100];
	char ckey[100]="0102030405060708090A0B0C0D0E0F10";
	//the default key which is 0x01 0x02 0x03 0x04 0x05 0x06 0x07 0x08 0x09 0x0A 0x0B 0x0C 0x0D 0x0E 0x0F 0x10
	stringToBlock(ckey,key); //cnvert the key is asccci to numeric

	//do encryption for each plain text in the input file and save the cipher text
	int i=0;
	for(i=0;i<count;i++){
		
		fscanf(input,"%s",cinp);
		stringToBlock(cinp,inp);
		AES128_ECB_encrypt(inp,key,out);
		blockToSTring(out,cout);
		fprintf(output,"%s\n",cout);
		
	}
		
    return 0;
}

