#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
 
#define ROR(x, r) ((x >> r) | (x << (64 - r)))
#define ROL(x, r) ((x << r) | (x >> (64 - r)))
#define R(x, y, k) (x = ROR(x, 8), x += y, x ^= k, y = ROL(y, 3), y ^= x)
 
void encrypt(uint64_t *pt, uint64_t *ct, uint64_t *K)
{
   uint64_t i, B = K[1], A = K[0];
   ct[0] = pt[0]; ct[1] = pt[1];
 
   for(i = 0; i < 32; i++)
   {
      R(ct[1], ct[0], A);
      R(B, A, i);
   }
}

void convertToBlock(unsigned char block[16],uint64_t ct[2]){
  block[0] = (unsigned char) (ct[0] >> 56);
  block[1] = (unsigned char) (ct[0] >> 48);
  block[2] = (unsigned char) (ct[0] >> 40);
  block[3] = (unsigned char) (ct[0] >> 32);
  block[4] = (unsigned char) (ct[0] >> 24);
  block[5] = (unsigned char) (ct[0] >> 16);
  block[6] = (unsigned char) (ct[0] >> 8);
  block[7] = (unsigned char) (ct[0]);  
  
  block[8] = (unsigned char) (ct[1] >> 56);
  block[9] = (unsigned char) (ct[1] >> 48);
  block[10] = (unsigned char) (ct[1] >> 40);
  block[11] = (unsigned char) (ct[1] >> 32);
  block[12] = (unsigned char) (ct[1] >> 24);
  block[13] = (unsigned char) (ct[1] >> 16);
  block[14] = (unsigned char) (ct[1] >> 8);
  block[15] = (unsigned char) (ct[1]); 
}

void stringToBlock(char *str,unsigned char block[16]){
	int i;
	for(i=0;i<16;i++){
		int temp;
		sscanf(&str[2*i],"%02X",&temp);
		block[i]=(unsigned char)temp;
	}
}

void convertToNum(unsigned char block[16],uint64_t pt[2]){
	pt[0] = (((uint64_t)block[0])<<56)|(((uint64_t)block[1])<<48)|(((uint64_t)block[2])<<40)\
			|(((uint64_t)block[3])<<32)|(((uint64_t)block[4])<<24)|(((uint64_t)block[5])<<16)\
			|(((uint64_t)block[6])<<8)|(((uint64_t)block[7]));
	pt[1] = (((uint64_t)block[8])<<56)|(((uint64_t)block[9])<<48)|(((uint64_t)block[10])<<40)\
			|(((uint64_t)block[11])<<32)|(((uint64_t)block[12])<<24)|(((uint64_t)block[13])<<16)\
			|(((uint64_t)block[14])<<8)|(((uint64_t)block[15]));			

}
void blockToString(unsigned char block[16],char *str){
	int i;
	for(i=0;i<16;i++){
		sprintf(&str[2*i],"%02X",(int)block[i]);
	}
	str[16*2]=0;
	
}

int main(int argc, char** argv)
{
	if(argc!=4){
		perror("Not enough args. eg : ./program <no of samples> <plain text> <cipher text>");
		exit(1);
	}

	int count=atoi(argv[1]);	
	FILE *input=fopen(argv[2],"r");
	FILE *output=fopen(argv[3],"w");
	
	uint64_t *pt=malloc(sizeof(uint64_t)*2);
	uint64_t *ct=malloc(sizeof(uint64_t)*2);
	uint64_t *K=malloc(sizeof(uint64_t)*2);
	
	unsigned char block[16];
	
	K[0]=0x0807060504030201;
	K[1]=0x0403020501010203;
	
	
	/*
	K[0]=0xFFEEDDCCBBAA9988;
	K[1]=0xEEDDCCBBAA998877;
	*/
	pt[0]=0;
	pt[1]=0;
	
	ct[0]=0;
	ct[1]=0;

	char cinp[100];
    //char ckey[100]="00000000000000000000000000000000";
    char cout[100];
	
	int i=0;
	for(i=0;i<count;i++){
		
		fscanf(input,"%s",cinp);
		stringToBlock(cinp,block);
		convertToNum(block,pt);
		encrypt(pt,ct,K);
		convertToBlock(block,ct);
		blockToString(block,cout);
		fprintf(output,"%s\n",cout);
		
	}
		
    return 0;
}
