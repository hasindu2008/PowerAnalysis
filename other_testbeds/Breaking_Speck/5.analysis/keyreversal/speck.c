#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

unsigned char pt0[8];
unsigned char pt1[8];
unsigned char ct0[8];
unsigned char ct1[8];
unsigned char K0[8];
unsigned char K1[8];

void print(unsigned char pr[8]){
	int i;
	for(i=0;i<8;i++){
		printf("%.2X ",(int)pr[i]);
	}
	printf("\n");

}
 
void copy(unsigned char ans[8],unsigned char x[8]){
	int i;
	for(i=0;i<8;i++){
		ans[i]=x[i];
	}

}
 
void _shiftR(unsigned char ans[8],unsigned char x[8],unsigned char r){
	int i;
	int shiftbytes=r/8;
	int shiftbits=r%8;
	
	unsigned char temp[8];
	
	for(i=0;i<shiftbytes;i++){
		ans[i]=0;
	}
	for(i=shiftbytes;i<8;i++){
		ans[i]=x[i-shiftbytes];
	}
	
	for(i=0;i<8;i++){
		temp[i]=ans[i];
	}
	
	if(shiftbits>0){
		for(i=shiftbytes;i<8;i++){
			if(i==shiftbytes){
				ans[i]=(temp[i]>>shiftbits);
			}
			else{
				ans[i]=(temp[i]>>shiftbits)|(temp[i-1]<<(8-shiftbits));
			}
		}
	}
	
} 

void _shiftL(unsigned char ans[8],unsigned char x[8],unsigned char r){
	int i;
	int shiftbytes=r/8;
	int shiftbits=r%8;
	unsigned char temp[8];
	
	for(i=0;i<shiftbytes;i++){
		ans[7-i]=0;
	}
	for(i=shiftbytes;i<8;i++){
		ans[7-i]=x[7-i+shiftbytes];
	}
	
	for(i=0;i<8;i++){
		temp[i]=ans[i];
	}	
	
	if(shiftbits>0){
		for(i=shiftbytes;i<8;i++){
			if(i==shiftbytes){
				ans[7-i]=(temp[7-i]<<shiftbits);
			}
			else{
				ans[7-i]=(temp[7-i]<<shiftbits)|(temp[8-i]>>(8-shiftbits));
			}
		}
	}	
} 
 
void _or(unsigned char ans[8],unsigned char x[8],unsigned char y[8]){
	int i=0;
	for(i=0;i<8;i++){
		ans[i]=(x[i])|(y[i]);
	}
} 

void _add(unsigned char ans[8],unsigned char x[8],unsigned char y[8]){
	int i;
	unsigned char q=0;
	// for(i=7;i>=0;i--){
		// unsigned int add=(int)(x[i]+y[i]+q);
		// ans[i]=(unsigned char)(add%256);
		// q=(unsigned char)(add/256);
	// }
	
 for(i=0;i<8;i++){
      unsigned int add;
      add=(int)(x[7-i]+y[7-i]+q);
      ans[7-i]=(unsigned char)(add%256);
      q=(unsigned char)(add/256);
  }	
	
}
 
void __sub(unsigned char ans[8],unsigned char x[8],unsigned char y[8]){
	int i;
	unsigned char q=0;
	// for(i=7;i>=0;i--){
		// unsigned int add=(int)(x[i]+y[i]+q);
		// ans[i]=(unsigned char)(add%256);
		// q=(unsigned char)(add/256);
	// }
	
 for(i=0;i<8;i++){
      unsigned int sub;
	  if(x[7-i]>y[7-i]){
		  sub=(int)(x[7-i]-q-y[7-i]);
		  q=0;
	  }
	  else{
		  sub=(int)(256+x[7-i]-q-y[7-i]);		
		  q=1;
	  }
	  ans[7-i]=(unsigned char)(sub);

  }	
	
} 
 
void _xor(unsigned char ans[8],unsigned char x[8],unsigned char y[8]){
	int i=0;
	for(i=0;i<8;i++){
		ans[i]=(x[i])^(y[i]);
	}
} 
void ROR(unsigned char ans[8],unsigned char x[8], unsigned char  r){
	unsigned char r1[8];
	unsigned char r2[8];
	_shiftR(r1,x,r);
	_shiftL(r2,x,64-r);
	_or(ans,r1,r2);
}

void ROL(unsigned char ans[8],unsigned char x[8], unsigned char  r){
	unsigned char r1[8];
	unsigned char r2[8];
	_shiftL(r1,x,r);
	_shiftR(r2,x,64-r);
	_or(ans,r1,r2);
}

void printkey(unsigned char A[8]){
	int i;
	for(i=0;i<8;i++){
		printf("%.2X ",A[i]);
	}
	printf("\n");
	
}

 
void encrypt()
{
   unsigned char i;		
   unsigned char ans[8];
   unsigned char B[8];
   unsigned char A[8];
	
   for(i=0;i<8;i++){
	B[i] = K1[i];
	A[i] = K0[i];
	ct0[i] = pt0[i]; 
	ct1[i] = pt1[i];
   }

 for(i = 0; i < 32; i++){

		ROR(ans,ct1,8);	
		copy(ct1,ans);		
		_add(ans,ct1,ct0);
		copy(ct1,ans);
	
		_xor(ans,ct1,A);
		copy(ct1,ans);

		ROL(ans,ct0,3);
		copy(ct0,ans);

		printkey(A);
		_xor(ans,ct1,ct0);
		copy(ct0,ans);
		
		ROR(ans,B,8);
		copy(B,ans);
		
		_add(ans,B,A);
		copy(B,ans);
	
		unsigned char index[8];
		int j=0;
		for(j=0;j<7;j++){
			index[j]=0;
		}
		index[7]=(unsigned char)i;
		_xor(ans,B,index);
		copy(B,ans);
			
		ROL(ans,A,3);
		copy(A,ans);
		_xor(ans,B,A);
		copy(A,ans);

  }
}

void blockToSTring(unsigned char block1[8],unsigned char block2[8],char *str){
	int i;
	for(i=0;i<8;i++){
		//sprintf(&str[2*i],"%.2x",(unsigned char)block1[i]);
		printf("%.2X ",block1[i]);
	}
	for(i=8;i<16;i++){
		//sprintf(&str[2*i],"%.2x",(unsigned char)block2[i-8]);
		printf("%.2X ",block2[i-8]);

	}	
	//str[16*2]=0;
	printf("\n");
	
}
void convertbyte(unsigned char block[16],uint64_t ct){
  block[0] = (unsigned char) (ct >> 56);
  block[1] = (unsigned char) (ct >> 48);
  block[2] = (unsigned char) (ct >> 40);
  block[3] = (unsigned char) (ct >> 32);
  block[4] = (unsigned char) (ct >> 24);
  block[5] = (unsigned char) (ct >> 16);
  block[6] = (unsigned char) (ct >> 8);
  block[7] = (unsigned char) (ct);  
}

void blockToSTringbyte(unsigned char block[8]){
	int i;
	for(i=0;i<8;i++){
		//sprintf(&str[2*i],"%.2x",(unsigned char)block[i]);
		printf("%.2X ",block[i]);
	}
	//str[16*2]=0;
	printf("\n");
	
}


int main(){
	
	int i=0;
	//char output[100];
	
	for(i=0;i<8;i++){
	
		pt0[i]=0;
		pt1[i]=0;
	
		ct0[i]=0;
		ct1[i]=0;
	}
	
   
    //left half key
    K1[0]=0x04;
    K1[1]=0x03;
    K1[2]=0x02;
    K1[3]=0x05;
    K1[4]=0x01;
    K1[5]=0x01;
    K1[6]=0x02;
    K1[7]=0x03;   	
	
	//right half key
    K0[0]=0x08;
    K0[1]=0x07;
    K0[2]=0x06;
    K0[3]=0x05;
    K0[4]=0x04;
    K0[5]=0x03;
    K0[6]=0x02;
    K0[7]=0x01;	
	
	/*pt0[4]=0x75;
	pt0[5]=0xDD;
	pt0[6]=0x9A;
	pt0[7]=0x53;
	
	pt1[3]=0x02;	
	pt1[4]=0x17;	
	pt1[5]=0x36;	
	pt1[6]=0x0B;
	pt1[7]=0x1A;*/
		
   unsigned char ans[8];
   unsigned char B[8];
   unsigned char A[8];
   unsigned char Aror[8];
	
   for(i=0;i<8;i++){
	B[i] = K1[i];
	A[i] = K0[i];
   }

   printf("Init right half key: ");
    printkey(A);
	
	ROR(ans,B,8);
	copy(B,ans);
   printf("Left half key after ROR 8 : ");
    printkey(B);
		
	_add(ans,B,A);
	copy(B,ans);
   printf("Left half key after add : ");
    printkey(B);
	
	ROL(ans,A,3);
	copy(A,ans);
   printf("Right half key after ROL3 :")	;
    printkey(A);	
	copy(Aror,A);
	
	_xor(ans,B,A);
	copy(A,ans);
   printf("Right half key  after XOR B (round key): ")	;
    printkey(A);	
	

    puts("");
	puts("going back");	
	_xor(ans,A,Aror);
	copy(A,ans);
   printf("Right half key  before XOR B : ")	;
    printkey(A);
	
	__sub(ans,A,K0);
	copy(A,ans);
   printf("Left half key before addition : ")	;
    printkey(A);
	
	ROL(ans,A,8);
	copy(A,ans);	
   printf("Left half key  before ROL : ")	;
    printkey(A);

	printkey(A);	
	
	//blockToSTring(pt0,pt1,output);
	//puts("");
	//encrypt();
	//puts("");	
	//blockToSTring(ct0,ct1,output);
	//puts("");
	//printf("\n\n");
	
	return 0;
}