 typedef unsigned char byte;


__device__ void copy(unsigned char ans[8],unsigned char x[8]){
	int i;
	for(i=0;i<8;i++){
		ans[i]=x[i];
	}

}

__device__ void copy2(unsigned char ans[8],unsigned int *x){
	int i;
	for(i=0;i<8;i++){
		ans[i]=(unsigned char)x[i];
	}

}
 
__device__ void _shiftR(unsigned char ans[8],unsigned char x[8],unsigned char r){
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

__device__ void _shiftL(unsigned char ans[8],unsigned char x[8],unsigned char r){
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
 
__device__ void _or(unsigned char ans[8],unsigned char x[8],unsigned char y[8]){
	int i=0;
	for(i=0;i<8;i++){
		ans[i]=(x[i])|(y[i]);
	}
} 

__device__ void _add(unsigned char ans[8],unsigned char x[8],unsigned char y[8]){
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
 
__device__ void _xor(unsigned char ans[8],unsigned char x[8],unsigned char y[8]){
	int i=0;
	for(i=0;i<8;i++){
		ans[i]=(x[i])^(y[i]);
	}
} 
__device__ void ROR(unsigned char ans[8],unsigned char x[8], unsigned char  r){
	unsigned char r1[8];
	unsigned char r2[8];
	_shiftR(r1,x,r);
	_shiftL(r2,x,64-r);
	_or(ans,r1,r2);
}

__device__ void ROL(unsigned char ans[8],unsigned char x[8], unsigned char  r){
	unsigned char r1[8];
	unsigned char r2[8];
	_shiftL(r1,x,r);
	_shiftR(r2,x,64-r);
	_or(ans,r1,r2);
}
