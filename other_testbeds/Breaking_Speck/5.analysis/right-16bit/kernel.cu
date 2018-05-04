/* Author : Hasindu Gamaarachchi
	CPA for 128/128 bit SPECK software implementation
	To derive right half key K2
	*/
	
#include <stdio.h>
#include "helpers.cuh"
#include "data.cuh"
#include <stdint.h>

//file name for all key-correlation pairs sorted in key order
#define FILEALL "all.txt"
//file name for all key-correlation pairs sorted using correlation coefficient
#define FILEALLSORTED "allsorted.txt"

//set 1 if your samples are hexadecimal separated by spaces
//set 0 if your samples are hexadecimal with no spaces
#define FORMAT 0

//set 0 if need to take fabs(), 1 if negative, 2 if positive
#define CORRELATION_SIGN 0

//Change SAMPLES to the number of power traces
#define SAMPLES 1000
//Change ALLWAVELEGTH to the number of sampling points you have in a single power trace
#define ALLWAVELEN 100000
//Due to memory restrictions on GPU if SAMPLES is large cannot keep all the things at once in memory
//In such case of a memory allocation failure reduce WAVELENGTH
//But make sure that ALLWAVELENGTH is divisible by WAVELENGTH 
#define WAVELENGTH 2000

//define for 128/128 Speck
#define KEYBYTES 16
#define KEYBYTESPART 4
#define KEYS 65536

//struct used for sorting correlation key pairs
struct key_corr{
	unsigned int key;
	double corr;
};


__device__ uint16_t hammingweight(uint16_t H){
	//byte H=M^R;
	// Count the number of set bits
	uint16_t dist=0;
	while(H){
		dist++; 
		H &= H - 1;
	}
	return dist;
}

__device__ uint16_t hamming(unsigned int *sample, unsigned int i,unsigned int n,unsigned int key) { //n is byteno  i is the sample
    
	byte pt0[8];
	copy2(pt0,&sample[i*KEYBYTES]);
	
	byte pt1[8];
	copy2(pt1,&sample[i*KEYBYTES+8]);	
	
	byte ans[8];
	ROR(ans,pt1,8);	
	copy(pt1,ans);		
	_add(ans,pt1,pt0);
	copy(pt1,ans);		
	
	uint16_t answer=(uint16_t)pt1[n*2]<<8|(uint16_t)pt1[n*2+1];
	uint16_t inter;
	if(n<4){	
		 inter= (uint16_t)(answer ^ key);
	}
	else{
		inter = (uint16_t)(answer ^ key);
	}

    uint16_t dist = hammingweight(inter);	  
	return dist;
}


__global__ void maxCorelationkernel(double *corelation,double *wavestat,double *wavestat2,double *hammingstat){
	
	int keyguess=blockDim.y*blockIdx.y+threadIdx.y;
	int keybyte=blockDim.x*blockIdx.x+threadIdx.x;

	if (keybyte<KEYBYTESPART && keyguess<KEYS ){ 
		
		double sigmaH,sigmaH2,sigmaW=0,sigmaW2=0,sigmaWH=0;	
		sigmaH=hammingstat[KEYBYTESPART*keyguess+keybyte];
		sigmaH2=hammingstat[KEYS*KEYBYTESPART+KEYBYTESPART*keyguess+keybyte];
		double temp_corelation=0;;
		double corelationmax=0;;
		unsigned int j;
		for(j=0;j<WAVELENGTH;j++){
			
			sigmaWH=wavestat2[j*KEYS*KEYBYTESPART + keyguess*KEYBYTESPART + keybyte];
			sigmaW=wavestat[j];
			sigmaW2=wavestat[WAVELENGTH+j];

			double numerator=SAMPLES*sigmaWH - sigmaW*sigmaH;
			double denominator=sqrt(SAMPLES*sigmaW2 - sigmaW*sigmaW)*sqrt(SAMPLES*sigmaH2 - sigmaH*sigmaH);

			if(CORRELATION_SIGN==0){
				temp_corelation=fabs(numerator/denominator);
			}
			else if(CORRELATION_SIGN==1){
				temp_corelation=-numerator/denominator;
			}
			else if(CORRELATION_SIGN==2){
				temp_corelation=numerator/denominator;
			}			
			else{
				temp_corelation=fabs(numerator/denominator);
			}			
		
			if(temp_corelation>corelationmax){
				corelationmax=temp_corelation;
			}
		}

		if(corelationmax>corelation[keyguess*KEYBYTESPART+keybyte]){
			corelation[keyguess*KEYBYTESPART+keybyte]=corelationmax;
		}
	}
	return;
}

__global__ void wavestatkernel(double *wavedata, double *wavestat,double *wavestat2,byte *hammingArray){

	int keyguess=blockDim.y*blockIdx.y+threadIdx.y;
	int keybyte=blockDim.x*blockIdx.x+threadIdx.x;
	int wave=blockDim.z*blockIdx.z+threadIdx.z;

	if (keyguess<KEYS && keybyte<KEYBYTESPART && wave<WAVELENGTH ){
		unsigned int i;
		double sigmaWH=0;
		for(i=0;i<SAMPLES;i++){
			sigmaWH+=wavedata[i*WAVELENGTH+wave]*(double)hammingArray[i*KEYS*KEYBYTESPART + keyguess*KEYBYTESPART + keybyte];
		}
		wavestat2[wave*KEYS*KEYBYTESPART + keyguess*KEYBYTESPART + keybyte ]=sigmaWH;
	}

	if (keyguess==0 && keybyte==0 && wave<WAVELENGTH ){
		unsigned int i;
		double sigmaW=0,sigmaW2=0,W=0;
		for(i=0;i<SAMPLES;i++){
			W=wavedata[i*WAVELENGTH+wave];
			sigmaW+=W;
			sigmaW2+=W*W;
		}
		wavestat[wave]=sigmaW;
		wavestat[WAVELENGTH+wave]=sigmaW2;
	}
	return;
}

__global__ void hammingkernel(unsigned int *sample,byte *hammingArray,double *hammingstat){
	int keyguess=blockDim.y*blockIdx.y+threadIdx.y;
	int keybyte=blockDim.x*blockIdx.x+threadIdx.x;

	if (keybyte<KEYBYTESPART && keyguess<KEYS ){
		double sigmaH=0,sigmaH2=0;
		byte H;
		unsigned int i;
		for(i=0;i<SAMPLES;i++){
			H=hamming(sample,i,keybyte,keyguess);
			hammingArray[i*KEYS*KEYBYTESPART + keyguess*KEYBYTESPART + keybyte]=H;
			sigmaH+=(double)H;
			sigmaH2+=(double)H*(double)H;
		}
		hammingstat[KEYBYTESPART*keyguess+keybyte]=sigmaH;
		hammingstat[KEYS*KEYBYTESPART+KEYBYTESPART*keyguess+keybyte]=sigmaH2;
	}
	return;
}


int main(int argc, char *argv[]){
	
	unsigned int i,j;
		
	//check args
	if(argc!=3){
		fprintf(stderr,"%s\n", "Not enough args. eg ./cpa wavedata.txt sample.txt");
		exit(EXIT_FAILURE);
	}
	
	if(ALLWAVELEN%WAVELENGTH !=0){
		fprintf(stderr,"Make sure that ALLWAVELEN is divisible by WAVELEN\n");
		exit(1);
	}	
	
	//get wave data
	double *wavedata=(double *)malloc(sizeof(double) * SAMPLES*  WAVELENGTH);
	isMemoryFull(wavedata);

	//get sample texts
	unsigned int *sample=(unsigned int *)malloc(sizeof(unsigned int)*SAMPLES*KEYBYTES);
	isMemoryFull(sample);
		
FILE *file=fopen(argv[2],"r");
	isFileOK(file);	
	
	if(FORMAT==1){
		for(i=0; i<SAMPLES ;i++){
			for(j=0; j<KEYBYTES; j++){
				fscanf(file,"%x",&sample[i*KEYBYTES+j]);
			}
		}

	}
	
	else if(FORMAT==0){
		char str[100];
		for(i=0; i<SAMPLES ;i++){
			fscanf(file,"%s",str);
			for(j=0; j<KEYBYTES; j++){
				sscanf(&str[2*j],"%02X",&sample[i*KEYBYTES+j]);		
			}
		}
	}
	
	else{
		fprintf(stderr,"Unknown FORMAT for sample text\n");
		exit(1);
	}
	fclose(file);


	//space for corelation
	double *corelation=(double *)malloc(sizeof(double) * KEYS * KEYBYTESPART);
	isMemoryFull(corelation);
	
	//Time
	cudaEvent_t start,stop;
	float elapsedtime;
	cudaEventCreate(&start);
	cudaEventRecord(start,0);
	
	//cuda arrays and copying
	double *dev_wavedata;
	unsigned int *dev_sample;
	double *dev_corelation,*dev_wavestat,*dev_wavestat2,*dev_hammingstat;
	byte *dev_hammingArray;
	checkCudaError(cudaMalloc((void**)&dev_wavedata, SAMPLES*WAVELENGTH*sizeof(double)));
	checkCudaError(cudaMalloc((void**)&dev_sample, SAMPLES*KEYBYTES*sizeof(unsigned int)));
	checkCudaError(cudaMalloc((void**)&dev_corelation, KEYS*KEYBYTESPART*sizeof(double)));
	checkCudaError(cudaMalloc((void**)&dev_hammingArray, KEYS*KEYBYTESPART*SAMPLES*sizeof(byte)));
	checkCudaError(cudaMalloc((void**)&dev_wavestat, 2*WAVELENGTH*sizeof(double)));
	checkCudaError(cudaMalloc((void**)&dev_wavestat2, KEYS*KEYBYTESPART*WAVELENGTH*sizeof(double)));
	checkCudaError(cudaMalloc((void**)&dev_hammingstat, 2*KEYS*KEYBYTESPART*sizeof(double)));
	
	checkCudaError(cudaMemset(dev_corelation,0, KEYS*KEYBYTESPART*sizeof(double)));
	checkCudaError(cudaMemcpy(dev_sample,sample, SAMPLES*KEYBYTES*sizeof(unsigned int),cudaMemcpyHostToDevice));
	
	dim3 grid(KEYBYTES/4,KEYS/64);
	dim3 block(4,64);

	//findhamming
	hammingkernel<<<grid,block>>>(dev_sample,dev_hammingArray,dev_hammingstat);
	checkCudaError(cudaGetLastError());


	int loops=0;
	for(loops=0;loops<ALLWAVELEN/WAVELENGTH;loops++){

		FILE *file=fopen(argv[1],"r");
		isFileOK(file);
		for(i=0; i<SAMPLES ;i++){
			unsigned int k=0;
			for(j=0; j<ALLWAVELEN; j++){
				float dat;
				fscanf(file,"%f",&dat);
				if(j<WAVELENGTH*(loops+1) && j>=WAVELENGTH*loops){
					wavedata[i*WAVELENGTH+k]=(double)dat;
					k++;
				}
			}
		}	
		fclose(file);

		checkCudaError(cudaMemcpy(dev_wavedata,wavedata,SAMPLES*WAVELENGTH*sizeof(double),cudaMemcpyHostToDevice));
		
		dim3 block3d(4,32,4);
		dim3 grid3d(KEYBYTESPART/4,KEYS/32,WAVELENGTH/4);
		
		//find wave stats
		wavestatkernel<<<grid3d,block3d>>>(dev_wavedata,dev_wavestat,dev_wavestat2,dev_hammingArray);
		checkCudaError(cudaGetLastError());

		//deploy double 
		maxCorelationkernel<<<grid,block>>>(dev_corelation,dev_wavestat,dev_wavestat2,dev_hammingstat);
		checkCudaError(cudaGetLastError());	

		//progress
		fprintf(stderr,"%d of %d completed\n",loops+1,ALLWAVELEN/WAVELENGTH);		
		
	}

	//copy back
	checkCudaError(cudaMemcpy(corelation,dev_corelation,KEYS*KEYBYTESPART*sizeof(double),cudaMemcpyDeviceToHost));
	checkCudaError(cudaFree(dev_wavedata));
	checkCudaError(cudaFree(dev_sample));
	checkCudaError(cudaFree(dev_corelation));
	checkCudaError(cudaFree(dev_wavestat));
	checkCudaError(cudaFree(dev_wavestat2));
	checkCudaError(cudaFree(dev_hammingstat));
	checkCudaError(cudaFree(dev_hammingArray));

	//Time
	cudaEventCreate(&stop);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedtime,start,stop);
	fprintf(stderr,"Time spent for CUDA operation : %.10f\n",elapsedtime/(float)1000);

	//form struct array
	struct key_corr key_corrpairs[KEYS][KEYBYTESPART];
	
	//print all information while putting to structs	
	file=fopen(FILEALL,"w");
	for (i=0;i<KEYS;i++){
		for(j=0;j<KEYBYTESPART;j++){
			key_corrpairs[i][j].key=i;
			key_corrpairs[i][j].corr=corelation[i*KEYBYTESPART+j];
			fprintf(file,"%.4X : %f\t",i,corelation[i*KEYBYTESPART+j]);
		}
		fprintf(file,"\n");
	}
	
	int k;
	//sort using insertion sort
	for (j=0;j<KEYBYTESPART;j++){	
		for (i=1;i<KEYS;i++){
			double corr=key_corrpairs[i][j].corr;
			unsigned int key=key_corrpairs[i][j].key;
			for (k=(int)(i-1);k>=0 && corr>key_corrpairs[k][j].corr;k--){
				key_corrpairs[k+1][j].corr=key_corrpairs[k][j].corr;
				key_corrpairs[k+1][j].key=key_corrpairs[k][j].key;
			}
			key_corrpairs[k+1][j].key=key;
			key_corrpairs[k+1][j].corr=corr;
		}
	}
	
	//print all in ascending order
	file=fopen(FILEALLSORTED,"w");
	for (i=0;i<KEYS;i++){
		for(j=0;j<KEYBYTESPART;j++){
			fprintf(file,"%.4X : %f\t",key_corrpairs[i][j].key,key_corrpairs[i][j].corr);
		}
		fprintf(file,"\n");
	}
	
	//print the best five to  the stdout
	for (i=0;i<5;i++){
		for(j=0;j<KEYBYTESPART;j++){
			printf("%.4X\t\t",key_corrpairs[i][j].key);
		}
		printf("\n");
		for(j=0;j<KEYBYTESPART;j++){
			printf("%f\t",key_corrpairs[i][j].corr);
		}		
		printf("\n\n");
	}	
		
	return 0;
}


