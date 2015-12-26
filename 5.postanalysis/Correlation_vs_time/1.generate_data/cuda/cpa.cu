/*
	
	The A to Z of Building a Testbed for Power Analysis Attacks
	CUDA C source code for Correlation Power Analysis algorithm 

	This is used to generate data for plotting the variation of correlation coefficient with time
	Saves the all correlation coefficients for all key possibilities along time
	This will generate 4096 files under names subbyte<x>_keyguess<y> where <x> and <y> are integers
	
	This program needs open 4096 files at the same time.
	For that you may need log as root using "sudo su"
	then increase the maximum simultaneously opened file limit by "ulimit -n 5000"
	
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
#include "helpers.cuh"
#include "data.cuh"

/*************************************************************** CHANGEBLE PARAMETERS ****************************************************************************/
/* Rather that modifying parameters here, it is recommended to run the provided script. The script will take modifiable parameters as arguments 
and will change this source code using "sed" to generate a temporary modified source code called cpatemp.cu
*/


/* file name for all key-correlation pairs sorted in key order
This file after running would contain all the key bytes sorted in key order with their respective correlation coefficients*/
#define FILEALL "all_key_corr.txt"

/* file name for all key-correlation pairs sorted using correlation coefficient
This file after running would contain all the key bytes and their respective correlation coefficients sorted in descending order of the correlation coefficient*/
#define FILEALLSORTED "all_key_corr_sorted.txt"

/* input file format for the power trace file
put 0 for ASCII format power traces. 
put 1 to save power traces in binary. Not readable directly, but file size is less*/
#define WAVFORMAT 1

/* The number of power traces used for calculation*/
#define SAMPLES 1000

/* The number of sampling points in a single power trace
This value can be found by inspecting the stat.txt generated after collecting a set of power traces*/
#define ALLWAVELEN 100000

/*Global memory on a GPU is limited (RAM as well) and hence if the power traces are large sized, all the things won't fit at once to memory
If you experience a memory allocation failure when running, reduce WAVELENGTH value
This will force the program to read the power traces part by part 
But when using this implementation make sure that ALLWAVELENGTH is divisible by WAVELENGTH 
*/
#define WAVELENGTH 25000

/************************************************************ END OF CHANGEBLE PARAMETERS ****************************************************************************/


//defined for 128 bit AES
#define KEYBYTES 16
#define KEYS 256

//struct used for sorting correlation key pairs
struct key_corr{
	unsigned int key;
	double corr;
};

//hamming weight of a number
__device__ byte hammingweight(byte H){

	// Count the number of set bits
	byte dist=0;
	while(H){
		dist++; 
		H &= H - 1;
	}
	return dist;
}

/********************************************************************** SELECTION FUNCTION ****************************************************************************/
//This will have to be modified if your selection function/intermediate values are different

//find hamming weight for the selection function
__device__ byte hamming(unsigned int *sample, unsigned int i,unsigned int n,unsigned int key) { //n is byteno  i is the sample
    byte inter = (byte)sbox[sample[i*KEYBYTES+n] ^ key];
    byte dist = hammingweight(inter);	  
	return dist;
}
/********************************************************************END SELECTION FUNCTION ****************************************************************************/


//find the correlation values and then the maximum
__global__ void maxCorelationkernel(double *corelation,double *wavestat,double *wavestat2,double *hammingstat,double *allcorelation){
	
	int keyguess=blockDim.y*blockIdx.y+threadIdx.y;
	int keybyte=blockDim.x*blockIdx.x+threadIdx.x;

	if (keybyte<KEYBYTES && keyguess<KEYS ){ 
		
		double sigmaH,sigmaH2,sigmaW=0,sigmaW2=0,sigmaWH=0;	
		sigmaH=hammingstat[KEYBYTES*keyguess+keybyte];
		sigmaH2=hammingstat[KEYS*KEYBYTES+KEYBYTES*keyguess+keybyte];
		double temp_corelation=0;;
		double corelationmax=0;;
		unsigned int j;
		for(j=0;j<WAVELENGTH;j++){
			
			sigmaWH=wavestat2[j*KEYS*KEYBYTES + keyguess*KEYBYTES + keybyte];
			sigmaW=wavestat[j];
			sigmaW2=wavestat[WAVELENGTH+j];

			double numerator=SAMPLES*sigmaWH - sigmaW*sigmaH;
			double denominator=sqrt(SAMPLES*sigmaW2 - sigmaW*sigmaW)*sqrt(SAMPLES*sigmaH2 - sigmaH*sigmaH);
			temp_corelation=-numerator/denominator;
			allcorelation[j*KEYS*KEYBYTES + keyguess*KEYBYTES + keybyte]=temp_corelation;
			
			if(temp_corelation>corelationmax){
				corelationmax=temp_corelation;
			}
		}

		if(corelationmax>corelation[keyguess*KEYBYTES+keybyte]){
			corelation[keyguess*KEYBYTES+keybyte]=corelationmax;
		}
	}
	return;
}

//find power trace statistics such as sigmaW sigmaw^2 etc
__global__ void wavestatkernel(double *wavedata, double *wavestat,double *wavestat2,byte *hammingArray){

	int keyguess=blockDim.y*blockIdx.y+threadIdx.y;
	int keybyte=blockDim.x*blockIdx.x+threadIdx.x;
	int wave=blockDim.z*blockIdx.z+threadIdx.z;

	if (keyguess<KEYS && keybyte<KEYBYTES && wave<WAVELENGTH ){
		unsigned int i;
		double sigmaWH=0;
		for(i=0;i<SAMPLES;i++){
			sigmaWH+=wavedata[i*WAVELENGTH+wave]*(double)hammingArray[i*KEYS*KEYBYTES + keyguess*KEYBYTES + keybyte];
		}
		wavestat2[wave*KEYS*KEYBYTES + keyguess*KEYBYTES + keybyte ]=sigmaWH;
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

//find hamming weight statitics such as signaH sigmaH^2
__global__ void hammingkernel(unsigned int *sample,byte *hammingArray,double *hammingstat){
	int keyguess=blockDim.y*blockIdx.y+threadIdx.y;
	int keybyte=blockDim.x*blockIdx.x+threadIdx.x;

	if (keybyte<KEYBYTES && keyguess<KEYS ){
		double sigmaH=0,sigmaH2=0;
		byte H;
		unsigned int i;
		for(i=0;i<SAMPLES;i++){
			H=hamming(sample,i,keybyte,keyguess);
			hammingArray[i*KEYS*KEYBYTES + keyguess*KEYBYTES + keybyte]=H;
			sigmaH+=(double)H;
			sigmaH2+=(double)H*(double)H;
		}
		hammingstat[KEYBYTES*keyguess+keybyte]=sigmaH;
		hammingstat[KEYS*KEYBYTES+KEYBYTES*keyguess+keybyte]=sigmaH2;
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
	
	//allocate RAM for waves
	double *wavedata=(double *)malloc(sizeof(double) * SAMPLES*  WAVELENGTH);
	checkAllocRAM(wavedata);

	//read sample texts
	unsigned int *sample=(unsigned int *)malloc(sizeof(unsigned int)*SAMPLES*KEYBYTES);
	checkAllocRAM(sample);
		
	FILE *file=fopen(argv[2],"r");
	isFileValid(file);
	
	char str[100];
	for(i=0; i<SAMPLES ;i++){
		fscanf(file,"%s",str);
		for(j=0; j<KEYBYTES; j++){
			sscanf(&str[2*j],"%02X",&sample[i*KEYBYTES+j]);		
		}
	}
	fclose(file);


	//space in rAMfor correlation values
	double *corelation=(double *)malloc(sizeof(double) * KEYS * KEYBYTES);
	checkAllocRAM(corelation);
	
	//space for all correlations
	double *allcorelation=(double *)malloc(sizeof(double) * KEYS * KEYBYTES * WAVELENGTH);
	checkAllocRAM(allcorelation);	
	
	//Time
	cudaEvent_t start,stop;
	float elapsedtime;
	cudaEventCreate(&start);
	cudaEventRecord(start,0);
	
	//cuda arrays and copying
	double *dev_wavedata;
	unsigned int *dev_sample;
	double *dev_corelation,*dev_wavestat,*dev_wavestat2,*dev_hammingstat,*dev_allcorrelations;
	byte *dev_hammingArray;
	cudaMalloc((void**)&dev_wavedata, SAMPLES*WAVELENGTH*sizeof(double)); 				checkCudaError();
	cudaMalloc((void**)&dev_sample, SAMPLES*KEYBYTES*sizeof(unsigned int)); 			checkCudaError();
	cudaMalloc((void**)&dev_corelation, KEYS*KEYBYTES*sizeof(double)); 					checkCudaError();
	cudaMalloc((void**)&dev_hammingArray, KEYS*KEYBYTES*SAMPLES*sizeof(byte)); 			checkCudaError();
	cudaMalloc((void**)&dev_wavestat, 2*WAVELENGTH*sizeof(double)); 					checkCudaError();
	cudaMalloc((void**)&dev_wavestat2, KEYS*KEYBYTES*WAVELENGTH*sizeof(double));		checkCudaError();
	cudaMalloc((void**)&dev_hammingstat, 2*KEYS*KEYBYTES*sizeof(double)); 				checkCudaError();
	cudaMalloc((void**)&dev_allcorrelations, WAVELENGTH*KEYS*KEYBYTES*sizeof(double)); 	checkCudaError();
	
	cudaMemset(dev_corelation,0, KEYS*KEYBYTES*sizeof(double)); 						checkCudaError();
	cudaMemset(dev_allcorrelations,0, WAVELENGTH*KEYS*KEYBYTES*sizeof(double)); 		checkCudaError();
	cudaMemcpy(dev_sample,sample, SAMPLES*KEYBYTES*sizeof(unsigned int),cudaMemcpyHostToDevice); checkCudaError();

	//cuda kernel configuraion parameters	
	dim3 grid(KEYBYTES/16,KEYS/16);
	dim3 block(16,16);

	//find hamming statistics
	hammingkernel<<<grid,block>>>(dev_sample,dev_hammingArray,dev_hammingstat);
	cudaDeviceSynchronize(); checkCudaError();

	//correlation value writing. Opening files
	char filename[100];
	FILE* filec[KEYBYTES][KEYS];
		
	int keyguess,keybyte;
	for(keybyte=0;keybyte<KEYBYTES;keybyte++){
		for(keyguess=0;keyguess<KEYS;keyguess++){
			sprintf(filename,"subbyte%d_keyguess%d",keybyte,keyguess);
			filec[keybyte][keyguess]=fopen(filename,"w");		
		}
	}	
	
	//start calculations	
	int loops=0;
	for(loops=0;loops<ALLWAVELEN/WAVELENGTH;loops++){

		if(WAVFORMAT==0){
	
			//read wave data
			FILE *file=fopen(argv[1],"r");
			isFileValid(file);
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
		
		}
		
		else if(WAVFORMAT==1){
			//read wave data
			FILE *file=fopen(argv[1],"rb");
			isFileValid(file);
			for(i=0; i<SAMPLES ;i++){
				fseek(file,sizeof(float)*(i*ALLWAVELEN+WAVELENGTH*loops),SEEK_SET);
				for(j=0; j<WAVELENGTH; j++){
					float dat;
					int ret=fread(&dat,sizeof(float),1,file);
					if(ret<1){
						perror("");
						exit(1);
					}
					wavedata[i*WAVELENGTH+j]=(double)dat;
				}
			}	
			fclose(file);			
			
		}

		else{
			
			fprintf(stderr,"Unknown wave file format\n");
			exit(1);
		}

		//copy wavedata to cuda
		cudaMemcpy(dev_wavedata,wavedata,SAMPLES*WAVELENGTH*sizeof(double),cudaMemcpyHostToDevice); checkCudaError();
	
		//cuda kernel configuration parameters	
		dim3 block3d(16,16,4);
		dim3 grid3d(KEYBYTES/16,KEYS/16,WAVELENGTH/4);
		
		//find wave statistics
		wavestatkernel<<<grid3d,block3d>>>(dev_wavedata,dev_wavestat,dev_wavestat2,dev_hammingArray);
		cudaDeviceSynchronize(); checkCudaError();


		//deploy the correlation calculation and max finding kernel
		maxCorelationkernel<<<grid,block>>>(dev_corelation,dev_wavestat,dev_wavestat2,dev_hammingstat,dev_allcorrelations);
		cudaDeviceSynchronize(); checkCudaError();	
		
		//copy intermediate temp valies
		cudaMemcpy(allcorelation,dev_allcorrelations,WAVELENGTH*KEYS*KEYBYTES*sizeof(double),cudaMemcpyDeviceToHost); checkCudaError();
		
		//write correlation values	
		int point,keyguess,keybyte;
		for(keybyte=0;keybyte<KEYBYTES;keybyte++){
			for(keyguess=0;keyguess<KEYS;keyguess++){		
				for(point=0;point<WAVELENGTH;point++){
						fprintf(filec[keybyte][keyguess],"%f ",allcorelation[point*KEYS*KEYBYTES + keyguess*KEYBYTES + keybyte]);
				}	
				if(loops==ALLWAVELEN/WAVELENGTH-1){
					fprintf(filec[keybyte][keyguess],"\n");				
					fclose(filec[keybyte][keyguess]);
				}
			}
		}		

		//progress
		fprintf(stderr,"%d of %d completed\n",loops+1,ALLWAVELEN/WAVELENGTH);		

	}

	//copy back results
	cudaMemcpy(corelation,dev_corelation,KEYS*KEYBYTES*sizeof(double),cudaMemcpyDeviceToHost); checkCudaError();
	cudaFree(dev_wavedata); 						checkCudaError();
	cudaFree(dev_allcorrelations); 					checkCudaError();
	cudaFree(dev_sample); 							checkCudaError();
	cudaFree(dev_corelation); 						checkCudaError();
	cudaFree(dev_wavestat);					 		checkCudaError();
	cudaFree(dev_wavestat2); 						checkCudaError();
	cudaFree(dev_hammingstat); 						checkCudaError();
	cudaFree(dev_hammingArray); 					checkCudaError();
	
	//Time
	cudaEventCreate(&stop);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedtime,start,stop);
	fprintf(stderr,"Time spent for CUDA operation : %.10f\n",elapsedtime/(float)1000);

	//form struct array
	struct key_corr key_corrpairs[KEYS][KEYBYTES];
	
	//print all information while putting to structs
	file=fopen(FILEALL,"w");
	for (i=0;i<KEYS;i++){
		for(j=0;j<KEYBYTES;j++){
			key_corrpairs[i][j].key=i;
			key_corrpairs[i][j].corr=corelation[i*KEYBYTES+j];
			fprintf(file,"%.2X : %f\t",i,corelation[i*KEYBYTES+j]);
		}
		fprintf(file,"\n");
	}
	fclose(file);
	
	int k;
	//sort using insertion sort
	for (j=0;j<KEYBYTES;j++){	
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
	
	//print all in desceding order
	file=fopen(FILEALLSORTED,"w");
	for (i=0;i<KEYS;i++){
		for(j=0;j<KEYBYTES;j++){
			fprintf(file,"%.2X : %f\t",key_corrpairs[i][j].key,key_corrpairs[i][j].corr);
		}
		fprintf(file,"\n");
	}
	
	//print the best five to  the stdout
	for (i=0;i<5;i++){
		for(j=0;j<KEYBYTES;j++){
			printf("%.2X\t\t\t",key_corrpairs[i][j].key);
		}
		printf("\n");
		for(j=0;j<KEYBYTES;j++){
			printf("%f\t",key_corrpairs[i][j].corr);
		}		
		printf("\n\n");
	}	
	
	return 0;
}


