/*
	
	The A to Z of Building a Testbed for Power Analysis Attacks
	parallel C source code (based on pthread) for Correlation Power Analysis algorithm 

	Takes the set of plain text used and the power traces as the inputs.
	Apply Pearson correlation between hypothetical power data calculated through hamming weight
	and real power data in collected power traces.
	Prints the most correlated keys and their respective correlation coefficients
	
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
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include "helpers.h"
#include "data.h"

/*************************************************************** CHANGEBLE PARAMETERS ****************************************************************************/
/* Rather that modifying parameters here, it is recommended to run the provided script. The script will take modifiable parameters as arguments 
and will change this source code using "sed" to generate a temporary modified source code called cpatemp.cu
*/

/* file name for all key-correlation pairs sorted in key order
This file after running would contain all the key bytes sorted in key order with their respective correlation coefficients*/
#define FILEALL "all_corr.txt"

/*number of threads to be launched*/
#define THREADS 32

/* input file format for the power trace file
put 0 for ASCII format power traces. 
put 1 to save power traces in binary. Not readable directly, but file size is less*/
#define WAVFORMAT 1

/* The number of power traces used for calculation*/
#define SAMPLES 1000

/* The number of sampling points in a single power trace
This value can be found by inspecting the stat.txt generated after collecting a set of power traces*/
#define WAVELENGTH 100000

/************************************************************ END OF CHANGEBLE PARAMETERS ****************************************************************************/


//defined for 128 bit AES
#define KEYBYTES 16
#define KEYS 256


//struct used for sorting correlation key pairs
struct key_corr{
	unsigned int key;
	double corr;
};

//for storing correlation values
double corelation[KEYS][KEYBYTES];

//argument struct for pthread
struct args{
	int position;
	double **wavedata;
	unsigned int **sample;
	unsigned int keyguess;
	unsigned int keybyte;
};


//calculates hamming weight of a 8 bit number
byte hammingweight(byte H){
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
byte hamming(unsigned int sample[], unsigned int n,unsigned int key) { //n is byteno sample is the line in sample text
    byte inter = (byte)sbox[sample[n] ^ key];
    byte dist = hammingweight(inter);		
	return dist;
}

/********************************************************************END SELECTION FUNCTION ****************************************************************************/


//do the CPA
double maxCorelation(double **wavedata, unsigned int **sample, unsigned int keyguess, unsigned int keybyte){
	  byte hammingArray[SAMPLES];
	  byte H;
	  double sigmaWH=0,sigmaW=0,sigmaH=0,sigmaW2=0,sigmaH2=0,W;	  
	  unsigned int i;
	  for(i=0;i<SAMPLES;i++){
		  H=hamming(sample[i],keybyte,keyguess);
		  hammingArray[i]=H;
		  sigmaH+=(double)H;
		  sigmaH2+=(double)H*H;
	  }
	  
	  double corelation=0,maxcorelation=0;
	  unsigned int j;
	  for(j=0;j<WAVELENGTH;j++){
		sigmaW=0;sigmaW2=0;sigmaWH=0;	
		for(i=0;i<SAMPLES;i++){
			W=wavedata[i][j];
			sigmaW+=W;
			sigmaW2+=W*W;
			sigmaWH+=W*(double)hammingArray[i];
		}
		double numerator=SAMPLES*sigmaWH - sigmaW*sigmaH;
		double denominator=sqrt(SAMPLES*sigmaW2 - sigmaW*sigmaW)*sqrt(SAMPLES*sigmaH2 - sigmaH*sigmaH);
		corelation=fabs(numerator/denominator);
		if(corelation>maxcorelation){
			maxcorelation=corelation;
		}

	  }
	  return maxcorelation;
}

//work for a single thread
void *thread_process(void *voidargs){
	int t,i,j;
	struct args *myargs=(struct args *)voidargs;
	int count=0;
	for(t=myargs->position;t<(myargs->position)+KEYBYTES*KEYS/THREADS;t++){
		i=t/KEYBYTES;
		j=t-t/KEYBYTES*KEYBYTES;
		corelation[i][j]=maxCorelation(myargs->wavedata, myargs->sample, i, j);
		if(myargs->position==0){
			fprintf(stderr,"%d of %d completed\n",count,(KEYBYTES*KEYS/THREADS));
		}
		count++;
	}
	//fprintf(stderr,"Thread %d done\n",(myargs->position)/THREADS);
	pthread_exit(0);
}

int main(int argc, char *argv[]){
	
	unsigned int i,j;
		
	//check args
	if(argc!=3){
		fprintf(stderr,"%s\n", "Not enough args. eg ./cpa <power_trace_file> <plain_text_samples_file>");
		exit(EXIT_FAILURE);
	}
	
	//allocate RAM for power traces
	double **wavedata=malloc(sizeof(double*) * SAMPLES);
	checkMalloc(wavedata);
	for (i=0; i<SAMPLES; i++){
		wavedata[i]=malloc(sizeof(double) * WAVELENGTH);
		checkMalloc(wavedata[i]);
	}
	
	//read power traces
	if(WAVFORMAT==0){	
		
		FILE *file=openFile(argv[1],"r");
		for(i=0; i<SAMPLES ;i++){
			for(j=0; j<WAVELENGTH; j++){
				float dat;
				fscanf(file,"%f",&dat);
				wavedata[i][j]=(double)dat;
			}
		}
		fclose(file);
		
	}
	
	else if(WAVFORMAT==1){

			FILE *file=openFile(argv[1],"rb");
			for(i=0; i<SAMPLES ;i++){
				for(j=0; j<WAVELENGTH; j++){
					float dat;
					int ret=fread(&dat,sizeof(float),1,file);
					if(ret<1){
						perror("");
						exit(1);
					}
					wavedata[i][j]=(double)dat;
				}
			}	
			fclose(file);			
			
		}

		else{
			
			fprintf(stderr,"Unknown wave file format\n");
			exit(1);
		}
		
	//allocate RAM for sample texts
	unsigned int **sample=malloc(sizeof(unsigned int*)*SAMPLES);
	checkMalloc(sample);
	for (i=0; i<SAMPLES; i++){
		sample[i]=malloc(sizeof(unsigned int)*KEYBYTES);
		checkMalloc(sample[i]);
	}
	
	//read the plain text samples
	FILE *file=openFile(argv[2],"r");
	char str[100];
	for(i=0; i<SAMPLES ;i++){
		fscanf(file,"%s",str);
		for(j=0; j<KEYBYTES; j++){
			sscanf(&str[2*j],"%02X",&sample[i][j]);		
		}
	}
	fclose(file);	
	
	//start time measurements
	time_t start=time(NULL);

	//create threads
	pthread_t thread[THREADS];
	int t,position=0,ret;
	for(t=0;t<THREADS;t++){
		struct args *myargs=malloc(sizeof(struct args));
			myargs->wavedata=wavedata;
			myargs->sample=sample;
			myargs->position=position;
			position+=KEYBYTES*KEYS/THREADS;
			ret=pthread_create(&thread[t],NULL,thread_process,(void*)(myargs));
			fprintf(stderr,"Thread %d created\n",t);	
			if(ret==-1){
				perror("Some error in thread creation");
				exit(EXIT_FAILURE);
			}					
	}
	
	//pthread joining
	for(t=0;t<THREADS;t++){
		int ret=pthread_join(thread[t],NULL);
		if(ret!=0){
			perror("Join error");
			exit(EXIT_FAILURE);
		}
	fprintf(stderr,"Thread %d joined\n",t);
	}
	
	//time measurement
	time_t stop=time(NULL);
	double cputime=(double)((stop-start));
    fprintf(stderr,"Time spent for operation : %.10f seconds\n",cputime);
	
	//form struct array for sorting
	struct key_corr key_corrpairs[KEYS][KEYBYTES];

	//print all correlation values to a file while putting to structs to be later sorted
	file=openFile(FILEALL,"w");
	for (i=0;i<KEYS;i++){
		for(j=0;j<KEYBYTES;j++){
			key_corrpairs[i][j].key=i;
			key_corrpairs[i][j].corr=corelation[i][j];			
			fprintf(file,"%f\t",corelation[i][j]);
		}
		fprintf(file,"\n");
	}
	
	int k;
	//sort based on the correlation coefficient using insertion sort
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
	
	//print the best five correlated keys to the stdout
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

