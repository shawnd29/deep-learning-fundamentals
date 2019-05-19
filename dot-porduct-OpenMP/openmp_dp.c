/*
This assignment calculates the dot product using OpenMP
done by Shawn Rahul D'Souza 
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include "openmp_dp.h"

int main(int argc ,char* argv[]) {

	FILE *fp;
	FILE *vp;
	size_t size;
	  
	unsigned int rows=atoi(argv[1]);
	unsigned int cols=atoi(argv[2]);
	int nprocs = atoi(argv[5]);
	
	

/*Host variable declaration */
	float* host_results = (float*) malloc(rows * sizeof(float)); 
	struct timeval starttime, endtime;
	clock_t start, end;
	float seconds = 0;
	unsigned int jobs; 
	unsigned long i;

	/*Kernel variable declaration */
    size_t len = 0;
	float matrix[rows][cols];
	float var ;
	int vrow =1;

	start = clock();

/* Validation to check if the data file is readable */
	fp = fopen(argv[3], "r");
	vp = fopen(argv[4],"r");
	
	if (fp == NULL) {
    		printf("Cannot Open the File");
		return 0;
	}
	if (vp == NULL){
		printf("cannot open the file");
	}
	size = (size_t)((size_t)rows * (size_t)cols);
	size_t sizeV = 0;
	sizeV = (size_t)((size_t)vrow*(size_t)cols);

	
	fflush(stdout);

	float *dataT = (float*)malloc((size)*sizeof(float));
	float *vecT = (float*)malloc((sizeV) * sizeof(float));

	if(dataT == NULL) {
	        printf("ERROR: Memory for data not allocated.\n");
	}
	
	if(vecT == NULL){
		printf("ERROR: Memory for data not allocated. \n");
	}
        gettimeofday(&starttime, NULL);
	int j = 0;

/* Transfer the Data from the file to CPU Memory */
        for (i =0; i< rows;i++){
		for(j=0; j<cols ; j++){
			fscanf(fp,"%f",&var);
                        matrix[i][j]=var;
		}
	}
	for (i =0;i<cols;i++){
		for(j= 0; j<rows; j++){
			dataT[rows*i+j]= matrix[j][i];
		}
	}		

		for (j=0;j<cols;j++){
			fscanf(vp,"%f",&vecT[j]);
		}
   
	fclose(fp);
	fclose(vp);

        fflush(stdout);

        gettimeofday(&endtime, NULL);
        seconds+=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);

        

/*define jobs*/	
	jobs = (unsigned int)((rows +nprocs -1)/nprocs);
	

        gettimeofday(&starttime, NULL);

        
/*Invoke kernel*/
        #pragma omp parallel num_threads(nprocs)
        kernel(rows,cols,dataT,vecT,host_results,jobs);
        
	gettimeofday(&endtime, NULL); seconds=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);
//	printf("time for kernel=%f\n", seconds);
	
/*print results*/
	printf("The dot product is :");
	printf("\n");
	int k;
	for(k = 0; k < rows; k++) {
		printf("%f ", host_results[k]);
		printf("\n");
	}
	printf("\n");

	end = clock();
	seconds = (float)(end - start) / CLOCKS_PER_SEC;
	

	return 0;

}

