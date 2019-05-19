/*This assignment is used to find the dot product of a matrix in Cuda using parallel computing

Done by Shawn D'Souza - srd59
*/


#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <time.h>
#include <sys/time.h>
//#include <cuda_runtime_api.h>
#include "dotp.h"


int main(int argc ,char* argv[]) {

        FILE *fp;
	FILE *vfp;
        size_t size;

/** Initialize rows, cols, ncases, ncontrols from the user */
        unsigned int rows=atoi(argv[1]);
        unsigned int cols=atoi(argv[2]);
        int CUDA_DEVICE = atoi(argv[5]);
        int THREADS = atoi(argv[6]);


        cudaError err = cudaSetDevice(CUDA_DEVICE);
        if(err != cudaSuccess) { printf("Error setting CUDA DEVICE\n"); exit(EXIT_FAILURE); }

/** Host variable declaration */

        //int THREADS = 32;
        int BLOCKS;
        float* host_results = (float*) malloc((size_t)size * sizeof(float));
        struct timeval starttime, endtime;
        clock_t start, end;
        float seconds;
        unsigned int jobs;
        unsigned long i;
        unsigned long ulone = 1;
        unsigned long ultwo = 2;

/** Kernel variable declaration */
        float *dev_dataT;
	float *dev_vecT;
        float *results;

/** Validation to check if the data file is readable */
        fp = fopen(argv[3], "r");
        if (fp == NULL) {
                printf("Cannot Open the File");
                return 0;
        }

	size = (size_t)((size_t)rows * (size_t)cols);
        fflush(stdout);
        float *dataT = (float*)malloc((size_t)sizeof(float)*(size_t)size);

        if(dataT == NULL) {
                printf("ERROR: Memory for data not allocated.\n");
        }


/** Transfer the matrix Data from the file to CPU Memory */
        i=0;

            float** matrix=(float**)malloc((size_t)size*sizeof(float));
            for(int i=0;i<rows;++i)
            matrix[i]=(float*)malloc((size_t)size*sizeof(float));
    		for(int i = 0; i < rows; i++)
       		 {
        	    for(int j = 0; j < cols; j++)
           	 {
                	if (!fscanf(fp, "%f", &matrix[j][i]))
               		break;
  
           	 }
    			}
	  fclose(fp);


/**convert the 2d array to a 1d array*/
	int k=0;
        for(int i = 0; i < rows; i++)
        {
          for(int j = 0; j < cols; j++)
          {
              dataT[k] = matrix[i][j];
              k++;
              
          }
        }



/** Validation to check if the vector file is readable */
        vfp = fopen(argv[4], "r");
        if (vfp == NULL) {
                printf("Cannot Open the File");
                return 0;
        }


        fflush(stdout);

        float *vecT = (float*)malloc((size_t)size);

        if(vecT == NULL) {
                printf("ERROR: Memory for data not allocated.\n");
        }

/** Transfer the vector Data from the file to CPU Memory */ 
	for (i=0;i<cols;i++)
	{
          if (!fscanf(vfp, "%f", &vecT[i]))     
	break;	
	}

        fclose(vfp);


 //   printf("read data\n");
        fflush(stdout);
/** Allocate the Memory in the GPU for matrix data */
 
        err = cudaMalloc((float**) &dev_dataT, (size_t) size * (size_t) sizeof(float) );
        if(err != cudaSuccess) { printf("Error mallocing data on GPU device\n"); }
 
        err = cudaMalloc((float**) &dev_vecT, (size_t) size * (size_t) sizeof(float) );
        if(err != cudaSuccess) { printf("Error mallocing data on GPU device\n"); }


 //        gettimeofday(&starttime, NULL);
        err = cudaMalloc((float**) &results, (size_t) size * sizeof(float) );
        if(err != cudaSuccess) { printf("Error mallocing results on GPU device\n"); }
 

/** Copy the matrix data to GPU */
        err = cudaMemcpy(dev_dataT, dataT, (size_t)size * (size_t)sizeof(float), cudaMemcpyHostToDevice);
        if(err != cudaSuccess) { printf("Error copying data to GPU\n"); }
        err = cudaMemcpy(dev_vecT, vecT, (size_t)size * (size_t)sizeof(float), cudaMemcpyHostToDevice); // copying vector data to GPU
        if(err != cudaSuccess) { printf("Error copying data to GPU\n"); }


        jobs = cols;
        BLOCKS = (jobs + THREADS - 1)/THREADS;


/** Calling the kernel function */
	kernel<<<BLOCKS,THREADS>>>(rows,cols,dev_dataT,dev_vecT,results);




/** Copy the results back in host*/
        cudaMemcpy(host_results,results, (size_t)size* (size_t)sizeof(float),cudaMemcpyDeviceToHost);
	printf("The values of the dot matrix are:");
        for(int k = 0; k < rows; k++) {
             printf("\n");   
             printf("%f ", host_results[k]);
        }
	printf("\n");

        cudaFree( dev_dataT );
	cudaFree( dev_vecT );
        cudaFree( results );


        return 0;

}


