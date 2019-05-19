#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "openmp_dp.h"

void kernel(unsigned int rows, unsigned int cols , float *matrixdata, float *vecdata, float *results,unsigned int jobs){
	int tid = omp_get_thread_num();
        int dp  = 0;
        int i,j, stop;

        if((tid+1)*jobs > rows) stop=rows;
        else stop = (tid+1)*jobs;
	
        
	for(j=tid*jobs;j<stop;j++){
		dp=0;
	for(i=0;i<cols;i++){
                dp = dp + vecdata[i]*matrixdata[i*rows +j];
		}
                results[j] = dp;
                }

 }
                      

