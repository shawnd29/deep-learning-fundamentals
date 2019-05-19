#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include "dotp.h"

__global__ void kernel(unsigned int rows, unsigned int cols , float *matrixdata, float *vecdata, float *results){
//        unsigned char y;
//        int m, n ;
//        unsigned int p = 0 ;
//        int cases[3];
//        int controls[3];
//        int tot_cases = 1;
//        int tot_controls= 1;
//        int total = 1;
//	size_t n = sizeof(vecdata);
//        float chisquare = 0.0f;
//        float exp[3];
//        float Conexpected[3];
//        float Cexpected[3];
//        float numerator1;
//        float numerator2;	
        int tid  = threadIdx.x + blockIdx.x * blockDim.x;
	int dp  = 0;

 		for(int j=0;j<cols;j++){
		dp = dp + vecdata[j]*matrixdata[j*rows+tid];
		
       		results[tid] = dp;
		}
}
