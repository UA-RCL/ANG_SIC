#include "main.h"

extern fx32	*glb_fxConvolveRAM_0;
extern fx32	*glb_fxConvolveRAM_1;
extern int	glb_nID;




/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void BackPropagate_Convolution(structCLN *clnData, structLayer *layerData, fx32 *fxInputArray)
{
	//fx32	*fxOutputArray = NULL;
	//fx32	*fxInputDifferentials = NULL;
	//fx32	*fxOutputDifferentials = NULL;
	//fx32	fxError;
	//fx32	fxErrorSum;
	//fx32	*fxErrorArray;
	//int		nInputWindowSizeSquared;
	//int		nOutputWindowSizeSquared;
	//int		nErrorIndex = 0;
	//int		nFilterArrayIndex = 0;
	//int		nInputCount;
	//int		nOutputWindowSize;
	//int		nInputMemoryOffset;
	//int		nOffset;
	//int		i, j, k, n;
	//int		p, q, r, s;
	//int		b;

	//if (layerData->nLayerType == SINGLE_CONV_LAYER)
	//{
	//	fxOutputArray = fxInputArray;
	//	fxInputDifferentials = layerData->next->fxDifferentials;
	//	nInputCount = 1;
	//	nOutputWindowSize = layerData->nInputSize;
	//	nInputMemoryOffset = layerData->nKernel * layerData->nKernel;
	//	nOffset = layerData->next->nOffset;

	//}
	//else if (layerData->nLayerType == MULTIPLE_CONV_LAYER)
	//{
	//	fxOutputArray = layerData->prev->fxOutputArray;
	//	fxInputDifferentials = layerData->next->fxDifferentials;
	//	nInputCount = layerData->prev->nKernelCount;
	//	nOutputWindowSize = layerData->prev->nOutputWindowSize;
	//	nInputMemoryOffset = (layerData->nKernel * layerData->nKernel * layerData->prev->nKernelCount);
	//	nOffset = 0;
	//}



	//if ((fxErrorArray = (fx32 *)calloc((nInputCount * layerData->nKernel * layerData->nKernel), sizeof(fx32))) == NULL)
	//	exit(0);

	//nInputWindowSizeSquared = layerData->nOutputWindowSize * layerData->nOutputWindowSize;
	//nOutputWindowSizeSquared = nOutputWindowSize * nOutputWindowSize;


	//for (i = 0, b = 0; i < layerData->nKernelCount; ++i)
	//{
	//	for (j = 0, fxErrorSum = 0; j <= layerData->nEnd; j += layerData->nStride)
	//	{
	//		for (k = 0; k <= layerData->nEnd; k += layerData->nStride)
	//		{
	//			//printf("***** %d\n", b);

	//			fxError = MULT_FX32(DSigmoid(layerData->fxOutputArray[b]), fxInputDifferentials[b++]);
	//			fxErrorSum += fxError;

	//			for (p = 0, nErrorIndex = 0; p < nInputCount; ++p)
	//			{
	//				s = (p * nOutputWindowSizeSquared) + k;

	//				for (q = j; q < layerData->nKernel + j; ++q)
	//				{
	//					n = (q * nOutputWindowSize) + s;

	//					for (r = n; r < layerData->nKernel + n; ++r, ++nErrorIndex)
	//					{
	//						if (!j && !k)
	//							fxErrorArray[nErrorIndex] = MULT_FX32(fxOutputArray[r], fxError);
	//						else
	//							fxErrorArray[nErrorIndex] += MULT_FX32(fxOutputArray[r], fxError);

	//						//printf("%d,%d\t", nErrorIndex, r);

	//						if (layerData->fxDifferentials != NULL)
	//						{
	//							if (!i)
	//								layerData->fxDifferentials[r] = MULT_FX32(layerData->fxFilterArray[nFilterArrayIndex + nErrorIndex], fxError);
	//							else
	//								layerData->fxDifferentials[r] += MULT_FX32(layerData->fxFilterArray[nFilterArrayIndex + nErrorIndex], fxError);
	//						}
	//					}

	//					//printf("\n");
	//				}

	//				//printf("\n");

	//			}

	//			if (k == layerData->nEnd)
	//				break;
	//			else if (k + layerData->nStride > layerData->nEnd)
	//				k = layerData->nInputSize - layerData->nKernel - layerData->nStride;
	//		}

	//		if (j == layerData->nEnd)
	//			break;
	//		else if (j + layerData->nStride > layerData->nEnd)
	//			j = layerData->nInputSize - layerData->nKernel - layerData->nStride;
	//	}

	//	layerData->fxBiasArray[i] -= MULT_FX32(clnData->fxLearningRate, fxErrorSum);

	//	for (p = 0; p < nInputMemoryOffset; ++p)
	//		layerData->fxFilterArray[nFilterArrayIndex + p] -= MULT_FX32(clnData->fxLearningRate, fxErrorArray[p]);


	//	nFilterArrayIndex += nInputMemoryOffset;
	//}

	//free(fxErrorArray);
}
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/


