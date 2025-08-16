#include "main.h"

int		*m_nCurBufferPtr;

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
int GetMin(int a, int b)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	return ((a<b) ? a : b);
}


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void JenkFish(structData *data, structData **dataBreak, int nClusters, int nCount)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structData	*dataNew;
	structData	temp;
	structData	*dataCumulate;
	float		*fPrevSquaredMean;
	float		*fCurSquaredMean;
	float		*fArray;
	float		cwv;
	int			w;
	int			cw;
	int			*nCurBuffer;
	int			nBufferSize;
	int			nNewCount;
	int			nCompleteRows;
	int			nLastClassBreakIndex;
	int			i, j, k;

	for (i = 0; i < nCount - 1; i++)
	{
		for (j = 0; j < nCount - i - 1; j++)
		{
			if (data[j].fOutput > data[j + 1].fOutput)
			{
				temp = data[j];
				data[j] = data[j + 1];
				data[j + 1] = temp;
			}
		}
	}

	nNewCount = 0;
	dataNew = (structData *)calloc(1, sizeof(structData));

	dataNew[nNewCount].nID = data[0].nID;
	dataNew[nNewCount].fOutput = data[0].fOutput;
	dataNew[nNewCount].nCount = 1;

	for (i = 1; i < nCount; i++)
	{
		if (data[i].fOutput == dataNew[nNewCount].fOutput)
		{
			++dataNew[nNewCount].nCount;
		}
		else
		{
			++nNewCount;
			dataNew = (structData *)realloc(dataNew, ((nNewCount + 1) * sizeof(structData)));

			dataNew[nNewCount].nID = data[i].nID;
			dataNew[nNewCount].fOutput = data[i].fOutput;
			dataNew[nNewCount].nCount = 1;
		}
	}

	++nNewCount;

	nBufferSize = nNewCount - (nClusters - 1);
	//printf("Buffer Size: %d\n", nBufferSize);

	if (nBufferSize < 2)
	{
		nBufferSize = 2;
		*dataBreak = (structData *)calloc(nClusters, sizeof(structData));
	}
	else
	{
		dataCumulate = (structData *)calloc(nNewCount, sizeof(structData));
		*dataBreak = (structData *)calloc(nClusters, sizeof(structData));
		fPrevSquaredMean = (float *)calloc(nBufferSize, sizeof(float));
		fCurSquaredMean = (float *)calloc(nBufferSize, sizeof(float));
		nCurBuffer = (int *)calloc((nBufferSize * (nClusters - 1)), sizeof(int));


		cw = 0;
		cwv = 0.0f;

		for (i = 0; i < nNewCount; ++i)
		{
			w = dataNew[i].nCount;
			cw += w;

			cwv += (float)w * dataNew[i].fOutput;

			dataCumulate[i].fOutput = cwv;
			dataCumulate[i].nCount = cw;

			if (i < nBufferSize)
				fPrevSquaredMean[i] = cwv * cwv / cw; // prepare SSM for fOutput class. Last (k-1) values are omitted
		}

		k = nClusters;

		if (k > 1)
		{
			nCompleteRows = CalculateAll_JenkFish(dataCumulate, fPrevSquaredMean, fCurSquaredMean, nCurBuffer, nClusters, nBufferSize);
			nLastClassBreakIndex = FindMaxBreakIndex_JenkFish(dataCumulate, fPrevSquaredMean, fCurSquaredMean, nBufferSize - 1, 0, nBufferSize, nCompleteRows);

			while (--k)
			{
				(*dataBreak)[k].nID = dataNew[nLastClassBreakIndex + k].nID;
				(*dataBreak)[k].fOutput = dataNew[nLastClassBreakIndex + k].fOutput;
				(*dataBreak)[k].nCount = dataNew[nLastClassBreakIndex + k].nCount;

				if (k > 1)
				{
					m_nCurBufferPtr -= nBufferSize;
					nLastClassBreakIndex = m_nCurBufferPtr[nLastClassBreakIndex];
				}
			}
		}

		(*dataBreak)[0].nID = dataNew[0].nID;
		(*dataBreak)[0].fOutput = dataNew[0].fOutput;
		(*dataBreak)[0].nCount = dataNew[0].nCount;

		free(dataNew);
		free(dataCumulate);
		free(fPrevSquaredMean);
		free(fCurSquaredMean);
		free(nCurBuffer);

		for (i = 1, j = 0; i < nClusters; ++i)
		{
			(*dataBreak)[i - 1].nCount = 0;

			(*dataBreak)[i - 1].fStart = data[j].fOutput;
			(*dataBreak)[i - 1].fStartInput = data[j].fInput;

			for (; j < nCount; ++j)
			{
				if (data[j].fOutput == (*dataBreak)[i].fOutput)
				{
					break;
				}

				(*dataBreak)[i - 1].fEnd = data[j].fOutput;
				(*dataBreak)[i - 1].fEndInput = data[j].fInput;
				++(*dataBreak)[i - 1].nCount;
			}
		}

		(*dataBreak)[i - 1].nCount = 0;
		(*dataBreak)[i - 1].fStart = data[j].fOutput;
		(*dataBreak)[i - 1].fStartInput = data[j].fInput;

		for (; j < nCount; ++j)
		{
			++(*dataBreak)[i - 1].nCount;
			(*dataBreak)[i - 1].fEnd = data[j].fOutput;
			(*dataBreak)[i - 1].fEndInput = data[j].fInput;
		}


		for (i = 0, k = 0; i < nClusters; ++i)
		{
			fArray = (float *)calloc((*dataBreak)[i].nCount, sizeof(float));

			for (j = 0; j < (*dataBreak)[i].nCount; ++j, ++k)
			{
				fArray[j] = data[k].fOutput;
			}

			CalculateStandardDeviationArray(fArray, &(*dataBreak)[i].fSD, &(*dataBreak)[i].fAverage, (*dataBreak)[i].nCount);

			free(fArray);
		}
	}

	return;
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
int CalculateAll_JenkFish(structData *dataCumulate, float *fPrevSquaredMean, float *fCurSquaredMean, int *nCurBuffer, int nClusters, int nBufferSize)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	float	*dTemp;
	int		nCompleteRows=0;
	
	if (nClusters >= 2)
	{
		m_nCurBufferPtr = &nCurBuffer[0];
		for (nCompleteRows=1; nCompleteRows < nClusters - 1; ++nCompleteRows)
		{
			CalculateRange_JenkFish(dataCumulate, 0, nBufferSize, 0, nBufferSize, nCompleteRows, fPrevSquaredMean, fCurSquaredMean);

			dTemp=fPrevSquaredMean;
			fPrevSquaredMean=fCurSquaredMean;
			fCurSquaredMean=dTemp;

			m_nCurBufferPtr += nBufferSize;
		}
	}

	return(nCompleteRows);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void CalculateRange_JenkFish(structData *dataCumulate, int nBIndex, int nEIndex, int nClusterPoint, int nEPoint, int nCompleteRows, float *fPrevSquaredMean, float *fCurSquaredMean)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	int nMidIndex;
	int nMidPoint;

	if (nBIndex == nEIndex)
		return; 

	nMidIndex = (nBIndex + nEIndex)/2;
	nMidPoint = FindMaxBreakIndex_JenkFish(dataCumulate, fPrevSquaredMean, fCurSquaredMean, nMidIndex, nClusterPoint, GetMin(nEPoint, nMidIndex+1), nCompleteRows);
		
	CalculateRange_JenkFish(dataCumulate, nBIndex, nMidIndex, nClusterPoint, GetMin(nMidIndex, nMidPoint+1), nCompleteRows, fPrevSquaredMean, fCurSquaredMean); 
	m_nCurBufferPtr[ nMidIndex ] = nMidPoint; // store result for the middle element.
	CalculateRange_JenkFish(dataCumulate, nMidIndex+1, nEIndex, nMidPoint, nEPoint, nCompleteRows, fPrevSquaredMean, fCurSquaredMean);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
int FindMaxBreakIndex_JenkFish(structData *dataCumulate, float *fPrevSquaredMean, float *fCurSquaredMean, int nIndex, int nClusterPoint, int nEPoint, int nCompleteRows)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	float	fMinSquaredMean = fPrevSquaredMean[nClusterPoint] + GetSquaredMean_JenkFish(dataCumulate, nClusterPoint+nCompleteRows, nIndex+nCompleteRows);
	float	fSquaredMean;
	int		nFoundPoint = nClusterPoint;

	while (++nClusterPoint < nEPoint)
	{
		fSquaredMean = fPrevSquaredMean[nClusterPoint] + GetSquaredMean_JenkFish(dataCumulate, nClusterPoint + nCompleteRows, nIndex + nCompleteRows);
		if (fSquaredMean > fMinSquaredMean)
		{
			fMinSquaredMean = fSquaredMean;
			nFoundPoint = nClusterPoint;
		}
	}
	
	fCurSquaredMean[nIndex] = fMinSquaredMean;
	
	return nFoundPoint;
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
float GetSquaredMean_JenkFish(structData *dataCumulate, int a, int b)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	float fWeightValue = GetWeightValue_JenkFish(dataCumulate, a, b);	
	return(fWeightValue * fWeightValue / (float)GetWeight_JenkFish(dataCumulate, a, b));
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
float GetWeightValue_JenkFish(structData *dataCumulate, int a, int b)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	float fWeightValue = dataCumulate[b].fOutput;
	fWeightValue -= dataCumulate[a-1].fOutput;
	return(fWeightValue);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
int GetWeight_JenkFish(structData *dataCumulate, int a, int b)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	int nWeight = dataCumulate[b].nCount;
	nWeight -= dataCumulate[a-1].nCount;
	return(nWeight);
}
