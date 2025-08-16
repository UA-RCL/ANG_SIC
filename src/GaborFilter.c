#include "main.h"

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void CreateGaborArray_GaborFilter(structLayer *layerData)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	float	fDegree;
	float	fSigma;
	float	fTheta;
	float	fCosRad;
	float	fSinRad;
	float	fX, fY;
	float	fJ, fK;
	float	fMax;
	float	fMin;
	int	nKernelCount = layerData->nKernelCount;

	int		nHeight = layerData->nKernelRowCount;
	int 	nWidth = layerData->nKernelColumnCount;
	float	fGamma = layerData->fGamma;
	float	fLambda = layerData->fLambda;


	float	fFilterHeight;
	float	fFilterWidth;
	float	fIncrement;

	int		nIndex;
	int		i, p, q;

	layerData->fGamma = 0.1f;
	layerData->fLambda = 0.1f;

	fGamma = layerData->fGamma;
	fLambda = layerData->fLambda;

	fSigma = 0.5F;


	fFilterHeight = -(float)(nHeight - 1) / 2;
	fFilterWidth = -(float)(nWidth - 1) / 2;
	fIncrement = 1.0;

	fFilterHeight = -1;
	fFilterWidth = -1;
	fIncrement = 1.0f / (float)nHeight;

	for (i = 0; i < nKernelCount; ++i)
	{
		fDegree = (i * (180.0F / (nKernelCount)));
		fTheta = fDegree * DEG_2_RAD;
		fCosRad = (float)cos(fTheta);
		fSinRad = (float)sin(fTheta);

		for (p = 0, fJ = fFilterHeight; p<nHeight; fJ += fIncrement, ++p)
		{
			for (q = 0, fK = fFilterWidth; q<nWidth; fK += fIncrement, ++q)
			{
				fX = (fJ * fCosRad) + (fK * fSinRad);
				fY = (-(fJ * fSinRad) + (fK * fCosRad));

				nIndex = (i * layerData->nKernelRowCount * layerData->nKernelColumnCount) + (p * layerData->nKernelColumnCount) + q;
				layerData->fWeightArray[nIndex] = (float)(exp(-(pow(fX, 2.0F) + pow(fGamma * fY, 2.0F)) / (2.0F * pow(fSigma, 2.0F))) * sin((2.0F * M_PI / fLambda * fX)));
			}
		}

		fMax = -99999.9F;
		fMin = 99999.9F;


		for (p = 0; p<nHeight; ++fJ, ++p)
		{
			for (q = 0; q<nWidth; ++fK, ++q)
			{
				nIndex = (i * layerData->nKernelRowCount * layerData->nKernelColumnCount) + (p * layerData->nKernelColumnCount) + q;

				if (layerData->fWeightArray[nIndex] > fMax)
					fMax = layerData->fWeightArray[nIndex];
				if (layerData->fWeightArray[nIndex] < fMin)
					fMin = layerData->fWeightArray[nIndex];
			}
		}

		for (p = 0; p<nHeight; ++fJ, ++p)
		{
			for (q = 0; q<nWidth; ++fK, ++q)
			{
				nIndex = (i * layerData->nKernelRowCount * layerData->nKernelColumnCount) + (p * layerData->nKernelColumnCount) + q;

				layerData->fWeightArray[nIndex] = (layerData->fWeightArray[nIndex] - fMin) / (fMax - fMin) - 0.5F;
			}
		}
	}

	return;
}


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
int FindLegal_GaborFilter(structArchitecture **archData, int nPrimingCycles, int nMaxWeight, int nRowCount)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	int		nKernelCount;
	int		nKernelSize;
	int		nStride;
	int		nArchCount = 0;
	int		WindowSize;
	int		nWeightCount;
	int		i;

	for (i = 0; i<2; ++i)
	{
		if (i == 1)
			(*archData) = (structArchitecture*)calloc(nArchCount, sizeof(structArchitecture));

		nArchCount = 0;

		for (nKernelCount = 2; nKernelCount <= 15; ++nKernelCount)
		{
			for (nKernelSize = 7; nKernelSize < 20; ++nKernelSize)
			{
				for (nStride = 1; nStride <= nKernelSize; ++nStride)
				{
					WindowSize = CalculateWindowSize(nRowCount, nKernelSize, nStride);
					WindowSize *= WindowSize;
					nWeightCount = ((nKernelSize * nKernelSize + 1) * nKernelCount) + (2 * (WindowSize + 1));

					if (nWeightCount > nMaxWeight)
						continue;

					if (!i)
					{
						++nArchCount;
					}
					else
					{
						(*archData)[nArchCount].bKeep = 1;
						(*archData)[nArchCount].nKernelCount = nKernelCount;
						(*archData)[nArchCount].nRowKernelSize = nKernelSize;
						(*archData)[nArchCount].nStrideRow = nStride;
						(*archData)[nArchCount].nWeightCount = nWeightCount;
						(*archData)[nArchCount].nID = nArchCount;

						printf("%d\t%d\t%d\t%d\t%d        \r", (*archData)[nArchCount].nID, (*archData)[nArchCount].nKernelCount, (*archData)[nArchCount].nRowKernelSize, (*archData)[nArchCount].nStrideRow, (*archData)[nArchCount].nWeightCount);
						++nArchCount;
					}
				}
			}
		}
	}

	return(nArchCount);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void FindBest_GaborFilter(structCLN **clnReturn, structNetwork *networkMain, structArchitecture *archData, int nArchCount, int nLabelID, int nMaxWeight, int nPrimingCycles, structInput *inputData, structInput *inputTestingData, structInput *inputTrainingData, structInput *inputVerifyData)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structNetwork	*network;
	structCLN		*clnNew = NULL;
	structMAC		*macData = NULL;
	float			fMovingTotal;
	float			fMovingAverage=0.0f;
	int				nKeepCount = nArchCount;
	int				nMovingCount;
	int				i;
	int				nTrainVerifySplit = networkMain->nTrainVerifySplit;

	if ((network = (structNetwork *)calloc(1, sizeof(structNetwork))) == NULL)
		exit(0);

	network->nRowCount = networkMain->nRowCount;
	network->nColumnCount = networkMain->nColumnCount;
	network->nClassCount = networkMain->nClassCount;
	network->fInitialError = networkMain->fInitialError;
	network->classHead = networkMain->classHead;

	strcpy(network->sFilePath, networkMain->sFilePath);
	strcpy(network->sTrainingFilePath, networkMain->sTrainingFilePath);
	strcpy(network->sTestingFilePath, networkMain->sTestingFilePath);


	if ((network->fInputArray = (float *)calloc(network->nRowCount * network->nColumnCount, sizeof(float))) == NULL)
		exit(0);

	CreateMatrix(&network->nMatrix, network->nClassCount);

	while (nKeepCount > 1)
	{
		fMovingTotal = 0.0f;
		nMovingCount = 0;

		printf("---> %d\t%d\t%d                               \n", nLabelID, nKeepCount, nTrainVerifySplit);

		SplitData_InputData(inputData, &inputTrainingData, &inputVerifyData, nTrainVerifySplit, 100);
		SiftClasses_InputData(inputTrainingData, &networkMain->classHead, TRAINING);
		SiftClasses_InputData(inputVerifyData, &networkMain->classHead, VERIFY);

		for (i = 0; i<nArchCount; ++i)
		{
			if (!archData[i].bKeep)
				continue;

			//CreateCLN(&clnNew, archData[i].nKernelCount, archData[i].nRowKernelSize, archData[i].nStrideRow, networkMain->fLearningRate, networkMain->fInitialError, networkMain->fThreshold, ADJUST, networkMain->nRowCount, networkMain->nColumnCount, &networkMain->nPerceptronID, &networkMain->nSynapseID, networkMain->fInputArray);
			//archData[i].fAccuracy = PrimeCLN(nLabelID, &clnNew, nPrimingCycles, networkMain->nRowCount, networkMain->nColumnCount, &networkMain->nSynapseID, networkMain->fInputArray, inputTestingData, inputTrainingData, inputVerifyData, networkMain->nMatrix, networkMain->sDrive, networkMain->sTitle, HIDE_DATA, &archData[i].fPercentResponse);
			//fMovingTotal += archData[i].fAccuracy;
			//fMovingAverage = fMovingTotal / (float)++nMovingCount;

			if (archData[i].fAccuracy >= fMovingAverage)
				printf("%d\t%d\t%d\t%d\t%0.2f\t%0.2f\n", archData[i].nKernelCount, archData[i].nRowKernelSize, archData[i].nStrideRow, archData[i].nWeightCount, archData[i].fAccuracy*100.0f, archData[i].fPercentResponse*100.0f);

			clnNew = DeleteCLN_ClassLevelNetworks(&clnNew, 0);
		}

		network->clnHead = DeleteCLN_ClassLevelNetworks(&network->clnHead, 0);
		fMovingAverage /= nKeepCount;
		nKeepCount = 0;

		for (i = 0; i < nArchCount; ++i)
		{
			if (!archData[i].bKeep || archData[i].fAccuracy < fMovingAverage)
				archData[i].bKeep = 0;
			else
				++nKeepCount;
		}

		nTrainVerifySplit *= 2;
	}

	free(network->fInputArray);
	free(network);
}

