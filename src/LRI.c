#include "main.h"

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void LearningRateInitialization(structCLN *cln, structInput *inputData, float *fInputArray, float fTargetWeight, int nMode)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	float	*fArrayX;
	float	*fArrayY;
	float	fSum;
	float	fValue;
	float	fCalculated;
	float	fDelta;
	int		nMaxLoop;
	int		nInputIndex;
	int		nSlopeCount;
	int		nDeltaID;
	int		nLayerTypeArray[100];
	float	fWeightArray[100];
	int		nLayerType;
	int		nLayerCount;
	int		nCompleted;
	int		i, j, p, q;

	nMaxLoop = 100;
	nSlopeCount = 0;

	if ((fArrayX = (float *)calloc(nMaxLoop, sizeof(float))) == NULL)
	{
		HoldDisplay("LearnRateInitialization fArrayX Error\n");
	}
	if ((fArrayY = (float *)calloc(nMaxLoop, sizeof(float))) == NULL)
	{
		HoldDisplay("LearnRateInitialization fArrayY Error\n");
	}

	for (p = 0; p < 100; ++p)
	{
		nLayerTypeArray[p] = 0;
		fWeightArray[p] = 1.0f;
	}

	nLayerType = -1;
	nLayerCount = -1;
	for (p = 0; p < cln->nMACCount; ++p)
	{
		if (cln->macData[p].nLayerType != nLayerType)
		{
			nLayerType = cln->macData[p].nLayerType;
			nLayerTypeArray[++nLayerCount] = nLayerType;
		}

		cln->macData[p].nLayerCount = nLayerCount;
	}

	j = 0;
	fArrayY[0] = 1.0F;
	nLayerType = nLayerTypeArray[j];
	nCompleted = 0;
	
	while (!nCompleted)
	{
		nCompleted = 1;
		ClearAverages_MAC(cln->macData, cln->nMACCount);

		for (nInputIndex = 0; nInputIndex < inputData->nInputCount && nCompleted; ++nInputIndex)
		{
			for (i = 0; i < cln->nSize; ++i)
				fInputArray[i] = inputData->data[nInputIndex].fIntensity[i];

			for (p = 0; p < cln->nMACCount; ++p)
			{
				if (cln->macData[p].nLayerCount > j)
				{
					break;
				}
				
				*(cln->macData[p]).fWeight[0] = fArrayY[0];
				fSum = *(cln->macData[p]).fWeight[0];

				for (q = 1; q < cln->macData[p].nCount; ++q)
				{
					*(cln->macData[p]).fWeight[q] = fArrayY[0];
					fSum += (*(cln->macData[p]).fWeight[q]) * (*(cln->macData[p]).fInput[q]);

					cln->macData[p].fAverage[q] += (*(cln->macData[p]).fInput[q]);
					++cln->macData[p].nAverageCount[q];
				}

				if (nMode == FULL_RANGE)
					*(cln->macData[p]).fOutput = fSum;
				else
					*(cln->macData[p]).fOutput = MTanH(fSum);


				if (fabs(*(cln->macData[p]).fOutput) >= SIG_PARM)
				{
					fArrayY[0] *= 0.95f;
						
					printf("%d\t%d\t%f                             \r", nInputIndex, j, fArrayY[0]);

					nCompleted = 0;
					break;
				}
			}
		}
	}


	nSlopeCount = 0;
	fArrayY[1] = fArrayY[0]/1000.0f;

	while (nSlopeCount < nMaxLoop)
	{
		InitializeWeights_ClassLevelNetworks(cln, SAME_WEIGHTS, fArrayY[nSlopeCount]);
		ClearAverages_MAC(cln->macData, cln->nMACCount);

		for (nInputIndex = 0; nInputIndex < inputData->nInputCount; ++nInputIndex)
		{
			for (i = 0; i < cln->nSize; ++i)
				fInputArray[i] = inputData->data[nInputIndex].fIntensity[i];

			for (p = 0; p < cln->nMACCount; ++p)
			{
				fSum = *(cln->macData[p]).fWeight[0];

				for (q = 1; q < cln->macData[p].nCount; ++q)
				{
					fSum += (*(cln->macData[p]).fWeight[q]) * (*(cln->macData[p]).fInput[q]);

					cln->macData[p].fAverage[q] += (*(cln->macData[p]).fInput[q]);
					++cln->macData[p].nAverageCount[q];
				}

				if (nMode == FULL_RANGE)
					*(cln->macData[p]).fOutput = fSum;
				else
					*(cln->macData[p]).fOutput = MTanH(fSum);
			}
		}

		CalculateAverages_MAC(cln->macData, cln->nMACCount);

		for (nInputIndex = 0; nInputIndex < inputData->nInputCount; ++nInputIndex)
		{
			for (i = 0; i < cln->nSize; ++i)
				fInputArray[i] = inputData->data[nInputIndex].fIntensity[i];

			for (p = 0; p < cln->nMACCount; ++p)
			{
				fSum = *(cln->macData[p]).fWeight[0];

				for (q = 1; q < cln->macData[p].nCount; ++q)
				{
					fSum += (*(cln->macData[p]).fWeight[q]) * (*(cln->macData[p]).fInput[q]);

					fValue = (*(cln->macData[p]).fInput[q]) - (cln->macData[p].fAverage[q]);
					cln->macData[p].fSumSquares[q] += (fValue* fValue);
				}

				if (nMode == FULL_RANGE)
					*(cln->macData[p]).fOutput = fSum;
				else
					*(cln->macData[p]).fOutput = MTanH(fSum);
			}
		}

		fArrayX[nSlopeCount] = CalculateStandardDeviations_MAC(cln->macData, cln->nMACCount, HIDE_DATA);

		fDelta = fabsf(((fTargetWeight - fArrayX[nSlopeCount]) / fTargetWeight) * 100.0f);


		printf("%d\t%0.8f\t%0.8f\t%0.8f                                                         \r", nSlopeCount, fArrayY[nSlopeCount], fArrayX[nSlopeCount], fDelta);

		if (fArrayX[nSlopeCount] == fTargetWeight) //  || fDelta < 0.0001f
		{
			printf("\nbreak on: if (fArrayX[nSlopeCount] == fTargetWeight)\n");
			break;
		}

		if (nSlopeCount > 0)
		{
			fDelta = 100000.0f;
			nDeltaID = nSlopeCount - 1;

			for (i = 0; i < nSlopeCount; ++i)
			{
				if (fabs(fArrayX[i] - fTargetWeight) < fDelta)
				{
					nDeltaID = i;
					fDelta = fabsf(fArrayX[i] - fTargetWeight);
				}
			}

			if (fArrayX[nSlopeCount] == fArrayX[nDeltaID])
			{
				break;
			}

			fCalculated = CalculateWeight(fArrayY[nSlopeCount], fArrayY[nDeltaID], fArrayX[nSlopeCount], fArrayX[nDeltaID], fTargetWeight);
			fArrayY[++nSlopeCount] = fCalculated;
		}
		else
		{
			fCalculated = ((fArrayY[nSlopeCount] * fTargetWeight) / fArrayX[nSlopeCount]) / 3.0f;
			//fArrayY[++nSlopeCount] = fCalculated;
			++nSlopeCount;
		}
	}

	printf("\n");
	CalculateStandardDeviations_MAC(cln->macData, cln->nMACCount, SHOW_DATA);
	ClearAverages_MAC(cln->macData, cln->nMACCount);

	free(fArrayX);
	free(fArrayY);
}


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void LearningRateInitializationZ(structCLN *cln, structInput *inputData, float *fInputArray, float fTargetWeight, int nMode)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	float	*fArrayX;
	float	*fArrayY;
	float	fSum;
	float	fValue;
	float	fCalculated;
	float	fDelta;
	int		nMaxLoop;
	int		nInputIndex;
	int		nSlopeCount;
	int		nDeltaID;
	int		i, p, q;

	nMaxLoop = 100;

	if ((fArrayX = (float *)calloc(nMaxLoop, sizeof(float))) == NULL)
	{
		HoldDisplay("LearnRateInitialization fArrayX Error\n");
	}
	if ((fArrayY = (float *)calloc(nMaxLoop, sizeof(float))) == NULL)
	{
		HoldDisplay("LearnRateInitialization fArrayY Error\n");
	}

	fArrayY[0] = 1.0F;
	fArrayY[1] = 0.0f;

	nSlopeCount = 0;

	while (nSlopeCount < nMaxLoop)
	{
		InitializeWeights_ClassLevelNetworks(cln, SAME_WEIGHTS, fArrayY[nSlopeCount]);
		ClearAverages_MAC(cln->macData, cln->nMACCount);

		for (nInputIndex = 0; nInputIndex < inputData->nInputCount; ++nInputIndex)
		{
			for (i = 0; i < cln->nSize; ++i)
				fInputArray[i] = inputData->data[nInputIndex].fIntensity[i];

			for (p = 0; p < cln->nMACCount; ++p)
			{
				fSum = *(cln->macData[p]).fWeight[0];

				for (q = 1; q < cln->macData[p].nCount; ++q)
				{
					fSum += (*(cln->macData[p]).fWeight[q]) * (*(cln->macData[p]).fInput[q]);

					cln->macData[p].fAverage[q] += (*(cln->macData[p]).fInput[q]);
					++cln->macData[p].nAverageCount[q];
				}

				if (nMode == FULL_RANGE)
					*(cln->macData[p]).fOutput = fSum;
				else
					*(cln->macData[p]).fOutput = MTanH(fSum);
			}
		}

		CalculateAverages_MAC(cln->macData, cln->nMACCount);

		for (nInputIndex = 0; nInputIndex < inputData->nInputCount; ++nInputIndex)
		{
			for (i = 0; i < cln->nSize; ++i)
				fInputArray[i] = inputData->data[nInputIndex].fIntensity[i];

			for (p = 0; p < cln->nMACCount; ++p)
			{
				fSum = *(cln->macData[p]).fWeight[0];

				for (q = 1; q < cln->macData[p].nCount; ++q)
				{
					fSum += (*(cln->macData[p]).fWeight[q]) * (*(cln->macData[p]).fInput[q]);

					fValue = (*(cln->macData[p]).fInput[q]) - (cln->macData[p].fAverage[q]);
					cln->macData[p].fSumSquares[q] += (fValue* fValue);
				}

				if (nMode == FULL_RANGE)
					*(cln->macData[p]).fOutput = fSum;
				else
					*(cln->macData[p]).fOutput = MTanH(fSum);
			}
		}

		fArrayX[nSlopeCount] = CalculateStandardDeviations_MAC(cln->macData, cln->nMACCount, HIDE_DATA);

		fDelta = fabsf(((fTargetWeight - fArrayX[nSlopeCount]) / fTargetWeight) * 100.0f);


		printf("%d\t%0.8f\t%0.8f\t%0.8f                                                                                  \r", nSlopeCount, fArrayY[nSlopeCount], fArrayX[nSlopeCount], fDelta);

		if (fArrayX[nSlopeCount] == fTargetWeight) //  || fDelta < 0.0001f
		{
			printf("\nbreak on: if (fArrayX[nSlopeCount] == fTargetWeight)\n");
			break;
		}

		if (nSlopeCount > 0)
		{
			fDelta = 100000.0f;
			nDeltaID = nSlopeCount - 1;

			for (i = 0; i < nSlopeCount; ++i)
			{
				if (fabs(fArrayX[i] - fTargetWeight) < fDelta)
				{
					nDeltaID = i;
					fDelta = fabsf(fArrayX[i] - fTargetWeight);
				}
			}

			if (fArrayX[nSlopeCount] == fArrayX[nDeltaID])
			{
				break;
			}

			fCalculated = CalculateWeight(fArrayY[nSlopeCount], fArrayY[nDeltaID], fArrayX[nSlopeCount], fArrayX[nDeltaID], fTargetWeight);
			fArrayY[++nSlopeCount] = fCalculated;
		}
		else
		{
			fCalculated = ((fArrayY[nSlopeCount] * fTargetWeight) / fArrayX[nSlopeCount]) / 3.0f;
			fArrayY[++nSlopeCount] = fCalculated;
		}
	}

	printf("\n");
	CalculateStandardDeviations_MAC(cln->macData, cln->nMACCount, SHOW_DATA);
	ClearAverages_MAC(cln->macData, cln->nMACCount);

	free(fArrayX);
	free(fArrayY);
}


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void LearningRateInitializationX(structCLN *cln, structInput *inputData, float *fInputArray, float fTargetWeight, int nMode)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	//float	*fArrayX;
	//float	*fArrayY;
	//float	fSum;
	//float	fValue;
	//float	fCalculated;
	//float	fDelta;
	//float	fSameWeight;
	//int		nMaxLoop;
	//int		nInputIndex;
	//int		nSlopeCount;
	//int		nDeltaID;
	//int		i, p, q;

	//nMaxLoop = 100;

	//if ((fArrayX = (float *)calloc(nMaxLoop, sizeof(float))) == NULL)
	//{
	//	HoldDisplay("LearnRateInitialization fArrayX Error\n");
	//}
	//if ((fArrayY = (float *)calloc(nMaxLoop, sizeof(float))) == NULL)
	//{
	//	HoldDisplay("LearnRateInitialization fArrayY Error\n");
	//}

	//fTargetWeight = fTargetWeight;
	//fArrayY[0] = 1.0F;
	//fArrayY[1] = 0.0f;

	//nSlopeCount = 0;
	//while (nSlopeCount < 20)
	//{
	//	InitializeWeights_ClassLevelNetworks(cln, SAME_WEIGHTS, fArrayY[nSlopeCount]);
	//	ClearAverages_MAC(cln->macData, cln->nMACCount);

	//	for (nInputIndex = 0; nInputIndex < inputData->nInputCount; ++nInputIndex)
	//	{
	//		for (i = 0; i < cln->nSize; ++i)
	//			fInputArray[i] = inputData->data[nInputIndex].fIntensity[i];

	//		ForwardPropagate(cln->macData, cln->nMACCount, CALCULATE_AVERAGE);
	//	}

	//	CalculateMACAverages(cln->macData, cln->nMACCount);

	//	for (nInputIndex = 0; nInputIndex < inputData->nInputCount; ++nInputIndex)
	//	{
	//		for (i = 0; i < cln->nSize; ++i)
	//			fInputArray[i] = inputData->data[nInputIndex].fIntensity[i];

	//		ForwardPropagate(cln->macData, cln->nMACCount, CALCULATE_SD);
	//	}

	//	fArrayX[nSlopeCount] = CalculateStandardDeviations(cln->macData, cln->nMACCount, HIDE_DATA);

	//	printf("%d\t%0.8f\t%0.8f\t%0.8f                    \r", nSlopeCount, fArrayY[nSlopeCount], fArrayX[nSlopeCount], fTargetWeight);

	//	if (fArrayX[nSlopeCount] == fTargetWeight)
	//		break;

	//	if (nSlopeCount > 0)
	//	{
	//		if (fArrayX[nSlopeCount] == fArrayX[nSlopeCount - 1])
	//			break;

	//		fSameWeight = CalculateWeight(fArrayY[nSlopeCount], fArrayY[nSlopeCount - 1], fArrayX[nSlopeCount], fArrayX[nSlopeCount - 1], fTargetWeight);
	//		fArrayY[++nSlopeCount] = fSameWeight;
	//	}
	//	else
	//		++nSlopeCount;
	//}

	//printf("                                                                     \r");
	//CalculateStandardDeviations(cln->macData, cln->nMACCount, SHOW_DATA);

	//ClearMACAverages(cln->macData, cln->nMACCount);

}
