#include "main.h"

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void MarkInputData(structNetwork* networkMain, structInput* inputTrainingData, structInput* inputVerifyData, structInput* inputTestingData)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	FILE	*fpFileOut = NULL;
	char	sTimeBuffer[32];

	InferCLN_Inference(networkMain->clnHead, networkMain->classHead, inputTrainingData, HIDE_DATA, networkMain->nMatrix, networkMain->fInputArray, 1, networkMain->sDrive, networkMain->sTitle, DO_NOT_MARK, NO_THRESHOLD, fpFileOut, sTimeBuffer, MARK_CORRECT_CLASSIFICATION, -1);
	networkMain->clnHead->fTrainAccuracy = ScoreClassLevelNetworkMatrix(networkMain->clnHead->nNetworkType, 0, networkMain->classHead, networkMain->nMatrix, HIDE_MATRIX, fpFileOut);
	printf("Initial Train Accuracy: %0.4f\n", networkMain->clnHead->fTrainAccuracy);

	InferCLN_Inference(networkMain->clnHead, networkMain->classHead, inputVerifyData, HIDE_DATA, networkMain->nMatrix, networkMain->fInputArray, 1, networkMain->sDrive, networkMain->sTitle, DO_NOT_MARK, NO_THRESHOLD, fpFileOut, sTimeBuffer, MARK_CORRECT_CLASSIFICATION, -1);
	networkMain->clnHead->fValidateAccuracy = ScoreClassLevelNetworkMatrix(networkMain->clnHead->nNetworkType, 0, networkMain->classHead, networkMain->nMatrix, HIDE_MATRIX, fpFileOut);
	printf("Initial Validate Accuracy: %0.4f\n", networkMain->clnHead->fValidateAccuracy);

	InferCLN_Inference(networkMain->clnHead, networkMain->classHead, inputTestingData, HIDE_DATA, networkMain->nMatrix, networkMain->fInputArray, 1, networkMain->sDrive, networkMain->sTitle, DO_NOT_MARK, NO_THRESHOLD, fpFileOut, sTimeBuffer, MARK_CORRECT_CLASSIFICATION, -1);
	networkMain->clnHead->fTestAccuracy = ScoreClassLevelNetworkMatrix(networkMain->clnHead->nNetworkType, 0, networkMain->classHead, networkMain->nMatrix, HIDE_MATRIX, fpFileOut);
	printf("Initial Test Accuracy: %0.4f\n\n", networkMain->clnHead->fTestAccuracy);
}






/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
float InferCLN_Inference(structCLN *cln, structClass *classHead, structInput *input, int nDisplayMode, int **nMatrix, float *fInputArray, int bWriteOutput, char *sDrive, char *sTitle, int bMark, int bThreshold, FILE *fpFileOut, char *sTimeBuffer, int nMode, int nClusterMax)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
#ifdef _WINDOWS
	LARGE_INTEGER	lnFrequency;
	LARGE_INTEGER	lnStart;
#endif
#ifdef _VXWORKS
	uint64_t	lnFrequency;
	uint64_t	lnStart;
#endif
#ifdef _LINUX
	uint64_t		lnFrequency;
	struct timespec lnStart;
#endif

	structLayer			*layerCur = NULL;
	structPerceptron	*perceptronCur;
	FILE				*pFile = NULL;
	FILE				*pOutputFile = NULL;
	float				fThreshold;
	float				fRatio;
	float				fAccuracy;
	float				fRatioTotal;
	int					nTargetClass;
	int					nInputIndex;
	int					nPredictedTargetClass;


	if (bThreshold == THRESHOLD)
		fThreshold = cln->fThreshold;
	else
		fThreshold = 0.0f;

	fRatioTotal = 0.0f;
	ClearClassLevelNetworkMatrix(classHead, nMatrix);

	StartTimer(&lnFrequency, &lnStart);


	if (nMode == MARK_CORRECT_CLASSIFICATION)
	{
		for (nInputIndex = 0; nInputIndex < input->nInputCount; ++nInputIndex)
		{
			input->data[nInputIndex].bCorrect=0;
			input->data[nInputIndex].nMissCount = 0;
		}
	}

	for (nInputIndex = 0; nInputIndex < input->nInputCount; ++nInputIndex)
	{
		if ((nMode == INFER_CORRECT_ONLY || nMode == BREAK_ON_BAD_CLASSIFICATION || nMode == SYNAPSE_STATISTICS_CORRECT_CLASSIFICATION) && !input->data[nInputIndex].bCorrectClassification)
			continue;

		if (nClusterMax != -1 && input->data[nInputIndex].nGroupA > nClusterMax)
			continue;
		
		if (input->data[nInputIndex].bTrained == 1)
			continue;

		if (cln->nNetworkType == COMPLETE_NETWORK)
			nTargetClass = input->data[nInputIndex].nLabelID;
		else if (cln->nNetworkType == CLASS_NETWORK)
			nTargetClass = (int)(input->data[nInputIndex].nLabelID != cln->nLabelID);
		else
		{
			HoldDisplay("cln->nNetworkType Error\n");
		}


		//for (i = 0; i < cln->nSize; ++i)
		//	fInputArray[i] = input->data[nInputIndex].fIntensity[i];

		memcpy(fInputArray, input->data[nInputIndex].fIntensity, (cln->nSize * sizeof(float)));


		if(!nInputIndex)
			ForwardPropagate_Train(cln->macData, cln->nMACCount, pOutputFile);
		else
			ForwardPropagate_Train(cln->macData, cln->nMACCount, NULL);


		if (bMark == MARK)
		{
			nPredictedTargetClass = CalculateThreshold_Perceptron(cln->perceptronClassifier, fThreshold, &fRatio, nDisplayMode, NULL);

			if (nPredictedTargetClass != -1)
			{
				++nMatrix[nTargetClass][nPredictedTargetClass];

				if (nPredictedTargetClass == nTargetClass)
					input->data[nInputIndex].bTrained = 1;
			}
		}
		else
		{
			nPredictedTargetClass = CalculateThreshold_Perceptron(cln->perceptronClassifier, fThreshold, &fRatio, nDisplayMode, NULL);

			if (nPredictedTargetClass != -1)
			{
				if (nDisplayMode == SHOW_DATA)
				{
					if (pOutputFile != NULL && nPredictedTargetClass != nTargetClass)
					{
						fprintf(pOutputFile, "%d,%d,%d,%f,%f,%f", input->data[nInputIndex].nID, input->data[nInputIndex].nLabelID, nPredictedTargetClass, input->data[nInputIndex].fAspect, input->data[nInputIndex].fRadialSpeed, input->data[nInputIndex].fSpread);
						for (perceptronCur = cln->perceptronClassifier; perceptronCur != NULL; perceptronCur = perceptronCur->next)
							fprintf(pOutputFile, ",%f", perceptronCur->fOutput);

						fprintf(pOutputFile, "\n");
					}

					if (pFile != NULL)
					{
						fprintf(pFile, "%d,%d,%d,%f,%f,%f", input->data[nInputIndex].nID, input->data[nInputIndex].nLabelID, nPredictedTargetClass, input->data[nInputIndex].fAspect, input->data[nInputIndex].fRadialSpeed, input->data[nInputIndex].fSpread);
						for (perceptronCur = cln->perceptronClassifier; perceptronCur != NULL; perceptronCur = perceptronCur->next)
							fprintf(pFile, ",%f", perceptronCur->fOutput);

						fprintf(pFile, "\n");
					}
					//else
					//	printf("%s,%s,%d,", input->data[nInputIndex].sDescription, input->data[nInputIndex].sLabel, input->data[nInputIndex].nLabelID);
				}

				++nMatrix[nTargetClass][nPredictedTargetClass];
			}
		}
		fRatioTotal += fRatio;


		if (nMode != BREAK_ON_BAD_CLASSIFICATION)
			input->data[nInputIndex].fDifference = fRatio;

		if (nMode == MARK_CORRECT_CLASSIFICATION && nPredictedTargetClass == nTargetClass)
		{
			input->data[nInputIndex].bCorrectClassification = 1;
		}
		else if (nMode == BREAK_ON_BAD_CLASSIFICATION && nPredictedTargetClass != nTargetClass)
		{
			++input->data[nInputIndex].nMissCount;
			fAccuracy = 0.0f;
			break;
		}


		if (/*nDisplayMode == SHOW_DATA && */nInputIndex > 0 && !(nInputIndex % 10))
		{
			fAccuracy = ScoreClassLevelNetworkMatrix(cln->nNetworkType, cln->nLabelID, classHead, nMatrix, HIDE_MATRIX, fpFileOut);
			printf("Infer: %d\t%0.2f\t                                    \r", nInputIndex, fAccuracy*100.0f);
		}
	}


	FormatTime(EndTimer(&lnFrequency, &lnStart), sTimeBuffer);
	cln->fRatioAverage = fRatioTotal / (float)input->nInputCount;

	fAccuracy = ScoreClassLevelNetworkMatrix(cln->nNetworkType, cln->nLabelID, classHead, nMatrix, nDisplayMode, fpFileOut);

	if (nDisplayMode == SHOW_DATA && nDisplayMode == SHOW_DATA)
	{
		printf("Infer: %d\t%0.2f\t                                    \r", nInputIndex, fAccuracy*100.0f);

		if (pFile != NULL)
		{
			fprintf(pFile, "%0.2f                                     \n", fAccuracy*100.0f);
			fclose(pFile);
		}
		else
			printf("%0.2f                                     \n", fAccuracy*100.0f);
	}

	if (pOutputFile != NULL)
		fclose(pOutputFile);


	return(fAccuracy);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
float Infer_Inference(structNetwork *networkMain, structInput *input, int nDisplayMode, structMAC **macData, int nMACCount, int bWriteOutput, int nMode)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	FILE				*pFile = NULL;
	FILE				*pOutputFile = NULL;
	float				fThreshold;
	float				fRatio;
	float				fAccuracy;
	int					nTargetClass;
	int					nInputIndex;
	int					nPredictedTargetClass;
	int					nSize = networkMain->nRowCount * networkMain->nColumnCount;
	int					i;
	char				sFilePath[256];

	if (nDisplayMode == SHOW_DATA)
	{
		sprintf(sFilePath, "%s\\outputs\\%s_inference.txt", networkMain->sDrive, networkMain->sTitle);
		if ((pFile = FOpenMakeDirectory(sFilePath, "wt")) == NULL)
		{
			printf("Error Infer_Inference(): Could not save network file: %s\n\n", sFilePath);
			while (1);
		}
	}

	if (bWriteOutput == 1)
	{
		sprintf(sFilePath, "%s\\outputs\\%s_inference_wrong.txt", networkMain->sDrive, networkMain->sTitle);
		if ((pOutputFile = FOpenMakeDirectory(sFilePath, "wt")) == NULL)
		{
			printf("Error Infer_Inference(): Could not save network file: %s\n\n", sFilePath);
			while (1);
		}
	}

	fThreshold = 0.0f;
	ClearMatrix(networkMain->nMatrix, input->nClassCount);

	for (nInputIndex = 0; nInputIndex < input->nInputCount; ++nInputIndex)
	{
		if (nMode == ALL_CLN)
			nTargetClass = input->data[nInputIndex].nLabelID;
		else
			nTargetClass = (int)(input->data[nInputIndex].nLabelID != networkMain->clnCur->nLabelID);

		for (i = 0; i < nSize; ++i)
			networkMain->fInputArray[i] = input->data[nInputIndex].fIntensity[i];

		ForwardPropagate_Train(*macData, nMACCount, NULL);

		if (nDisplayMode == SHOW_DATA)
		{
			if (pFile != NULL)
				fprintf(pFile, "%d,%s,%s,%d,", input->data[nInputIndex].nID, input->data[nInputIndex].sDescription, input->data[nInputIndex].sLabel, input->data[nInputIndex].nLabelID);
			else
				printf("%s,%s,%d,", input->data[nInputIndex].sDescription, input->data[nInputIndex].sLabel, input->data[nInputIndex].nLabelID);
		}

		nPredictedTargetClass = CalculateThreshold_Perceptron(networkMain->clnCur->perceptronClassifier, fThreshold, &fRatio, nDisplayMode, pFile);

		if (bWriteOutput == 1 && nPredictedTargetClass != nTargetClass)
		{
			fprintf(pOutputFile, "%d,%f,%f,%f\n", input->data[nInputIndex].nLabelID, input->data[nInputIndex].fAspect, input->data[nInputIndex].fRadialSpeed, input->data[nInputIndex].fSpread);
		}

		++networkMain->nMatrix[nTargetClass][nPredictedTargetClass];

		if (nInputIndex > 0 && !(nInputIndex % 1000))
		{
			fAccuracy = ScoreMatrixV2(networkMain, HIDE_MATRIX, nMode);
			printf("Infer: %d\t%0.2f\t                                    \r", nInputIndex, fAccuracy*100.0f);
		}
	}

	fAccuracy = ScoreMatrixV2(networkMain, HIDE_MATRIX, nMode);
	printf("Infer: %d\t%0.2f\t                                    \r", nInputIndex, fAccuracy*100.0f);

	if (nDisplayMode == SHOW_DATA)
	{
		if (pFile != NULL)
		{
			fprintf(pFile, "%0.2f                                     \n", fAccuracy*100.0f);
			fclose(pFile);
		}
		else
			printf("%0.2f                                     \n", fAccuracy*100.0f);
	}

	if (bWriteOutput == 1)
		fclose(pOutputFile);


	return(fAccuracy);
}


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void Analyze_Inference(structCLN *cln, structInput *input, float *fInputArray)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	FILE				*pFile = NULL;
	FILE				*pOutputFile = NULL;
	int					nInputIndex;

		float	fSum;
		float	fClusterSum;
		int		nCount = 0;
		int		i, j, k;



	for (nInputIndex = 0; nInputIndex < input->nInputCount; ++nInputIndex)
	{
		for (i = 0; i < cln->nSize; ++i)
			fInputArray[i] = input->data[nInputIndex].fIntensity[i];

		for (i = 0; i < cln->nMACCount; ++i)
		{
			if (1 || cln->macData[i].nLayerType == CONV_2D_LAYER)
			{
				fSum = *cln->macData[i].fWeight[0];

				for (j = 1; j < cln->macData[i].nCount; ++j)
				{
					if (cln->macData[i].nInputCount[j] > 0)
					{
						fClusterSum = *(cln->macData[i].fInput[j]);

						for (k = 0; k < cln->macData[i].nInputCount[j]; ++k)
						{
							fClusterSum += *(cln->macData[i].fInputArray[j][k]);
							cln->macData[i].fInputSum[j][k] += *(cln->macData[i].fInputArray[j][k]);
						}

						fSum += *(cln->macData[i].fWeight[j]) * fClusterSum;
					}
					else
					{
						fSum += *(cln->macData[i].fWeight[j]) * *(cln->macData[i].fInput[j]);
					}
				}

				*cln->macData[i].fOutput = MTanH(fSum);
			}
		}
	}

	for (i = 0; i < cln->nMACCount; ++i)
	{
		if (1 || cln->macData[i].nLayerType == CONV_2D_LAYER)
		{
			for (j = 1; j < cln->macData[i].nCount; ++j)
			{
				if (cln->macData[i].nInputCount[j] > 0)
				{
					for (k = 0; k < cln->macData[i].nInputCount[j]; ++k)
					{
						cln->macData[i].fInputAverage[j][k] = cln->macData[i].fInputSum[j][k] / (float)input->nInputCount;
						cln->macData[i].fInputSum[j][k] = 0.0f;
					}
				}
			}
		}
	}

	for (nInputIndex = 0; nInputIndex < input->nInputCount; ++nInputIndex)
	{
		for (i = 0; i < cln->nSize; ++i)
			fInputArray[i] = input->data[nInputIndex].fIntensity[i];

		for (i = 0; i < cln->nMACCount; ++i)
		{
			if (1 || cln->macData[i].nLayerType == CONV_2D_LAYER)
			{
				fSum = *cln->macData[i].fWeight[0];

				for (j = 1; j < cln->macData[i].nCount; ++j)
				{
					if (cln->macData[i].nInputCount[j] > 0)
					{
						fClusterSum = *(cln->macData[i].fInput[j]);

						for (k = 0; k < cln->macData[i].nInputCount[j]; ++k)
						{
							fClusterSum += *(cln->macData[i].fInputArray[j][k]);
							cln->macData[i].fInputSum[j][k] += ((*(cln->macData[i].fInputArray[j][k]) - cln->macData[i].fInputAverage[j][k]) * (*(cln->macData[i].fInputArray[j][k]) - cln->macData[i].fInputAverage[j][k]));
						}

						fSum += *(cln->macData[i].fWeight[j]) * fClusterSum;
					}
					else
					{
						fSum += *(cln->macData[i].fWeight[j]) * *(cln->macData[i].fInput[j]);
					}
				}

				*cln->macData[i].fOutput = MTanH(fSum);
			}
		}
	}

	for (i = 0; i < cln->nMACCount; ++i)
	{
		if (1 || cln->macData[i].nLayerType == CONV_2D_LAYER)
		{
			for (j = 1; j < cln->macData[i].nCount; ++j)
			{
				if (cln->macData[i].nInputCount[j] > 0)
				{
					for (k = 0; k < cln->macData[i].nInputCount[j]; ++k)
					{
						cln->macData[i].fInputAverage[j][k] = sqrtf(cln->macData[i].fInputSum[j][k] / (float)(input->nInputCount - 1));
					}
				}
			}
		}
	}


	return;
}


//fClusterSum = *(cln->macData[i].fInput[j]);
//fTotal = 0.0f;
//
//for (k = 0; k < cln->macData[i].nInputCount[j]; ++k)
//{
//	fClusterSum += *(cln->macData[i].fInputArray[j][k]);
//	fTotal += *(cln->macData[i].fInputArray[j][k]);
//}
//fAverage = fTotal / (float)cln->macData[i].nInputCount[j];
//
//fTotal = 0.0f;
//
//for (k = 0; k < cln->macData[i].nInputCount[j]; ++k)
//{
//	fTotal += ((*(cln->macData[i].fInputArray[j][k]) - fAverage) * (*(cln->macData[i].fInputArray[j][k]) - fAverage));
//}
//
//cln->macData[i].fAverage[j] = sqrtf(fTotal / (float)(cln->macData[i].nInputCount[j] - 1));
//
//fSum += *(cln->macData[i].fWeight[j]) * fClusterSum;
