#include <omp.h>
#include "main.h"


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void Network_Train(structNetwork *networkMain, structCLN *clnCur, structInput **inputTrainingData, structInput **inputVerifyData, structInput *inputTestingData, structInput *inputData, int nCycleCount, FILE *fpFileOut, int nExecuteTestInference)
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

	int				nEpochCount;
	float			fVerifyAccuracy = 0.0f;
	float			fTrainAccuracy = 0.0f;
	float			fDelta = 0.0f;
	float			fTimeTotal = 0.0f;
	float			fMaxTestPercent = 0.0f;
	float			fBestThreshold = 0.0f;
	float			fTimeAverage = 0.0f;
	float			fTestAccuracy = 0.0f;
	float			*fWeights = NULL;
	float			*fThresholds = NULL;
	float			fCurrentLearningRate = networkMain->fLearningRate;
	int				nNoProgressCount = 0;
	int				nNoProgressResplitCount = 0;
	int				nNoProgressResortCount = 0;
	char			**sDisplayArray;
	int				*nDisplayArray;
	int				i;
	char			sTimeBuffer[32];

	fMaxTestPercent = 0.0f;
	fBestThreshold = 0.0f;

	fWeights = (float *)calloc(clnCur->nWeightCount, sizeof(float));
	fThresholds = (float *)calloc(networkMain->nClassCount, sizeof(float));

	nDisplayArray = (int *)calloc(9, sizeof(int));
	sDisplayArray = (char **)calloc(9, sizeof(char *));
	for (i = 0; i<9; ++i)
		sDisplayArray[i] = (char *)calloc(256, sizeof(char));

	i = 0;
	strcpy(sDisplayArray[i++], "Cycle");
	strcpy(sDisplayArray[i++], "Train");
	strcpy(sDisplayArray[i++], "Verify");
	strcpy(sDisplayArray[i++], "Test");
	strcpy(sDisplayArray[i++], "Threshold");
	strcpy(sDisplayArray[i++], "Weights");
	strcpy(sDisplayArray[i++], "Time");
	strcpy(sDisplayArray[i++], "TimeAvg");
	strcpy(sDisplayArray[i++], "% Backprop");

	//CopyClassLevelNetworkWeights(clnCur, fWeights, fThresholds, NETWORK_TO_MEMORY);
	CopyWeights(clnCur, NETWORK_TO_MEMORY);
	
	for (nEpochCount = 1; nEpochCount <= nCycleCount; ++nEpochCount)
	{
		if ((networkMain->nTrainResplit > 0 && !(nEpochCount % networkMain->nTrainResplit)) || (networkMain->nNoProgressResplitCount > 0 && nNoProgressResplitCount == networkMain->nNoProgressResplitCount))
		{
			printf("Splitting Input Data...                                                               \n");
			Sort_InputData(inputData->data, inputData->nInputCount, inputData->nSize, RANDOMIZE);
			SplitData_InputData(inputData, inputTrainingData, inputVerifyData, networkMain->nTrainVerifySplit, 100);
			clnCur->fThresholdPercent = 0.25f;
			nNoProgressResplitCount = 0;
		}
		else if ((networkMain->nTrainSort > 0 && !(nEpochCount % networkMain->nTrainSort)) || (networkMain->nNoProgressResortCount > 0 && nNoProgressResortCount == networkMain->nNoProgressResortCount))
		{
			//printf("Sorting Input Data...                                                               \r");
			//Sort_InputData((*inputTrainingData)->data, (*inputTrainingData)->nInputCount, (*inputTrainingData)->nSize, RANDOMIZE);
			
			nNoProgressResortCount = 0;
		}

		StartTimer(&lnFrequency, &lnStart);

		fTrainAccuracy = ClassLevelNetwork_Train(clnCur, networkMain->classHead, *inputTrainingData, *inputVerifyData, HIDE_DATA, nEpochCount, networkMain->nMatrix, networkMain->fInputArray, fpFileOut);

		if ((*inputVerifyData)->nInputCount > 0)
			fVerifyAccuracy = InferCLN_Inference(clnCur, networkMain->classHead, *inputVerifyData, HIDE_DATA, networkMain->nMatrix, networkMain->fInputArray, 0, networkMain->sDrive, networkMain->sTitle, DO_NOT_MARK, NO_THRESHOLD, fpFileOut, sTimeBuffer, 0, -1);
		else
			fVerifyAccuracy = InferCLN_Inference(clnCur, networkMain->classHead, inputTestingData, HIDE_DATA, networkMain->nMatrix, networkMain->fInputArray, 0, networkMain->sDrive, networkMain->sTitle, DO_NOT_MARK, NO_THRESHOLD, fpFileOut, sTimeBuffer, 0, -1);

		fDelta = EndTimer(&lnFrequency, &lnStart);
		fTimeTotal += fDelta;

		if (fVerifyAccuracy >= fMaxTestPercent)
		{
			if (fMaxTestPercent <= 0.0f)
			{
				DisplayResults(fpFileOut, sDisplayArray, nDisplayArray, 9);
			}

			fMaxTestPercent = fVerifyAccuracy;
			fBestThreshold = clnCur->fThreshold;

			fTimeAverage = fTimeTotal / (float)nEpochCount;
			
			//CopyClassLevelNetworkWeights(clnCur, fWeights, fThresholds, NETWORK_TO_MEMORY);
			CopyWeights(networkMain->clnHead, NETWORK_TO_MEMORY);

			if (networkMain->nTestInferenceExecute)
			{
				if ((*inputVerifyData)->nInputCount > 0)
					fTestAccuracy = InferCLN_Inference(clnCur, networkMain->classHead, inputTestingData, HIDE_DATA, networkMain->nMatrix, networkMain->fInputArray, 0, networkMain->sDrive, networkMain->sTitle, DO_NOT_MARK, NO_THRESHOLD, fpFileOut, sTimeBuffer, 0, -1);
				else
					fTestAccuracy = fVerifyAccuracy;
			}
			else
				fTestAccuracy = 0.0f;

			i = 0;
			sprintf(sDisplayArray[i++], "%d", nEpochCount);
			sprintf(sDisplayArray[i++], "%0.2f", fTrainAccuracy*100.0f);
			sprintf(sDisplayArray[i++], "%0.2f", fVerifyAccuracy*100.0f);
			sprintf(sDisplayArray[i++], "%0.2f", fTestAccuracy*100.0f);
			sprintf(sDisplayArray[i++], "%0.4f", clnCur->fThreshold);
			sprintf(sDisplayArray[i++], "%d", clnCur->nWeightCount);
			sprintf(sDisplayArray[i++], "%0.2f", fDelta);
			sprintf(sDisplayArray[i++], "%0.2f", fTimeAverage);
			sprintf(sDisplayArray[i++], "%0.2f", clnCur->fPercentBackProp*100.0f);

			DisplayResults(fpFileOut, sDisplayArray, nDisplayArray, 9);

			fTestAccuracy = 0.0f;
			nNoProgressCount = 0;
			nNoProgressResplitCount = 0;
			nNoProgressResortCount = 0;
		}
		else
		{
			printf("Less Accurate                                                                            \r");
			++nNoProgressCount;
			++nNoProgressResplitCount;
			++nNoProgressResortCount;
		}

		if ((networkMain->nNoProgressCount > 0 && nNoProgressCount == networkMain->nNoProgressCount))
			break;

		if (clnCur->bAdjustGlobalLearningRate == 1)
		{
			fCurrentLearningRate *= 0.99f;
			SetLearningRates_ClassLevelNetworks(clnCur, fCurrentLearningRate, fCurrentLearningRate);
			CreateMACArray_ClassLevelNetworks(clnCur);
		}

		if (networkMain->nPruneNetwork == PRUNE_EACH_CYCLE)
		{
			printf("Pruning Network ...                                                                         \r");
			
			if(1)
				PruneWeights(&networkMain->clnHead->layerHead, networkMain->fPruneConvThreshold, networkMain->fPruneFCThreshold);
			else
				PruneWeights_V2(networkMain, &networkMain->clnHead->layerHead, *inputVerifyData, *inputTrainingData);
			
			GetClassLevelNetworkWeightCount(networkMain->clnHead);
			CreateMACArray_ClassLevelNetworks(networkMain->clnHead);
		}
	}

	if (fTestAccuracy > 0.0f)
	{
		i = 0;
		sprintf(sDisplayArray[i++], "%d", nEpochCount);
		sprintf(sDisplayArray[i++], "%0.2f", fTrainAccuracy*100.0f);
		sprintf(sDisplayArray[i++], "%0.2f", fVerifyAccuracy*100.0f);
		sprintf(sDisplayArray[i++], "%0.2f", fTestAccuracy*100.0f);
		sprintf(sDisplayArray[i++], "%0.4f", clnCur->fThreshold);
		sprintf(sDisplayArray[i++], "%d", clnCur->nWeightCount);
		sprintf(sDisplayArray[i++], "%0.2f", fDelta);
		sprintf(sDisplayArray[i++], "%0.2f", fTimeAverage);
		sprintf(sDisplayArray[i++], "%0.2f", clnCur->fPercentBackProp*100.0f);

		DisplayResults(fpFileOut, sDisplayArray, nDisplayArray, 9);
	}


	printf("\n\n");

	//CopyClassLevelNetworkWeights(clnCur, fWeights, fThresholds, MEMORY_TO_NETWORK);
	CopyWeights(networkMain->clnHead, MEMORY_TO_NETWORK);

	clnCur->fThreshold = fBestThreshold;
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
float ClassLevelNetwork_Train(structCLN *cln, structClass *classHead, structInput *inputTrain, structInput *inputVerify, int nDisplayMode, int nCycle, int **nMatrix, float *fInputArray, FILE *fpFileOut)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	float				fRatio;
	float				fAccuracy;
	float				fRatioTotal;
	float				fPrevThreshold;
	float				*fErrorArray;
	int					nTargetClass;
	int					nInputIndex;
	int					nPredictedTargetClass;
	int					i, q;
	int					nHit;
	int					nMiss;
	int					nRatioCount;
	int					*nIndexArray;
	float				fPercent;
	float				fThresholdPercent = cln->fThresholdPercent;

	nHit = 0;
	nMiss = 0;
	nRatioCount = 0;
	fRatioTotal = 0.0f;
	fPercent = 0.0f;
	fPrevThreshold = 0.0f;
	fAccuracy = 0.0f;
	
	fErrorArray = (float *)calloc(200000, sizeof(float));  // perceptronHead->nWeightCount
	ClearMatrix(nMatrix, inputTrain->nClassCount);
	
	if (cln->bAdjustThreshold == 1)
		cln->fThreshold = fPrevThreshold = SIG_PARM;

	nIndexArray = (int*)calloc(inputTrain->nInputCount, sizeof(int));
	for (q = 0; q < inputTrain->nInputCount; ++q)
		nIndexArray[q] = q;

	ShuffleArray(nIndexArray, inputTrain->nInputCount, 1);

	for (q = 0; q < inputTrain->nInputCount; ++q)
	{
		//nInputIndex = q; // (rand() * rand()) % (inputTrain->nInputCount);
		nInputIndex = nIndexArray[q]; 
		
		if (cln->nNetworkType == COMPLETE_NETWORK)
			nTargetClass = inputTrain->data[nInputIndex].nLabelID;
		else if (cln->nNetworkType == CLASS_NETWORK)
			nTargetClass = (int)(inputTrain->data[nInputIndex].nLabelID != cln->nLabelID);


		for (i = 0; i < cln->nSize; ++i)
		{
			fInputArray[i] = inputTrain->data[nInputIndex].fIntensity[i];
		}

		ForwardPropagate_Train(cln->macData, cln->nMACCount, NULL);
		nPredictedTargetClass = CalculateThreshold_Perceptron(cln->perceptronClassifier, cln->fThreshold, &fRatio, HIDE_DATA, NULL);
		fRatioTotal += fRatio;
		++nRatioCount;

		if (nPredictedTargetClass != -1)
		{
			if ((q > 0 && !(q % 100)))
			{
				fAccuracy = ScoreClassLevelNetworkMatrix(cln->nNetworkType, cln->nLabelID, classHead, nMatrix, HIDE_MATRIX, fpFileOut);
				printf("%d\t%d\t%0.4f\t%0.6f\t%0.2f\t%0.4f\r", nCycle, q, fAccuracy, cln->fThreshold, fPercent*100.0f, fThresholdPercent);
			}

			if (!cln->bAdjustThreshold)
			{
				BackPropagate_Train(cln, cln->layerClassifier, nTargetClass, cln->fInitialError, fErrorArray, cln->macData, cln->nMACCount, cln->bAdjustPerceptronLearningRate);
				++nMiss;
			}
			else
			{
				fPrevThreshold = cln->fThreshold;
				++nHit;
			}

			++nMatrix[nTargetClass][nPredictedTargetClass];
		}
		else
		{
			if (cln->bAdjustThreshold == 1)
			{
				BackPropagate_Train(cln, cln->layerClassifier, nTargetClass, cln->fInitialError, fErrorArray, cln->macData, cln->nMACCount, cln->bAdjustPerceptronLearningRate);
				++nMiss;
			}
		}
		
		fPercent = (float)nMiss / (float)(nMiss + nHit);


		if (cln->bAdjustThreshold == 1)
		{
			if (nPredictedTargetClass == -1)
				cln->fThreshold = fPrevThreshold;
			else
				cln->fThreshold = fRatioTotal / (float)(nRatioCount);

			if (fPercent > 0.0f && fPercent < fThresholdPercent)
				cln->fThreshold = (fThresholdPercent * cln->fThreshold) / fPercent;
		}


	}

	printf("%d\t%d\t%0.4f\t%0.6f\t%0.2f\t%0.4f\r", nCycle, q, fAccuracy, cln->fThreshold, fPercent*100.0f, fThresholdPercent);

	cln->fThresholdPercent = fPercent;
	cln->fRatioAverage = fRatioTotal / (float)inputTrain->nInputCount;
	fAccuracy = ScoreClassLevelNetworkMatrix(cln->nNetworkType, cln->nLabelID, classHead, nMatrix, HIDE_MATRIX, fpFileOut);

	if (nDisplayMode == SHOW_DATA)
	{
		printf("%d\t%d\t%0.2f\t\r", nCycle, nInputIndex, fAccuracy);
		printf("%0.2f                                     \n", fAccuracy);
	}

	free(fErrorArray);
	free(nIndexArray);

	if ((nMiss + nHit) > 0)
		cln->fPercentBackProp = (float)nMiss / (float)(nMiss + nHit);
	else
		cln->fPercentBackProp = 1.0f;

	return(fAccuracy);
}


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
float ClassLevelNetworkGroup_Train(structNetwork *networkMain, structCLN *cln, structInput *inputTrainingData, structInput *inputVerifyData, structInput *inputTestingData, float *fRandomWeightarray, int nEpochMax, int nRandomizeMode, int nThresholdMode, FILE *fpFileOut)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	float	fTrainAccuracy;
	float	fVerifyAccuracy;
	float	fMaxTestPercent = 0.0f;
	float	fTimeTotal = 0.0f;
	float	*fWeights = (float *)calloc(cln->nWeightCount, sizeof(float));
	float	*fThresholds = (float *)calloc(networkMain->nClassCount, sizeof(float));
	float	fThreshold = 0.0f;
	int		nEpochCount;
	char	sTimeBuffer[32];

	InitializeWeights_ClassLevelNetworks(cln, nRandomizeMode, 0.0f);
	CopyClassLevelNetworkWeights(cln, fWeights, fThresholds, NETWORK_TO_MEMORY);

	for (nEpochCount = 1; nEpochCount <= nEpochMax; ++nEpochCount)
	{
		//printf("%d\r", nEpochCount);

		fTrainAccuracy = ClassLevelNetwork_Train(cln, networkMain->classHead, inputTrainingData, inputVerifyData, HIDE_DATA, nEpochCount, networkMain->nMatrix, networkMain->fInputArray, fpFileOut);

		if (inputVerifyData->nInputCount > 0)
			fVerifyAccuracy = InferCLN_Inference(cln, networkMain->classHead, inputVerifyData, HIDE_DATA, networkMain->nMatrix, networkMain->fInputArray, 0, networkMain->sDrive, networkMain->sTitle, DO_NOT_MARK, nThresholdMode, fpFileOut, sTimeBuffer, 0, -1);
		else
			fVerifyAccuracy = InferCLN_Inference(cln, networkMain->classHead, inputTestingData, HIDE_DATA, networkMain->nMatrix, networkMain->fInputArray, 0, networkMain->sDrive, networkMain->sTitle, DO_NOT_MARK, nThresholdMode, fpFileOut, sTimeBuffer, 0, -1);

		if (fVerifyAccuracy >= fMaxTestPercent)
		{
			fMaxTestPercent = fVerifyAccuracy;
			fThreshold = cln->fThreshold;
			CopyClassLevelNetworkWeights(cln, fWeights, fThresholds, NETWORK_TO_MEMORY);
		}
	}

	CopyClassLevelNetworkWeights(cln, fWeights, fThresholds, MEMORY_TO_NETWORK);
	cln->fThreshold = fThreshold;
	cln->fAccuracy = InferCLN_Inference(cln, networkMain->classHead, inputVerifyData, HIDE_DATA, networkMain->nMatrix, networkMain->fInputArray, 1, networkMain->sDrive, networkMain->sTitle, DO_NOT_MARK, nThresholdMode, fpFileOut, sTimeBuffer, 0, -1);

	free(fThresholds);
	free(fWeights);

	//printf("\n");

	return(fTrainAccuracy);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
float CombiningClassifier_Train(structNetwork *networkMain, structInput *inputTestingData, structInput *inputTrain, structInput *inputVerify, FILE *fpFileOut)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structCLN			*clnCur = NULL;
	structCLN			*clnCombiningClassifier = NULL;
	structPerceptron	*perceptronCur = NULL;
	float				*fErrorArray;
	float				fPercent;
	float				fMaxPercent;
	float				fError;
	float				fXError;
	float				fRatioTotal;
	float				fTestAccuracy;
	int					nInputIndex;
	int					nTargetClass;
	int					nRight = 0;
	int					nWrong = 0;
	int					*nPredictedTargetClass;
	int					nSize = networkMain->nRowCount * networkMain->nColumnCount;
	int					nEpoch;
	int					nZeroCount;
	int					nSingleCount;
	int					nMultipleCount;
	int					nSinglePredictedTargetClass;
	int					nRatioCount;
	int					i, j;
	float				*fWeights=NULL;
	float				*fThresholds = NULL;
	float				fThreshold=0.0f;

	ClearMatrix(networkMain->nMatrix, inputTrain->nClassCount);

	fErrorArray = (float *)calloc(100000, sizeof(float));  // perceptronHead->nWeightCount
	nPredictedTargetClass = (int *)calloc(networkMain->nClassCount, sizeof(int));  // perceptronHead->nWeightCount

	fMaxPercent = 0.0f;

	for (clnCur = networkMain->clnHead; clnCur!=NULL; clnCur = clnCur->next)
	{
		if (clnCur->layerHead->nLayerType == COMBINING_CLASSIFIER_LAYER)
		{
			fWeights = (float *)calloc(clnCur->nWeightCount, sizeof(float));
			fThresholds = (float *)calloc(networkMain->nClassCount, sizeof(float));
			clnCombiningClassifier = clnCur;
			break;
		}
	}

	if (fWeights == NULL)
	{
		HoldDisplay("fWeights == NULL\n");
	}
	
	CopyClassLevelNetworkWeights(clnCombiningClassifier, fWeights, fThresholds, NETWORK_TO_MEMORY);




	for (nEpoch=1; nEpoch<=10; ++nEpoch)
	{
		Sort_InputData(inputTrain->data, inputTrain->nInputCount, inputTrain->nSize, RANDOMIZE);
		nRight = 0;
		nWrong = 0;
		fRatioTotal = 0.0f;
		nRatioCount = 0;

		for (nInputIndex = 0; nInputIndex < inputTrain->nInputCount; ++nInputIndex)
		{
			nTargetClass = inputTrain->data[nInputIndex].nLabelID;

			for (i = 0; i < nSize; ++i)
				networkMain->fInputArray[i] = inputTrain->data[nInputIndex].fIntensity[i];


			nSinglePredictedTargetClass = ForwardPropagateCombiningClassifier_Train(networkMain->clnHead, &nZeroCount, &nSingleCount, &nMultipleCount);
			
			if (nSinglePredictedTargetClass == nTargetClass)
				++nRight;
			else
				++nWrong;

			if (nSingleCount != 1 && nZeroCount != inputTrain->nClassCount)
			{
				for (perceptronCur = clnCombiningClassifier->perceptronClassifier; perceptronCur != NULL; perceptronCur = perceptronCur->next)
				{
					if (perceptronCur->nIndex == nTargetClass)
						perceptronCur->fDifferential = perceptronCur->fOutput - 1.00f;
					else
						perceptronCur->fDifferential = perceptronCur->fOutput + 1.00f;
				}


				for (i = 0; i < clnCombiningClassifier->nMACCount; ++i)
				{
					fError = D_MTanH(*clnCombiningClassifier->macData[i].fOutput) * *clnCombiningClassifier->macData[i].fDifferential;
					*clnCombiningClassifier->macData[i].fLearningRate -= (fError / 10000.0f * *clnCombiningClassifier->macData[i].fLearningRate);
					fXError = *clnCombiningClassifier->macData[i].fLearningRate * fError;

					for (j = 0; j < clnCombiningClassifier->macData[i].nCount; ++j)
						*clnCombiningClassifier->macData[i].fWeight[j] -= ((*clnCombiningClassifier->macData[i].fInput[j] * fXError));
				}
			}

			printf("%d\t%d   \r", nEpoch, nInputIndex);
		}


		nRight = 0;
		nWrong = 0;
		for (nInputIndex = 0; nInputIndex < inputVerify->nInputCount; ++nInputIndex)
		{
			nTargetClass = inputVerify->data[nInputIndex].nLabelID;

			for (i = 0; i < nSize; ++i)
				networkMain->fInputArray[i] = inputVerify->data[nInputIndex].fIntensity[i];

			nSinglePredictedTargetClass = ForwardPropagateCombiningClassifier_Train(networkMain->clnHead, &nZeroCount, &nSingleCount, &nMultipleCount);

			if (nSinglePredictedTargetClass == nTargetClass)
				++nRight;
			else
				++nWrong;
		}

		fPercent = (float)((float)nRight / ((float)(nRight + nWrong))) * 100.0f;

		if (fPercent >= fMaxPercent)
		{
			printf("%d\t%0.2f    \n", nEpoch, fPercent);
			CopyClassLevelNetworkWeights(clnCombiningClassifier, fWeights, fThresholds, NETWORK_TO_MEMORY);
			fMaxPercent = fPercent;
			fThreshold = clnCombiningClassifier->fThreshold;
		}
	}

	printf("\n");
	CopyClassLevelNetworkWeights(clnCombiningClassifier, fWeights, fThresholds, MEMORY_TO_NETWORK);

	int		nMultipleRight = 0;
	int		nMultipleWrong = 0;
	int		nSingleRight = 0;
	int		nSingleWrong = 0;
	int		nZeroRight = 0;
	int		nZeroWrong = 0;

	
	
	nRight = 0;
	nWrong = 0;
	ClearMatrix(networkMain->nMatrix, inputTestingData->nClassCount);
	for (nInputIndex = 0; nInputIndex < inputTestingData->nInputCount; ++nInputIndex)
	{
		nTargetClass = inputTestingData->data[nInputIndex].nLabelID;

		for (i = 0; i < nSize; ++i)
			networkMain->fInputArray[i] = inputTestingData->data[nInputIndex].fIntensity[i];

		nSinglePredictedTargetClass = ForwardPropagateCombiningClassifier_Train(networkMain->clnHead, &nZeroCount, &nSingleCount, &nMultipleCount);

		if (nSingleCount == 1)
		{
			if (nSinglePredictedTargetClass == nTargetClass)
				++nSingleRight;
			else
				++nSingleWrong;
		}
		else
		{
			if (nZeroCount == inputTestingData->nClassCount)
			{
				if (nSinglePredictedTargetClass == nTargetClass)
					++nZeroRight;
				else
					++nZeroWrong;
			}
			else
			{
				if (nSinglePredictedTargetClass == nTargetClass)
					++nMultipleRight;
				else
					++nMultipleWrong;
			}
		}
		
		
		if (nSinglePredictedTargetClass == nTargetClass)
			++nRight;
		else
			++nWrong;
	
		++networkMain->nMatrix[nSinglePredictedTargetClass][nTargetClass];
	
	}

	fPercent = (float)((float)nRight / ((float)(nRight + nWrong))) * 100.0f;
	printf("%0.2f    \n", fPercent);

	fTestAccuracy = ScoreClassLevelNetworkMatrix(COMPLETE_NETWORK, 0, networkMain->classHead, networkMain->nMatrix, SHOW_MATRIX, fpFileOut);
	printf("%0.2f    \n\n", fTestAccuracy*100.0f);

	fPercent = (float)((float)(nZeroRight + nZeroWrong) / ((float)(nRight + nWrong))) * 100.0f;
	fTestAccuracy = (float)((float)nZeroRight / ((float)(nZeroRight + nZeroWrong))) * 100.0f;
	printf("    Zero: %-2.2f\t%-2.2f\n", fPercent, fTestAccuracy);

	fPercent = (float)((float)(nSingleRight + nSingleWrong) / ((float)(nRight + nWrong))) * 100.0f;
	fTestAccuracy = (float)((float)nSingleRight / ((float)(nSingleRight + nSingleWrong))) * 100.0f;
	printf("  Single: %-2.2f\t%-2.2f\n", fPercent, fTestAccuracy);

	fPercent = (float)((float)(nMultipleRight + nMultipleWrong) / ((float)(nRight + nWrong))) * 100.0f;
	fTestAccuracy = (float)((float)nMultipleRight / ((float)(nMultipleRight + nMultipleWrong))) * 100.0f;
	printf("Multiple: %-2.2f\t%-2.2f\n", fPercent, fTestAccuracy);

	free(fErrorArray);

	return(0);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
int ForwardPropagateCombiningClassifier_Train(structCLN *clnHead, int *nZeroCount, int *nSingleCount, int *nMultipleCount)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structCLN	*clnCur = NULL;
	float		fRatio;
	int			nPredictedTargetClass;
	int			nWinner;

	*nZeroCount = 0;
	*nSingleCount = 0;
	*nMultipleCount = 0;
	
	for (clnCur = clnHead; clnCur->layerHead->nLayerType != COMBINING_CLASSIFIER_LAYER; clnCur = clnCur->next)
	{
		ForwardPropagate_Train(clnCur->macData, clnCur->nMACCount, NULL);
		nPredictedTargetClass = CalculateThreshold_Perceptron(clnCur->perceptronClassifier, clnCur->fThreshold, &fRatio, HIDE_DATA, NULL);

		if (nPredictedTargetClass == -1)
			++(*nZeroCount);
		else
		{
			if (nPredictedTargetClass == 0)
			{
				++(*nSingleCount);
				++(*nMultipleCount);

				nWinner = clnCur->nLabelID;
			}

		}
		
	}

	if ((*nSingleCount) != 1)
	{
		ForwardPropagate_Train(clnCur->macData, clnCur->nMACCount, NULL);
		nWinner = CalculateThreshold_Perceptron(clnCur->perceptronClassifier, clnCur->fThreshold, &fRatio, HIDE_DATA, NULL);
	}

	return(nWinner);
}

/*

	Avg	    Var	    SD
Avg	5649	4951	3751
Max	2698	2642	2642
Min	629	    1156	18643

*/



// na_38.exe d:\data\mnist\config\mnist_11.cfg
// d:\data\mnist\config\mnist_01.cfg




/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
int ForwardPropagate_Train(structMAC *macData, int nPerceptronCount, FILE *pFile)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	float	fSum=0.0f;
	float	fFeedback=0.0f;
	float	fClusterSum;
	int		nCount = 0;
	int		i, j, k;

	//FILE	*pFileX = NULL;
	//char	sFilePath[256];

	//sprintf(sFilePath, "D:/Data/MNIST/outputs/1.02.04.txt");
	//if ((pFileX = FOpenMakeDirectory(sFilePath, "wt")) == NULL)
	//{
	//	printf("Error Infer_Inference(): Could not save network file: %s\n\n", sFilePath);
	//	while (1);
	//}


	for (i = 0; i < nPerceptronCount; ++i)
	{
		if (macData[i].fDifferential != NULL)
			*(macData[i]).fDifferential = 0.0f;

		if (macData[i].nLayerType == MAX_POOLING_LAYER)
		{
			*(macData[i]).fOutput = *(macData[i]).fInput[0];
			macData[i].nKernelID = 0;

			for (j = 1; j < macData[i].nCount; ++j)
			{
				if (*(macData[i]).fInput[j] > *(macData[i]).fOutput)
				{
					*(macData[i]).fOutput = *(macData[i]).fInput[j];
					macData[i].nKernelID = j;
				}
			}
		}
		else
		{
			//if (*(macData[i]).fWeight[0] == 0.0f)
			//	printf("zero\n");
			
			fSum = *(macData[i]).fWeight[0];
			for (j = 1; j < macData[i].nCount; ++j)
			{
				//if (*(macData[i]).fWeight[j] == 0.0f)
				//	printf("zero\n");

				
				if (macData[i].nInputCount[j] > 0)
				{
					fClusterSum = (*(macData[i]).fInput[j]);

					for (k = 0; k < macData[i].nInputCount[j]; ++k)
						fClusterSum += *(macData[i].fInputArray[j][k]);

					fSum += (*(macData[i]).fWeight[j]) * fClusterSum;
				}
				else
				{
					fSum += (*(macData[i]).fWeight[j]) * (*(macData[i]).fInput[j]);
				}

				//fprintf(pFileX, "%d\t%0.7f\t%0.7f\n", nCount++, fSum, *(macData[i]).fWeight[j]);
			}

			 fFeedback = *(macData[i]).fOutput * macData[i].fFeedBackWeight;
			*(macData[i]).fOutput = MTanH(fSum - fFeedback);
		}
	}

	//fclose(pFileX);

	return(0);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
int ForwardPropagateAnalyze_Train(structMAC *macData, int nPerceptronCount, int **nAdditionArray, int **nMultiplicationArray, int *nLayerCount, int nSIR)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	float	fSum;
	float	fClusterSum;
	int		i, j, k;

	int		nAdditions = 0;
	int		nMultiplications = 0;

	int		nLayerID = macData[0].nLayerCount;

	*nLayerCount = 1;
	for (i = 0; i < nPerceptronCount; ++i)
	{
		if (nLayerID != macData[i].nLayerCount)
		{
			++(*nLayerCount);
			nLayerID = macData[i].nLayerCount;
		}
	}

	*nAdditionArray = (int *)calloc((*nLayerCount), sizeof(int));
	*nMultiplicationArray = (int *)calloc((*nLayerCount), sizeof(int));

	*nLayerCount = 0;
	nLayerID = macData[0].nLayerCount;
	
	for (i = 0; i < nPerceptronCount; ++i)
	{
		if (nLayerID != macData[i].nLayerCount)
		{
			(*nAdditionArray)[(*nLayerCount)] = nAdditions;
			(*nMultiplicationArray)[(*nLayerCount)] = nMultiplications;
			++(*nLayerCount);

			nAdditions = 0;
			nMultiplications = 0;

			nLayerID = macData[i].nLayerCount;
		}


		if (macData[i].nLayerType == MAX_POOLING_LAYER || macData[i].nLayerType == MAX_POOL_LAYER)
		{
			*(macData[i]).fOutput = *(macData[i]).fInput[0];
			macData[i].nKernelID = 0;

			for (j = 1; j < macData[i].nCount; ++j)
			{
				if (*(macData[i]).fInput[j] > *(macData[i]).fOutput)
				{
					*(macData[i]).fOutput = *(macData[i]).fInput[j];
					macData[i].nKernelID = j;
				}
			}
		}
		else
		{
			fSum = *(macData[i]).fWeight[0];

			for (j = 1; j < macData[i].nCount; ++j)
			{
				if (macData[i].nInputCount[j] > 0)
				{
					fClusterSum = (*(macData[i]).fInput[j]);

					for (k = 0; k < macData[i].nInputCount[j]; ++k)
					{
						fClusterSum += *(macData[i].fInputArray[j][k]);
						++nAdditions;
					}

					fSum += (*(macData[i]).fWeight[j]) * fClusterSum;
					++nAdditions;
					++nMultiplications;
				}
				else
				{
					fSum += (*(macData[i]).fWeight[j]) * (*(macData[i]).fInput[j]);
					++nAdditions;
					++nMultiplications;
				}
			}

			*(macData[i]).fOutput = MTanH(fSum);
		}
	}

	(*nAdditionArray)[*nLayerCount] = nAdditions;
	(*nMultiplicationArray)[*nLayerCount] = nMultiplications;
	++(*nLayerCount);

	return(0);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void GetForwardPropagateAverage_Train(structMAC *macData, int nPerceptronCount, int nMode)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	float	fSum;
	int		i, j;

	for (i = 0; i<nPerceptronCount; ++i)
	{
		fSum = *(macData[i]).fWeight[0];

		for (j = 1; j < macData[i].nCount; ++j)
		{
			fSum += (*(macData[i]).fWeight[j]) * (*(macData[i]).fInput[j]);

			macData[i].fAverage[j] += (*(macData[i]).fInput[j]);
			++macData[i].nAverageCount[j];
		}

		if(nMode == COMPRESS)
			*(macData[i]).fOutput = MTanH(fSum);
		else if(nMode == FULL_RANGE)
			*(macData[i]).fOutput = fSum;
		else
			HoldDisplay("GetForwardPropagateAverage_Train MODE error\n");
	}

	return;
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void GetForwardPropagateSD_Train(structMAC *macData, int nPerceptronCount, int nMode)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	float	fValue;
	float	fSum;
	int		i, j;

	for (i = 0; i<nPerceptronCount; ++i)
	{
		fSum = *(macData[i]).fWeight[0];

		for (j = 1; j < macData[i].nCount; ++j)
		{
			fSum += (*(macData[i]).fWeight[j]) * (*(macData[i]).fInput[j]);

			fValue = *(macData[i]).fInput[j] - macData[i].fAverage[j];
			macData[i].fSumSquares[j] += (fValue* fValue);
		}

		if (nMode == COMPRESS)
			*(macData[i]).fOutput = MTanH(fSum);
		else if (nMode == FULL_RANGE)
			*(macData[i]).fOutput = fSum;
		else
			HoldDisplay("GetForwardPropagateSD_Train MODE error\n");
	}

	return;
}


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void BackPropagate_Train(structCLN *cln, structLayer *layerClassifier, int nTargetClass, float fInitialError, float *fErrorArray, structMAC *macData, int nPerceptronCount, int bAdjustLearningRate)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	float				fXError;
	float				fError;
	int					nCurrentKernelID;
	int					nLayer;
	int					i, j;

	for (nLayer = cln->nLayerCount; nLayer > -1; --nLayer)
	{
		for (i = cln->nStartArray[nLayer]; i < cln->nEndArray[nLayer]; ++i)
		{
			if (macData[i].nLayerType == FULLY_CONNECTED_LAYER || macData[i].nLayerType == CLASSIFIER_LAYER)
			{
				if (macData[i].nLayerType == CLASSIFIER_LAYER)
				{
					if (macData[i].nID == nTargetClass)
						*(macData[i]).fDifferential = *(macData[i]).fOutput - fInitialError;
					else
						*(macData[i]).fDifferential = *(macData[i]).fOutput + fInitialError;
				}

				fError = D_MTanH(*(macData[i]).fOutput) * *(macData[i]).fDifferential;

				if (bAdjustLearningRate == 1)
				{
					*(macData[i]).fLearningRate -= ((fError / 100.0f) * *(macData[i]).fLearningRate);
				}

				fXError = *(macData[i]).fLearningRate * fError;
				macData[i].fFeedBackWeight -= fXError;
				
				for (j = 0; j < macData[i].nCount; ++j)
				{
					if (macData[i].fConnectToDifferential[j] != NULL)
						*(macData[i]).fConnectToDifferential[j] += (fError * *(macData[i]).fWeight[j]);

					*(macData[i]).fWeight[j] -= ((*(macData[i]).fInput[j] * fXError));
				}
			}
			else if (macData[i].nLayerType == SINGLE_CONV_LAYER || macData[i].nLayerType == MULTIPLE_CONV_LAYER)
			{
				nCurrentKernelID = -1;

				for (; i < cln->nEndArray[nLayer]; ++i)
				{
					fError = D_MTanH(*(macData[i]).fOutput) * *(macData[i]).fDifferential;
					

					if (bAdjustLearningRate == 1)
					{
						*(macData[i]).fLearningRate -= ((fError / 10000.0f) * *(macData[i]).fLearningRate);
					}

					if (macData[i].nKernelID != nCurrentKernelID) // New Kernel
					{
						macData[i].fFeedBackWeight -= (fError * *(macData[i]).fLearningRate);

						for (j = 0; j < macData[i].nCount; ++j)  // each synapse
						{
							if (nCurrentKernelID != -1)
								*(macData[i - 1]).fWeight[j] -= (*(macData[i - 1]).fLearningRate * fErrorArray[j]);

							fErrorArray[j] = ((*(macData[i]).fInput[j] * fError));

							if (macData[i].fConnectToDifferential[j] != NULL)
								*(macData[i]).fConnectToDifferential[j] += (fError * *(macData[i]).fWeight[j]);
						}

						nCurrentKernelID = macData[i].nKernelID;
					}
					else
					{
						for (j = 0; j < macData[i].nCount; ++j)  // each synapse
						{
							fErrorArray[j] += ((*(macData[i]).fInput[j] * fError));

							if (macData[i].fConnectToDifferential[j] != NULL)
								*(macData[i]).fConnectToDifferential[j] += (fError * *(macData[i]).fWeight[j]);
						}
					}

				}

				for (j = 0; j < macData[i - 1].nCount; ++j)  // each synapse
				{
					*(macData[i - 1]).fWeight[j] -= (*(macData[i - 1]).fLearningRate * fErrorArray[j]);
				}
			}
			else if (macData[i].nLayerType == MAX_POOLING_LAYER)
			{
				//fError = D_MTanH(*(macData[i]).fOutput) * *(macData[i]).fDifferential;
				
				for (j = 0; j < macData[i].nCount; ++j)
				{
					if (macData[i].fConnectToDifferential[j] != NULL && macData[i].nKernelID == j)
						*(macData[i]).fConnectToDifferential[j] = *(macData[i]).fDifferential;
					else
						*(macData[i]).fConnectToDifferential[j] = 0.0f;
				}
			}
		}
	}
}
