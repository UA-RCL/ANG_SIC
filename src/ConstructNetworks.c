#include "main.h"


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
float BuildCompleteNetwork_ConstructNetworks(structNetwork *networkMain, structInput *inputTrainingData, structInput *inputVerifyData, structInput *inputTestingData, structInput *inputData, FILE *fpFileOut)
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

	structCLN		*clnCur = NULL;
	structCLN		*clnNew = NULL;
	structLayer		*layerCur;
	char			sTimeBuffer[32];
	float			*fWeights = NULL;
	float			*fThresholds = NULL;
	float			fTrainAccuracy = 0.0f;
	float			fVerifyAccuracy = 0.0f;
	float			fTestAccuracy = 0.0f;
	float			fDelta = 0.0f;
	float			fTimeTotal = 0.0f;
	float			fMaxTestPercent = 0.0f;
	float			fTimeAverage = 0.0f;
	float			*fRandomWeightarray = NULL;
	float			fBestThreshold = 0.0f;
	float			fCurrentLearningRate = 0.0f;
	float			fSameWeight = 0.0f;
	float			fTargetWeight = 0.0f;
	int				nCurrentLayerType;
	int				i;


	StartTimer(&lnFrequency, &lnStart);

	fCurrentLearningRate = networkMain->fLearningRate;

	CreateCLN_ClassLevelNetworks(&clnNew, networkMain->nLayerCount, networkMain->architecture, COMPLETE_NETWORK, networkMain->nClassCount, &networkMain->nClassLevelNetworkCount, networkMain->fLearningRate, networkMain->fInitialError, networkMain->fThreshold, networkMain->nRowCount, networkMain->nColumnCount, &networkMain->nPerceptronID, &networkMain->nSynapseID, networkMain->fInputArray, RANDOMIZE);
	AddNew_ClassLevelNetworks(&networkMain->clnHead, clnNew);
	
	CalculateOutputSize(networkMain->clnHead);
	GetClassLevelNetworkWeightCount(networkMain->clnHead);

	DescribeClassLevelNetwork(clnNew, NULL);
	
	if(fpFileOut != NULL)
		DescribeClassLevelNetwork(clnNew, fpFileOut);

	for (clnCur = networkMain->clnHead; clnCur != NULL; clnCur = clnCur->next)
	{
		// Find Layer Boundries
		for (layerCur = clnCur->layerHead, clnCur->nLayerCount = 0; layerCur != NULL; layerCur = layerCur->next, ++clnCur->nLayerCount);

		if((clnCur->nStartArray=(int *)calloc(clnCur->nLayerCount, sizeof(int))) == NULL)
			exit(0);
		
		if ((clnCur->nEndArray = (int *)calloc(clnCur->nLayerCount, sizeof(int))) == NULL)
			exit(0);

		clnCur->nLayerCount = 0;
		clnCur->nStartArray[clnCur->nLayerCount] = 0;
		nCurrentLayerType = clnCur->macData[clnCur->nStartArray[clnCur->nLayerCount]].nLayerType;

		for (i = 0; i < clnCur->nMACCount; ++i)
		{
			if (clnCur->macData[i].nLayerType != nCurrentLayerType)
			{
				clnCur->nEndArray[clnCur->nLayerCount++] = i;
				clnCur->nStartArray[clnCur->nLayerCount] = i;

				nCurrentLayerType = clnCur->macData[i].nLayerType;
			}

		}
		clnCur->nEndArray[clnCur->nLayerCount] = i;
		
		clnCur->bAdjustGlobalLearningRate = networkMain->bAdjustGlobalLearningRate;
		clnCur->fLearningRateMinimum = networkMain->fLearningRateMinimum;
		clnCur->fLearningRateMaximum = networkMain->fLearningRateMaximum;
		clnCur->bAdjustPerceptronLearningRate = networkMain->bAdjustPerceptronLearningRate;
		clnCur->bAdjustThreshold = networkMain->bAdjustThreshold;
		clnCur->fThresholdPercent = networkMain->fThresholdPercent;

//// Initial Learning Rates /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		if (networkMain->bLearnRateInitialization)
		{
			LearningRateInitialization(clnCur, inputTrainingData, networkMain->fInputArray, networkMain->fTargetWeight, COMPRESS);
		}
		else
		{
			SetLearningRates_ClassLevelNetworks(clnCur, networkMain->fLearningRate, networkMain->fLearningRate);
		}

		InitializeWeights_ClassLevelNetworksMAC(clnCur->macData, clnCur->nMACCount);
		//InitializeWeights_ClassLevelNetworks(clnCur, NULL, RANDOMIZE);

		Network_Train(networkMain, clnCur, &inputTrainingData, &inputVerifyData, inputTestingData, inputData, networkMain->nTrainingCycles, fpFileOut, EXECUTE_TEST_INFERENCE);

		InferCLN_Inference(clnCur, networkMain->classHead, inputTestingData, SHOW_DATA, networkMain->nMatrix, networkMain->fInputArray, 1, networkMain->sDrive, networkMain->sTitle, DO_NOT_MARK, NO_THRESHOLD, fpFileOut, sTimeBuffer, 0, -1);
		fTestAccuracy = ScoreClassLevelNetworkMatrix(clnCur->nNetworkType, 0, networkMain->classHead, networkMain->nMatrix, SHOW_MATRIX, fpFileOut);

		printf("%0.2f                                                      \n\n", fTestAccuracy*100.0f);
		if(fpFileOut != NULL)
			fprintf(fpFileOut, "%0.2f                                                      \n\n", fTestAccuracy*100.0f);

		SaveNetwork(networkMain, networkMain->sDNAOutputPath);
	}

	fDelta = EndTimer(&lnFrequency, &lnStart);
	FormatTime(fDelta, sTimeBuffer);
	printf("\nTotal Time: %s\n", sTimeBuffer);

	return(fTestAccuracy);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void RebuildCompleteNetwork_ConstructNetworks(structNetwork *networkMain, FILE *fpFileOut, int nDisplayNetwork, structInput *inputTrainingData)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structCLN		*clnCur = NULL;
	structCLN		*clnNew = NULL;
	structLayer		*layerCur = NULL;
	float			*fWeights = NULL;
	float			fTestAccuracy = 0.0f;
	float			fDelta = 0.0f;
	float			fTimeTotal = 0.0f;
	float			fTimeAverage = 0.0f;

	int				*nAdditionArray;
	int				*nMultiplicationArray;
	int				nLayer;

	strcpy(networkMain->sInputStructurePath, "seed.dna");
  printf("sInputStructurePath:%s\n",networkMain->sInputStructurePath);
  if (networkMain->sInputStructurePath == NULL) {
      printf("sInputStructurePath is NULL.\n");
  } else if ((strstr(networkMain->sInputStructurePath, ".dna")) != NULL) {
      printf("'.dna' found in the path.\n");
  } else {
      printf("'.dna' not found in the path.\n");
  }
	if (nDisplayNetwork == SHOW_DATA)
		printf("\nBuilding Network...   ");

	if ((strstr(networkMain->sInputStructurePath, ".dna")) != NULL){
		Read_Network(networkMain, networkMain->sInputStructurePath);
    printf("entered the .dna check\n");
  }  
	else if ((strstr(networkMain->sInputStructurePath, ".dnx")) != NULL)
		Read_DNX_Network(networkMain, networkMain->sInputStructurePath);
	else{
    printf("File Type Error: %s\n", networkMain->sInputStructurePath);

		DisplayMessage("Memory calloc Error: RebuildCompleteNetwork_ConstructNetworks() File Type Error", PAUSE);
  }

	if (nDisplayNetwork == SHOW_DATA)
		printf("Done                                                    \r");

	for (clnCur = networkMain->clnHead; clnCur != NULL; clnCur = clnCur->next)
	{
		clnCur->nNetworkType = COMPLETE_NETWORK;
		GetClassLevelNetworkWeightCount(clnCur);
		CreateMACArrayFromDNX_ClassLevelNetworks(clnCur);
		
		ForwardPropagateAnalyze_Train(clnCur->macData, clnCur->nMACCount, &nAdditionArray, &nMultiplicationArray, &clnCur->nLayerCount, 0);

		for (layerCur = clnCur->layerHead, nLayer=0; layerCur != NULL; layerCur = layerCur->next, ++nLayer)
		{
			layerCur->nAdditions = nAdditionArray[nLayer];
			layerCur->nMultiplications = nMultiplicationArray[nLayer];
		}


		if (nDisplayNetwork == SHOW_DATA)
		{
			DescribeClassLevelNetwork(clnCur, NULL);
			if (fpFileOut != NULL)
				DescribeClassLevelNetwork(clnCur, fpFileOut);
		}

	}

	return;
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
float BuildClassLevelNetwork_ConstructNetworks(structNetwork *networkMain, structInput *inputTrainingData, structInput *inputVerifyData, structInput *inputTestingData, structInput *inputData, FILE *fpFileOut)
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

	structCLN		*clnTempHead = NULL;
	structCLN		*clnCur = NULL;
	structCLN		*clnNew = NULL;
	structClass		*classCur;
	structClass		*classTemp;
	float			*fWeights = NULL;
	float			fAccuracy = 0.0f;
	float			fTotal = 0.0f;
	float			fAverage = 0.0f;
	float			fDelta = 0.0f;
	float			fTimeTotal = 0.0f;
	float			fMaxTestPercent = 0.0f;
	float			fMaxThreshold = 0.0f;
	float			fTimeAverage = 0.0f;

	float			fTrainAccuracy = 0.0f;
	float			fVerifyAccuracy = 0.0f;
	float			fTestAccuracy = 0.0f;
	float			fMedian = 0.0f;
	float			fMaxAccuracy = 0.0f;
	float			fPercent = 0.0f;

	float			*fRandomWeightarray = NULL;

	int				nTrainVerifySplit;
	int				nCLNID = 0;
	int				i, j, k;
	int				nDataCountArray[10] = { 1,  2,  5, 10, 20, 50, 100 };
	int				nEpochCountArray[10] = { 10, 10, 10, 20, 20, 20, 30 };
	int				nAverageCount;
	int				nWeightMax;
	int				nRandomizeMode = ORDERED;
	int				bFlag = 0;
	int				nCount = 0;
	int				nSubNet = 0;
	int				nKeepID = -1;
	int				*nClassMemberCountArray;
	int				nBuildThreshold = -1;
	int				nTrainThreshold = -1;
	int				nWindowSize = 0;
	int				nWindowSizeMax = 9999;
	char			sTimeBuffer[32];

	StartTimer(&lnFrequency, &lnStart);

	if (networkMain->nBuildThreshold == 1)
		nBuildThreshold = THRESHOLD;
	else
		nBuildThreshold = NO_THRESHOLD;

	if (networkMain->nTrainThreshold == 1)
		nTrainThreshold = THRESHOLD;
	else
		nTrainThreshold = NO_THRESHOLD;




	if ((fRandomWeightarray = (float *)calloc(32768, sizeof(float))) == NULL)
	{
		HoldDisplay("fRandomWeightarray Error\n");
	}
	else
	{
		for (i = 0; i < 32768; ++i)
			fRandomWeightarray[i] = MULTIPLIER * UNIFORM_PLUS_MINUS_ONE;
	}

	nTrainVerifySplit = networkMain->nTrainVerifySplit;

	networkMain->nLayerCount = 2;
	networkMain->architecture = (structArchitecture *)calloc(networkMain->nLayerCount, sizeof(structArchitecture));

	for (classCur = networkMain->classHead; classCur != NULL; classCur = classCur->next)
	{
		for (nSubNet = 0; nSubNet<networkMain->nSubNetCount; ++nSubNet)
		{
			SplitData_InputData(inputData, &inputTrainingData, &inputVerifyData, nTrainVerifySplit, nDataCountArray[0]);
			SiftClasses_InputData(inputTrainingData, &networkMain->classHead, TRAINING);
			SiftClasses_InputData(inputVerifyData, &networkMain->classHead, VERIFY);
			if (networkMain->nBuildSort == 1)
				Sort_InputData(inputTrainingData->data, inputTrainingData->nInputCount, inputTrainingData->nSize, RANDOMIZE);
			//DisplayInputData(networkMain, 1);

			////////////////////////////////////////////////////

			nWeightMax = 0;
			nAverageCount = 0;
			clnTempHead = NULL;

			networkMain->nKernelCountEnd = 20;

			for (i = networkMain->nKernelCountStart; i <= networkMain->nKernelCountEnd; ++i)
			{
				for (j = 2; j <= networkMain->nRowCount; ++j)
				{
					for (k = 2; k <= j; ++k)
					{
						networkMain->nPerceptronID = 0;
						networkMain->nSynapseID = 0;

						//networkMain->architecture[0].nID = nCLNID;
						//networkMain->architecture[0].nLayerType = FULLY_CONNECTED_LAYER;
						//networkMain->architecture[0].nKernelCount = 100;

						networkMain->architecture[0].nID = nCLNID;
						networkMain->architecture[0].nLayerType = SINGLE_CONV_LAYER;
						networkMain->architecture[0].nKernelCount = i;
						networkMain->architecture[0].nRowKernelSize = j;
						networkMain->architecture[0].nColumnKernelSize = j;
						networkMain->architecture[0].nStrideRow = k;
						networkMain->architecture[0].nStrideColumn = k;

						networkMain->architecture[1].nID = nCLNID;
						networkMain->architecture[1].nLayerType = CLASSIFIER_LAYER;
						networkMain->architecture[1].nKernelCount = 2;

						nWindowSize=CreateCLN_ClassLevelNetworks(&clnNew, networkMain->nLayerCount, networkMain->architecture, CLASS_NETWORK, classCur->nID, &networkMain->nClassLevelNetworkCount, networkMain->fLearningRate, networkMain->fInitialError, networkMain->fThreshold, networkMain->nRowCount, networkMain->nColumnCount, &networkMain->nPerceptronID, &networkMain->nSynapseID, networkMain->fInputArray, ORDERED);

						clnNew->fThreshold = 0.0f;
						//fTrainAccuracy = ClassLevelNetworkGroup_Train(networkMain, clnNew, inputTrainingData, inputVerifyData, inputTestingData, fRandomWeightarray, nEpochCountArray[0], ORDERED, nBuildThreshold, fpFileOut);


						if (clnNew->fAccuracy <= 0.5f)
						{
							DeleteCLN_V2_ClassLevelNetworks(&clnNew, clnNew->nID);
						}
						else
						{
							clnNew->bKeep = 1;
							AddNew_ClassLevelNetworks(&clnTempHead, clnNew);
							printf("%d\t%0.2f\t%0.2f\t%0.2f\t%0.4f\t\t%d\t%d\t%d\t%d                                                     \r", ++nAverageCount, fTrainAccuracy*100.0f, clnNew->fAccuracy*100.0f, fTestAccuracy*100.0f, clnNew->fThreshold, clnNew->nWeightCount, clnNew->layerHead->nKernelCount, clnNew->layerHead->nKernelRowCount, clnNew->layerHead->nStrideRow);
						}
					}
				}
			}

			fMedian = GetMedian_ClassLevelNetworks(clnTempHead);


			bFlag = 1;
			while (bFlag)
			{
				bFlag = 0;
				for (clnCur = clnTempHead; clnCur != NULL; clnCur = clnCur->next)
				{
					if (clnCur->fAccuracy < fMedian)
					{
						DeleteCLN_V2_ClassLevelNetworks(&clnTempHead, clnCur->nID);
						bFlag = 1;
					}
				}
			}


			if (clnTempHead == NULL)
			{
				printf("clnTempHead == NULL\n");
				continue;
			}

			printf("%d\t%d\t%d\t%d\t%f                                                                                     \n", 0, nDataCountArray[0], nEpochCountArray[0], nAverageCount, fMedian);


			////////////////////////////////////////////////////

			for (i = 1; i < 7; ++i)
			{
				SplitData_InputData(inputData, &inputTrainingData, &inputVerifyData, nTrainVerifySplit, nDataCountArray[i]);
				SiftClasses_InputData(inputTrainingData, &networkMain->classHead, TRAINING);
				SiftClasses_InputData(inputVerifyData, &networkMain->classHead, VERIFY);
				if (networkMain->nTrainSort == 1)
					Sort_InputData(inputTrainingData->data, inputTrainingData->nInputCount, inputTrainingData->nSize, RANDOMIZE);
				//DisplayInputData(networkMain, 1);

				nAverageCount = 0;
				fTotal = 0.0f;

				for (clnCur = clnTempHead; clnCur != NULL; clnCur = clnCur->next)
				{
					clnCur->fThreshold = 0.0f;
					fTrainAccuracy = ClassLevelNetworkGroup_Train(networkMain, clnCur, inputTrainingData, inputVerifyData, inputTestingData, fRandomWeightarray, nEpochCountArray[i], ORDERED, nTrainThreshold, fpFileOut);
					printf("%d\t%0.2f\t%0.2f\t%0.2f\t%0.4f\t\t%d\t%d\t%d\t%d                                                     \r", ++nAverageCount, fTrainAccuracy*100.0f, clnCur->fAccuracy*100.0f, fTestAccuracy*100.0f, clnCur->fThreshold, clnCur->nWeightCount, clnCur->layerHead->nKernelCount, clnCur->layerHead->nKernelRowCount, clnCur->layerHead->nStrideRow);
					fTotal += clnCur->fAccuracy;
				}

				fMedian = clnTempHead->fAccuracy;
				for (clnCur = clnTempHead; clnCur != NULL; clnCur = clnCur->next)
				{
					if (clnCur->fAccuracy != fMedian)
						break;
				}

				if (clnCur == NULL)
					nAverageCount = 0;
				else if (nAverageCount > 1)
				{
					if (nAverageCount <= 3)
					{
						fMedian = fTotal / (float)nAverageCount;
					}
					else
					{
						fMedian = GetMedian_ClassLevelNetworks(clnTempHead);
					}


					printf("%d\t%d\t%d\t%d\t%f                                                                                     \n", i, nDataCountArray[i], nEpochCountArray[i], nAverageCount, fMedian);

					bFlag = 1;
					while (bFlag)
					{
						bFlag = 0;
						for (clnCur = clnTempHead; clnCur != NULL; clnCur = clnCur->next)
						{
							if (clnCur->fAccuracy < fMedian)
							{
								DeleteCLN_V2_ClassLevelNetworks(&clnTempHead, clnCur->nID);
								bFlag = 1;
							}
						}
					}
				}
				else
					break;
			}

			// Select Best CLN
			nCount = 0;
			fMaxAccuracy = 0.0f;
			fMaxThreshold = 0.0f;
			nKeepID = -1;
			for (clnCur = clnTempHead; clnCur != NULL; clnCur = clnCur->next)
			{
				if (clnCur->fAccuracy >= fMaxAccuracy)
				{
					if (clnCur->fAccuracy == fMaxAccuracy)
					{
						if (clnCur->fThreshold > fMaxThreshold)
						{
							fMaxThreshold = clnCur->fThreshold;
							fMaxAccuracy = clnCur->fAccuracy;
							nKeepID = clnCur->nID;

							++nCount;
						}
					}
					else
					{
						fMaxThreshold = clnCur->fThreshold;
						fMaxAccuracy = clnCur->fAccuracy;
						nKeepID = clnCur->nID;

						++nCount;
					}
				}
			}

			// Remove failed CLNs
			bFlag = 1;
			while (bFlag)
			{
				bFlag = 0;
				for (clnCur = clnTempHead; clnCur != NULL; clnCur = clnCur->next)
				{
					if (clnCur->nID != nKeepID)
					{
						DeleteCLN_V2_ClassLevelNetworks(&clnTempHead, clnCur->nID);
						bFlag = 1;
					}
				}
			}

			AddNew_ClassLevelNetworks(&networkMain->clnHead, clnTempHead);



			// Reorganize and isolate selected CLNs
			nCount = 0;
			for (clnCur = clnTempHead; clnCur != NULL; clnCur = clnCur->next)
			{
				clnCur->nID = nCount++;
				clnCur->bKeep = 2;
			}

			printf("\nMark Correct --> %d\t%s\t%d\n", classCur->nID, classCur->sLabel, nCount);

			for (i = 0; i < inputData->nInputCount; ++i)
				inputData->data[i].bTrained = 0;

			for (clnCur = networkMain->clnHead; clnCur != NULL; clnCur = clnCur->next)
			{
				if (clnCur->nLabelID == classCur->nID)
				{
					DescribeClassLevelNetwork(clnCur, NULL);
					if (fpFileOut != NULL)
						DescribeClassLevelNetwork(clnCur, fpFileOut);

					InferCLN_Inference(clnCur, networkMain->classHead, inputData, SHOW_DATA, networkMain->nMatrix, networkMain->fInputArray, 1, networkMain->sDrive, networkMain->sTitle, MARK, THRESHOLD, fpFileOut, sTimeBuffer, 0, -1);
				}
			}


			if ((nClassMemberCountArray = (int *)calloc(inputData->nClassCount, sizeof(int))) == NULL)
				exit(0);

			for (i = 0; i < inputData->nInputCount; ++i)
			{
				if (inputData->data[i].bTrained == 1)
					++nClassMemberCountArray[inputData->data[i].nLabelID];
			}

			printf("\n");
			for (classTemp = networkMain->classHead; classTemp != NULL; classTemp = classTemp->next)
			{
				fPercent = (float)nClassMemberCountArray[classTemp->nID] / (float)(inputData->nClassMemberCount[classTemp->nID]);
				printf("%d\t%d\t%0.2f\n", classTemp->nID, nClassMemberCountArray[classTemp->nID], fPercent);
			}
			printf("\n\n");


			free(nClassMemberCountArray);

			SaveNetwork(networkMain, networkMain->sDNAOutputPath);
		}

		for (i = 0; i < inputData->nInputCount; ++i)
			inputData->data[i].bTrained = 0;
	}



	for (i = 0; i < inputData->nInputCount; ++i)
		inputData->data[i].bTrained = 0;

	if (networkMain->nPostTrain == 1)

	{
		for (clnCur = networkMain->clnHead; clnCur != NULL; clnCur = clnCur->next)
		{
			fTrainAccuracy = ClassLevelNetworkGroup_Train(networkMain, clnCur, inputTrainingData, inputVerifyData, inputTestingData, fRandomWeightarray, 20, ORDERED, THRESHOLD, fpFileOut);
			InferCLN_Inference(clnCur, networkMain->classHead, inputData, SHOW_DATA, networkMain->nMatrix, networkMain->fInputArray, 1, networkMain->sDrive, networkMain->sTitle, DO_NOT_MARK, THRESHOLD, fpFileOut, sTimeBuffer, 0, -1);
		}
	}


	CreateCombiningClassifier_Network(networkMain, fRandomWeightarray, nRandomizeMode);

	for (clnCur = networkMain->clnHead; clnCur != NULL; clnCur = clnCur->next)
	{
		DescribeClassLevelNetwork(clnCur, NULL);
		if (fpFileOut != NULL)
			DescribeClassLevelNetwork(clnCur, fpFileOut);
	}

	CombiningClassifier_Train(networkMain, inputTestingData, inputTrainingData, inputVerifyData, fpFileOut);


	DeleteAll_ClassLevelNetworks(&networkMain->clnHead);

	if (fRandomWeightarray)
		free(fRandomWeightarray);

	return(fMaxAccuracy);
}


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
float RebuildClassLevelNetwork_ConstructNetworks(structNetwork *networkMain, structInput *inputTrainingData, structInput *inputVerifyData, structInput *inputTestingData, structInput *inputData, FILE *fpFileOut)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structCLN		*clnCur = NULL;
	structCLN		*clnNew = NULL;
	float			*fWeights = NULL;
	float			fAccuracy = 0.0f;
	float			fDelta = 0.0f;
	float			fTimeTotal = 0.0f;
	float			fMaxTestPercent = 0.0f;
	float			fTimeAverage = 0.0f;
	float			fTrainAccuracy = 0.0f;
	float			*fRandomWeightarray = NULL;
	int				nRandomizeMode = ORDERED;
	int				i;
	char			sTimeBuffer[32];

	if ((fRandomWeightarray = (float *)calloc(32768, sizeof(float))) == NULL)
	{
		HoldDisplay("fRandomWeightarray Error\n");
	}
	else
	{
		for (i = 0; i < 32768; ++i)
			fRandomWeightarray[i] = MULTIPLIER * UNIFORM_PLUS_MINUS_ONE;
	}



	printf("\nBuilding Network...   ");

	Read_Network(networkMain, networkMain->sNetworkFilePath);

	printf("\nDone\n");

	for (i = 0; i < inputData->nInputCount; ++i)
		inputData->data[i].bTrained = 0;

	for (clnCur = networkMain->clnHead; clnCur != NULL; clnCur = clnCur->next)
	{
		clnCur->nNetworkType = CLASS_NETWORK;

		GetClassLevelNetworkWeightCount(clnCur);
		CreateMACArray_ClassLevelNetworks(clnCur);

		//The problem is here:
		printf("I will try\n");
		fTrainAccuracy = ClassLevelNetworkGroup_Train(networkMain, clnCur, inputTrainingData, inputVerifyData, inputTestingData, fRandomWeightarray, 20, ORDERED, THRESHOLD, fpFileOut);
		InferCLN_Inference(clnCur, networkMain->classHead, inputData, SHOW_DATA, networkMain->nMatrix, networkMain->fInputArray, 1, networkMain->sDrive, networkMain->sTitle, DO_NOT_MARK, THRESHOLD, fpFileOut, sTimeBuffer, 0, -1);
		printf("and fail\n");
	}

	CreateCombiningClassifier_Network(networkMain, fRandomWeightarray, nRandomizeMode);

	for (clnCur = networkMain->clnHead; clnCur != NULL; clnCur = clnCur->next)
	{
		DescribeClassLevelNetwork(clnCur, NULL);
		if (fpFileOut != NULL)
			DescribeClassLevelNetwork(clnCur, fpFileOut);
	}

	CombiningClassifier_Train(networkMain, inputTestingData, inputTrainingData, inputVerifyData, fpFileOut);

	return(fMaxTestPercent);
}




/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
structCLN	*CreateSeedNetwork_ConstructNetworks(structNetwork *networkMain)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structCLN		*clnNew = NULL;

	CreateCLN_ClassLevelNetworks(&clnNew, networkMain->nLayerCount, networkMain->architecture, COMPLETE_NETWORK, networkMain->nClassCount, &networkMain->nClassLevelNetworkCount, networkMain->fLearningRate, networkMain->fInitialError, networkMain->fThreshold, networkMain->nRowCount, networkMain->nColumnCount, &networkMain->nPerceptronID, &networkMain->nSynapseID, networkMain->fInputArray, RANDOMIZE);

	clnNew->bAdjustThreshold = networkMain->bAdjustThreshold;
	clnNew->nClassifierMode = HARDMAX;

	
	
	AddNew_ClassLevelNetworks(&networkMain->clnHead, clnNew);

	return(clnNew);
}


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
float PrimeSeedNetwork_ConstructNetworks(structNetwork *networkMain, structCLN *cln, structClass *classHead, structInput *inputData, structInput **inputTrainingData, structInput **inputVerifyData, structInput *inputTestingData, float *fInputArray, int nCycles, float fTargetWeight, int bRandomizeWeights, int nLearningRateMode, float fLearningRate, int bMatrixDisplay, FILE *fpFileOut, int bLearnRateInitializationOverride)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structLayer	*layerCur;
	float		fTestAccuracy=0.0f;
	float		*fWeights = NULL;
	float		*fThresholds = NULL;
	int			nCurrentLayerType;
	int			i;

	// Find Layer Boundries
	for (layerCur = cln->layerHead, cln->nLayerCount = 0; layerCur != NULL; layerCur = layerCur->next, ++cln->nLayerCount);

	if ((cln->nStartArray = (int *)calloc(cln->nLayerCount, sizeof(int))) == NULL)
		exit(0);

	if ((cln->nEndArray = (int *)calloc(cln->nLayerCount, sizeof(int))) == NULL)
		exit(0);

	cln->nLayerCount = 0;
	cln->nStartArray[cln->nLayerCount] = 0;
	nCurrentLayerType = cln->macData[cln->nStartArray[cln->nLayerCount]].nLayerType;

	for (i = 0; i < cln->nMACCount; ++i)
	{
		if (cln->macData[i].nLayerType != nCurrentLayerType)
		{
			cln->nEndArray[cln->nLayerCount++] = i;
			cln->nStartArray[cln->nLayerCount] = i;

			nCurrentLayerType = cln->macData[i].nLayerType;
		}

	}
	cln->nEndArray[cln->nLayerCount] = i;
	cln->bAdjustGlobalLearningRate = networkMain->bAdjustGlobalLearningRate;
	cln->fLearningRateMinimum = networkMain->fLearningRateMinimum;
	cln->fLearningRateMaximum = networkMain->fLearningRateMaximum;
	cln->bAdjustPerceptronLearningRate = networkMain->bAdjustPerceptronLearningRate;
	cln->bAdjustThreshold = networkMain->bAdjustThreshold;
	cln->fThresholdPercent = networkMain->fThresholdPercent;


	//// Initial Learning Rates /////////////////////////////////////////////////////////////////////////////////////////////////////////////////


	if (networkMain->bLearnRateInitialization && !bLearnRateInitializationOverride)
	{
		LearningRateInitialization(cln, *inputTrainingData, fInputArray, networkMain->fTargetWeight, COMPRESS);
	}
	else
	{
		SetLearningRates_ClassLevelNetworks(cln, fLearningRate, fLearningRate);
	}

	if (bRandomizeWeights == RANDOMIZE)
	{
		//InitializeWeights_ClassLevelNetworks(cln, RANDOMIZE, 0.0f);
		InitializeWeights_ClassLevelNetworksMAC(cln->macData, cln->nMACCount);
	}

	Network_Train(networkMain, cln, inputTrainingData, inputVerifyData, inputTestingData, inputData, nCycles, fpFileOut, EXECUTE_TEST_INFERENCE);

	free(cln->nStartArray);
	free(cln->nEndArray);

	return(fTestAccuracy);
}