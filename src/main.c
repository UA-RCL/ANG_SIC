
#include "main.h"

int glblSegmentCount;

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
int main(int argc, char *argv[])
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structNetwork *networkMain = NULL;
	structCLN *clnCur = NULL;
	structCLN *clnNew = NULL;
	structInput *inputData = NULL;
	structInput *inputTestingData = NULL;
	structInput *inputTrainingData = NULL;
	structInput *inputVerifyData = NULL;
	float fAccuracy = 0.0f;
	float fMaxTestPercent = 0.0f;
	float fRatio = 0.0f;
	float *fWeights = NULL;
	float fDelta = 0.0f;
	float fTimeTotal = 0.0f;
	float fTimeAverage = 0.0f;
	int **nMaxMatrix;
	int nMACCount = 0;

	int nPerceptronCount = 0;
	int nCount = 0;
	int nWeightCount = 0;
	int i = 0;

	float fTrainAccuracy = 1.0f;
	float fVerifyAccuracy = 1.0f;
	float fTestAccuracy = 1.0f;
	structMAC *macData = NULL;

	FILE *fpFileOut = NULL;
	char sTemp[256];
	int j;
	char sTimeBuffer[32];

	if ((networkMain = (structNetwork *)calloc(1, sizeof(structNetwork))) == NULL)
		exit(0);

	Initialize_Network(networkMain);
	strcpy(networkMain->sTitle, "ANG v1.02.23");
	networkMain->nClassifierMode = HARDMAX;
	networkMain->nNumberFormat = FLOAT_POINT;

	if (argc < 2)
	{
		DisplayHelpFile();
		while (1)
			;
	}

	strcpy(networkMain->sConfigFilePath, "NULL");
	strcpy(networkMain->sNetworkFilePath, "NULL");

	while (++nCount < argc)
	{
		if (!STRICMP(argv[nCount], "-mode"))
		{
			networkMain->nOpMode = atoi(argv[++nCount]);
		}
		else if (!STRICMP(argv[nCount], "-config"))
		{
			// Read Config File
			sprintf(networkMain->sConfigFilePath, "%s", argv[++nCount]);

			strcpy(sTemp, networkMain->sConfigFilePath);
			STRREV(sTemp);

			int bReverse = 1;
			for (i = 0, j = 0; i < (int)strlen(sTemp); ++i)
			{
				if (sTemp[i] == '.')
				{
					for (++i; i < (int)strlen(sTemp); ++i)
					{
						if (sTemp[i] == '\\')
						{
							networkMain->sConfigFile[j] = '\0';
							STRREV(networkMain->sConfigFile);
							i = (int)strlen(sTemp);
							bReverse = 0;
						}
						else
						{
							networkMain->sConfigFile[j++] = sTemp[i];
						}
					}

					if (bReverse)
						STRREV(networkMain->sConfigFile);
				}
			}

			ReadFile_Config(networkMain, networkMain->sConfigFilePath);
		}
		else if (!STRICMP(argv[nCount], "-network"))
		{
			sprintf(networkMain->sNetworkFilePath, "%s", argv[++nCount]);
		}
		else if (!STRICMP(argv[nCount], "-h"))
		{
			DisplayHelpFile();
			while (1)
				;
		}
		else
		{
			HoldDisplay("\n\n-h for help file\n\n");
		}
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	PrintHeader_Network(networkMain, fpFileOut);

	// Read Training and Test data
	Get_InputData(&inputData, &inputTrainingData, &inputVerifyData, &inputTestingData, networkMain->nTrainVerifySplit, networkMain->sDrive, networkMain->sTrainingFilePath, networkMain->sTestingFilePath, networkMain->sDataSource, &networkMain->classHead, &networkMain->nClassCount, &networkMain->nDataSource, &networkMain->nRowCount, &networkMain->nColumnCount);
	networkMain->nInputRowCount = inputTrainingData->nRowCount;
	networkMain->nInputColumnCount = inputTrainingData->nColumnCount;

	CreateMatrix(&networkMain->nMatrix, networkMain->nClassCount);
	CreateMatrix(&nMaxMatrix, networkMain->nClassCount);

	// Get Class Members
	printf("Calculate Input Data Statistics                                                           \r");
	SetStatistics_InputData(inputTrainingData, inputVerifyData, inputTestingData, networkMain->classHead, &networkMain->nClassCount, &networkMain->nClassMemberCount);

	DisplayInputData(networkMain, NULL);
	if (fpFileOut != NULL)
		DisplayInputData(networkMain, fpFileOut);

	if ((networkMain->fInputArray = (float *)calloc(networkMain->nRowCount * networkMain->nColumnCount, sizeof(float))) == NULL)
		exit(0);
	// Print input data details for debugging
	printf("Input Training Data:\n");
	printf("Row Count: %d\n", inputTrainingData->nRowCount);
	printf("Column Count: %d\n", inputTrainingData->nColumnCount);
	printf("Number of Classes: %d\n", networkMain->nClassCount);
	fTestAccuracy = 0.0f;

	if (networkMain->nOpMode == ANALYZE_INPUT)
	{
		AnalyzeInputs(networkMain->fInputArray, inputTrainingData, inputTestingData);
	}
	else if (networkMain->nOpMode == REBUILD_COMPLETE_NETWORK)
	{
		printf("REBUILD_COMPLETE_NETWORK\n");
		RebuildCompleteNetwork_ConstructNetworks(networkMain, fpFileOut, SHOW_DATA, inputTrainingData);
	}
	else if (networkMain->nOpMode == BUILD_COMPLETE_NETWORK)
	{
		printf("BUILD_COMPLETE_NETWORK\n");
		fTestAccuracy = BuildCompleteNetwork_ConstructNetworks(networkMain, inputTrainingData, inputVerifyData, inputTestingData, inputData, fpFileOut);
	}
	else if (networkMain->nOpMode == REBUILD_CLASS_LEVEL_NETWORK)
	{
		printf("REBUILD_CLASS_LEVEL_NETWORK\n");
		fTestAccuracy = RebuildClassLevelNetwork_ConstructNetworks(networkMain, inputTrainingData, inputVerifyData, inputTestingData, inputData, fpFileOut);
	}
	else if (networkMain->nOpMode == BUILD_CLASS_LEVEL_NETWORK) // D:\Data\MNIST\config\CLN\cln.cfg
	{
		printf("BUILD_CLASS_LEVEL_NETWORK\n");
		fTestAccuracy = BuildClassLevelNetwork_ConstructNetworks(networkMain, inputTrainingData, inputVerifyData, inputTestingData, inputData, fpFileOut);
	}
	else if (networkMain->nOpMode == NEUROGENESIS)
	{
		printf("NEUROGENESIS\n");
		NeuroGenesis(networkMain, inputData, inputTestingData, &inputTrainingData, &inputVerifyData, networkMain->fLearningRate, networkMain->fThreshold, networkMain->fMinimumThreshold, networkMain->fMaximumThreshold, &macData, nMACCount, 0, &fTestAccuracy, fpFileOut);
	}
	else if (networkMain->nOpMode == REBUILD_NEUROGENESIS)
	{
		printf("REBUILD_NEUROGENESIS\n");
		NeuroGenesis(networkMain, inputData, inputTestingData, &inputTrainingData, &inputVerifyData, networkMain->fLearningRate, networkMain->fThreshold, networkMain->fMinimumThreshold, networkMain->fMaximumThreshold, &macData, nMACCount, 1, &fTestAccuracy, fpFileOut);
	}
	else
	{
		printf("UNKNOWN\n");
		fTestAccuracy = BuildCompleteNetwork_ConstructNetworks(networkMain, inputTrainingData, inputVerifyData, inputTestingData, inputData, fpFileOut);
	}

	if (networkMain->nSIC > 0)
	{
		// LARGE_INTEGER	lnFrequency;
		// LARGE_INTEGER	lnStart;
#ifdef _WINDOWS
		LARGE_INTEGER lnFrequency;
		LARGE_INTEGER lnStart;
#else
		struct timespec start, end;
#endif
		structLayer *layerCur;
		float fTargetAccuracy = 1.0f;
		int nPrevWeightCount = -1;
		int nReturn = 1;
		int *nAdditionArray = NULL;
		int *nMultiplicationArray = NULL;
		int nLayerCount;
		int nLayer;
		int nLoop = 0;
		int nMaxCluster = 0;

		if (networkMain->nSIC == 1)
			printf("----- SIC -----\n");
		else if (networkMain->nSIC == 2)
			printf("----- SIC Prune -----\n");
		else if (networkMain->nSIC == 3)
			printf("----- Prune SIC -----\n");
		else
			printf("----- Prune -----\n");

		fflush(stdout);

		nLoop = 1;
		MarkInputData(networkMain, inputTrainingData, inputVerifyData, inputTestingData);
		fTargetAccuracy = networkMain->clnHead->fTrainAccuracy;
		nMaxCluster = 10;

// Start timer
#ifdef _WINDOWS
		StartTimer(&lnFrequency, &lnStart);
#else
		clock_gettime(CLOCK_MONOTONIC, &start);
#endif

		do
		{
			nPrevWeightCount = networkMain->clnHead->nWeightCount;

			GroupByDifference(inputTrainingData->data, inputTrainingData->nInputCount, nMaxCluster);
			SortByDifference(inputTrainingData->data, inputTrainingData->nInputCount);

			if (networkMain->nSIC == 1 || (networkMain->nSIC == 2 && nLoop == 1))
				SIC_V4(networkMain, inputTrainingData, inputTestingData, nMaxCluster, fTargetAccuracy);

			if (networkMain->nSIC == 2 || (networkMain->nSIC == 3 && nLoop == 1) || networkMain->nSIC == 4)
				PrunePerceptronWeightsSinglePass(networkMain, inputTrainingData, inputTestingData, 1);

			if (networkMain->nSIC == 3)
				SIC_V4(networkMain, inputTrainingData, inputTestingData, nMaxCluster, fTargetAccuracy);

			ForwardPropagateAnalyze_Train(networkMain->clnHead->macData, networkMain->clnHead->nMACCount, &nAdditionArray, &nMultiplicationArray, &nLayerCount, 0);

			for (layerCur = networkMain->clnHead->layerHead, nLayer = 0; layerCur != NULL; layerCur = layerCur->next, ++nLayer)
			{
				layerCur->nAdditions = nAdditionArray[nLayer];
				layerCur->nMultiplications = nMultiplicationArray[nLayer];
			}

			DescribeClassLevelNetwork(networkMain->clnHead, NULL);

			nLoop = 1;
		} while (networkMain->clnHead->nWeightCount != nPrevWeightCount);

#ifdef _WINDOWS
		FormatTime(EndTimer(&lnFrequency, &lnStart), sTimeBuffer);
#else
		clock_gettime(CLOCK_MONOTONIC, &end);
		double fElapsedTime = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
		snprintf(sTimeBuffer, sizeof(sTimeBuffer), "%f seconds", fElapsedTime);
#endif

		printf("\n%s\n", sTimeBuffer);

		DeleteCLN_ClassLevelNetworks(&networkMain->clnHead, 0);
		strcpy(networkMain->sInputStructurePath, networkMain->sDNXOutputPath);
		RebuildCompleteNetwork_ConstructNetworks(networkMain, fpFileOut, SHOW_DATA, inputTrainingData);
	}

	if (networkMain->nPruneNetwork == PRUNE_AFTER)
	{
		structLayer *layerCur;
		FILE *fpFileOut = NULL;
		float fTestAccuracy = 0.0f;
		float fValidateAccuracy = 0.0f;
		float fTrainAccuracy = 0.0f;
		float fAccuracy = 0.0f;
		float fPreAccuracy = 0.0f;
		int nTotalWeightCount = 0;
		char sTimeBuffer[32];
		int nMaxCluster = 1;
		int nStartWeightCount = networkMain->clnHead->nWeightCount;
		int nFailedCount = 0;
		int nReturn = 1;

		CopyWeights(networkMain->clnHead, NETWORK_TO_MEMORY);

		InferCLN_Inference(networkMain->clnHead, networkMain->classHead, inputTrainingData, HIDE_DATA, networkMain->nMatrix, networkMain->fInputArray, 1, networkMain->sDrive, networkMain->sTitle, DO_NOT_MARK, NO_THRESHOLD, fpFileOut, sTimeBuffer, MARK_CORRECT_CLASSIFICATION, -1);
		fTrainAccuracy = ScoreClassLevelNetworkMatrix(networkMain->clnHead->nNetworkType, 0, networkMain->classHead, networkMain->nMatrix, HIDE_MATRIX, fpFileOut);
		printf("Pre_Prune_Train: %0.4f\t%s\n", fTrainAccuracy, sTimeBuffer);

		InferCLN_Inference(networkMain->clnHead, networkMain->classHead, inputVerifyData, HIDE_DATA, networkMain->nMatrix, networkMain->fInputArray, 1, networkMain->sDrive, networkMain->sTitle, DO_NOT_MARK, NO_THRESHOLD, fpFileOut, sTimeBuffer, MARK_CORRECT_CLASSIFICATION, -1);
		fValidateAccuracy = ScoreClassLevelNetworkMatrix(networkMain->clnHead->nNetworkType, 0, networkMain->classHead, networkMain->nMatrix, HIDE_MATRIX, fpFileOut);
		printf("Pre_Prune_Validate: %0.4f\t%s\n", fValidateAccuracy, sTimeBuffer);

		InferCLN_Inference(networkMain->clnHead, networkMain->classHead, inputTestingData, HIDE_DATA, networkMain->nMatrix, networkMain->fInputArray, 1, networkMain->sDrive, networkMain->sTitle, DO_NOT_MARK, NO_THRESHOLD, fpFileOut, sTimeBuffer, MARK_CORRECT_CLASSIFICATION, -1);
		fTestAccuracy = ScoreClassLevelNetworkMatrix(networkMain->clnHead->nNetworkType, 0, networkMain->classHead, networkMain->nMatrix, HIDE_MATRIX, fpFileOut);
		printf("Pre_Prune_Test: %0.4f\t%s\n\n", fTestAccuracy, sTimeBuffer);

		GroupByDifference(inputTrainingData->data, inputTrainingData->nInputCount, 10);
		SortByDifference(inputTrainingData->data, inputTrainingData->nInputCount);

		int *nAdditionArray = NULL;
		int *nMultiplicationArray = NULL;
		int nLayerCount;
		int nLayer;

		PrunePerceptronWeights(networkMain, inputTrainingData, inputTestingData, &fAccuracy, 1);
		ForwardPropagateAnalyze_Train(networkMain->clnHead->macData, networkMain->clnHead->nMACCount, &nAdditionArray, &nMultiplicationArray, &nLayerCount, 0);

		for (layerCur = networkMain->clnHead->layerHead, nLayer = 0; layerCur != NULL; layerCur = layerCur->next, ++nLayer)
		{
			layerCur->nAdditions = nAdditionArray[nLayer];
			layerCur->nMultiplications = nMultiplicationArray[nLayer];
		}

		DescribeClassLevelNetwork(networkMain->clnHead, NULL);
	}
	else if (networkMain->nPruneNetwork == PRUNE_AFTER_ZERO)
	{
		int nNoProgress;
		int nMinWeightCount = 999999;

		fMaxTestPercent = InferCLN_Inference(networkMain->clnHead, networkMain->classHead, inputTestingData, HIDE_DATA, networkMain->nMatrix, networkMain->fInputArray, 1, networkMain->sDrive, networkMain->sTitle, DO_NOT_MARK, NO_THRESHOLD, fpFileOut, sTimeBuffer, 0, -1);

		for (networkMain->fPruneConvThreshold = 0.0f; networkMain->fPruneConvThreshold < 0.05f; networkMain->fPruneConvThreshold += 0.0001f)
		{
			nNoProgress = 0;

			for (networkMain->fPruneFCThreshold = 0.0f; networkMain->fPruneFCThreshold < 0.05f; networkMain->fPruneFCThreshold += 0.0001f)
			{
				DeleteAll_ClassLevelNetworks(&networkMain->clnHead);
				RebuildCompleteNetwork_ConstructNetworks(networkMain, fpFileOut, HIDE_DATA, inputTrainingData);

				if (1)
					PruneWeights(&networkMain->clnHead->layerHead, networkMain->fPruneConvThreshold, networkMain->fPruneFCThreshold);
				else
					PruneWeights_V2(networkMain, &networkMain->clnHead->layerHead, inputVerifyData, inputTrainingData);

				GetClassLevelNetworkWeightCount(networkMain->clnHead);
				CreateMACArray_ClassLevelNetworks(networkMain->clnHead);

				if (networkMain->clnHead->nWeightCount <= nMinWeightCount)
				{
					fTestAccuracy = InferCLN_Inference(networkMain->clnHead, networkMain->classHead, inputTestingData, HIDE_DATA, networkMain->nMatrix, networkMain->fInputArray, 1, networkMain->sDrive, networkMain->sTitle, DO_NOT_MARK, NO_THRESHOLD, fpFileOut, sTimeBuffer, 0, -1);
					printf("%0.2f\t%d\t%0.4f\t%0.4f\t%d                           \r", fTestAccuracy * 100.0f, networkMain->clnHead->nWeightCount, networkMain->fPruneConvThreshold, networkMain->fPruneFCThreshold, nNoProgress);

					if (fTestAccuracy >= fMaxTestPercent)
					{
						if (fMaxTestPercent == 0.0f)
							fMaxTestPercent = fTestAccuracy;

						nMinWeightCount = networkMain->clnHead->nWeightCount;
						printf("\n");
						nNoProgress = 0;
					}
					else
						++nNoProgress;
				}
				else
				{
					++nNoProgress;
					printf("%0.2f\t%d\t%0.4f\t%0.4f\t%d                           \r", 0.0f, networkMain->clnHead->nWeightCount, networkMain->fPruneConvThreshold, networkMain->fPruneFCThreshold, nNoProgress);
				}
			}
		}
	}

	if (networkMain->nTrainInferenceExecute)
	{
		InferCLN_Inference(networkMain->clnHead, networkMain->classHead, inputTrainingData, HIDE_DATA, networkMain->nMatrix, networkMain->fInputArray, 1, networkMain->sDrive, networkMain->sTitle, DO_NOT_MARK, NO_THRESHOLD, fpFileOut, sTimeBuffer, 0, -1);
		fTrainAccuracy = ScoreClassLevelNetworkMatrix(networkMain->clnHead->nNetworkType, 0, networkMain->classHead, networkMain->nMatrix, HIDE_MATRIX, fpFileOut);
		printf("DNX_Train: %f\t%s\n", fTrainAccuracy, sTimeBuffer);
	}

	if (networkMain->nValidateInferenceExecute)
	{
		InferCLN_Inference(networkMain->clnHead, networkMain->classHead, inputVerifyData, HIDE_DATA, networkMain->nMatrix, networkMain->fInputArray, 1, networkMain->sDrive, networkMain->sTitle, DO_NOT_MARK, NO_THRESHOLD, fpFileOut, sTimeBuffer, 0, -1);
		fVerifyAccuracy = ScoreClassLevelNetworkMatrix(networkMain->clnHead->nNetworkType, 0, networkMain->classHead, networkMain->nMatrix, HIDE_MATRIX, fpFileOut);
		printf("DNX_Validate: %f\t%s\n", fVerifyAccuracy, sTimeBuffer);
	}

	if (networkMain->nTestInferenceExecute)
	{
		InferCLN_Inference(networkMain->clnHead, networkMain->classHead, inputTestingData, HIDE_DATA, networkMain->nMatrix, networkMain->fInputArray, 1, networkMain->sDrive, networkMain->sTitle, DO_NOT_MARK, NO_THRESHOLD, fpFileOut, sTimeBuffer, 0, -1);
		fTestAccuracy = ScoreClassLevelNetworkMatrix(networkMain->clnHead->nNetworkType, 0, networkMain->classHead, networkMain->nMatrix, HIDE_MATRIX, fpFileOut);
		printf("DNX_Test: %f\t%s\n\n", fTestAccuracy, sTimeBuffer);
	}

	
	if ((strlen(networkMain->sDNAOutputPath) > 0) && (networkMain->nOpMode != ANALYZE_INPUT)){
		SaveNetwork(networkMain, networkMain->sDNAOutputPath);
	}

	if (strlen(networkMain->sDNXOutputPath) > 0)
	{
		//		ConvertToDNX(networkMain, networkMain->fInputArray);
		//		AnalyzeCLN(networkMain->clnHead, 0);
		DescribeClassLevelNetwork(networkMain->clnHead, NULL);
		CreateMACArrayFromDNX_ClassLevelNetworks(networkMain->clnHead);

		printf("Converted to DNX\n");

		InferCLN_Inference(networkMain->clnHead, networkMain->classHead, inputTrainingData, HIDE_DATA, networkMain->nMatrix, networkMain->fInputArray, 1, networkMain->sDrive, networkMain->sTitle, DO_NOT_MARK, NO_THRESHOLD, fpFileOut, sTimeBuffer, 0, -1);
		fTrainAccuracy = ScoreClassLevelNetworkMatrix(networkMain->clnHead->nNetworkType, 0, networkMain->classHead, networkMain->nMatrix, HIDE_MATRIX, fpFileOut);
		printf("DNX_Train: %f\t%s\n", fTrainAccuracy, sTimeBuffer);

		InferCLN_Inference(networkMain->clnHead, networkMain->classHead, inputVerifyData, HIDE_DATA, networkMain->nMatrix, networkMain->fInputArray, 1, networkMain->sDrive, networkMain->sTitle, DO_NOT_MARK, NO_THRESHOLD, fpFileOut, sTimeBuffer, 0, -1);
		fVerifyAccuracy = ScoreClassLevelNetworkMatrix(networkMain->clnHead->nNetworkType, 0, networkMain->classHead, networkMain->nMatrix, HIDE_MATRIX, fpFileOut);
		printf("DNX_Validate: %f\t%s\n", fVerifyAccuracy, sTimeBuffer);

		InferCLN_Inference(networkMain->clnHead, networkMain->classHead, inputTestingData, HIDE_DATA, networkMain->nMatrix, networkMain->fInputArray, 1, networkMain->sDrive, networkMain->sTitle, DO_NOT_MARK, NO_THRESHOLD, fpFileOut, sTimeBuffer, 0, -1);
		fTestAccuracy = ScoreClassLevelNetworkMatrix(networkMain->clnHead->nNetworkType, 0, networkMain->classHead, networkMain->nMatrix, HIDE_MATRIX, fpFileOut);
		printf("DNX_Test: %f\t%s\n\n", fTestAccuracy, sTimeBuffer);

		if (strlen(networkMain->sDNXOutputPath) < 1)
			sprintf(networkMain->sDNXOutputPath, "%s.dnx", networkMain->sConfigFile);

		SaveNetwork(networkMain, networkMain->sDNXOutputPath);
	}

	// DumpWeights(networkMain);

#ifdef _WINDOWS
	HoldDisplay("Done");
#endif

	return 0;
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
structArchitecture *AllocateArchitecture(int *nID)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structArchitecture *newArchitecture = NULL;

	if ((newArchitecture = (structArchitecture *)calloc(1, sizeof(structArchitecture))) == NULL)
		DisplayMessage("Memory calloc Error: AllocateArchitecture() newArchitecture", PAUSE);

	newArchitecture->nID = (*nID)++;

	return (newArchitecture);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void AddArchitecture(structArchitecture **head, structArchitecture *newArchitecture)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structArchitecture *cur = *head;

	if (*head == NULL)
	{
		*head = newArchitecture;
		(*head)->nID = 0;
	}
	else
	{
		while (cur->next != NULL)
		{
			cur = cur->next;
		}

		cur->next = newArchitecture;
		cur->next->nID = cur->nID + 1;

		newArchitecture->prev = cur;
	}

	return;
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
structArchitecture *DeleteArchitecture(structArchitecture **head, int nID)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structArchitecture *prev = NULL;
	structArchitecture *target = (*head);

	// If head node itself holds the key to be deleted
	if (target != NULL && target->nID == nID)
	{
		(*head) = target->next; // Changed head

		free(target); // free old head

		return ((*head));
	}

	// Search for the key to be deleted, keep track of the
	// previous node as we need to change 'prev->next'
	while (target != NULL && target->nID != nID)
	{
		prev = target;
		target = target->next;
	}

	// If key was not present in linked list
	if (target == NULL)
		return ((*head));

	// Unlink the node from linked list
	prev->next = target->next;

	if (prev->next != NULL) // Tail
		prev->next->prev = prev;

	free(target); // Free memory

	return (prev);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
int DeleteAllArchitecture(structArchitecture **head)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structArchitecture *cur = *head;
	structArchitecture *next = NULL;
	int nDeleteCount = 0;

	while (cur != NULL)
	{
		next = cur->next;
		DeleteArchitecture(head, cur->nID);
		++nDeleteCount;

		cur = next;
	}

	*head = NULL;

	return (nDeleteCount);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
float SIC_Individual(structNetwork *networkMain, structInput *inputData, structInput *inputTestingData, structInput *inputVerifyData, structInput *inputTrainingData)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structLayer *layerCur;
	FILE *fpFileOut = NULL;
	float fTestAccuracy = 0.0f;
	float fValidateAccuracy = 0.0f;
	float fTrainAccuracy = 0.0f;
	float fAccuracy = 0.0f;
	float fPreAccuracy = 0.0f;
	int nTotalWeightCount = 0;
	char sTimeBuffer[32];
	int nMaxCluster = 1;
	int nStartWeightCount = networkMain->clnHead->nWeightCount;
	int nFailedCount = 0;
	int nReturn = 1;

	printf("Makeing Sure I enter SIC_Individual\n");

	CopyWeights(networkMain->clnHead, NETWORK_TO_MEMORY);

	InferCLN_Inference(networkMain->clnHead, networkMain->classHead, inputTrainingData, HIDE_DATA, networkMain->nMatrix, networkMain->fInputArray, 1, networkMain->sDrive, networkMain->sTitle, DO_NOT_MARK, NO_THRESHOLD, fpFileOut, sTimeBuffer, MARK_CORRECT_CLASSIFICATION, -1);
	fTrainAccuracy = ScoreClassLevelNetworkMatrix(networkMain->clnHead->nNetworkType, 0, networkMain->classHead, networkMain->nMatrix, HIDE_MATRIX, fpFileOut);
	printf("PRE_SIC_Train: %0.4f\t%s\n", fTrainAccuracy, sTimeBuffer);

	InferCLN_Inference(networkMain->clnHead, networkMain->classHead, inputVerifyData, HIDE_DATA, networkMain->nMatrix, networkMain->fInputArray, 1, networkMain->sDrive, networkMain->sTitle, DO_NOT_MARK, NO_THRESHOLD, fpFileOut, sTimeBuffer, MARK_CORRECT_CLASSIFICATION, -1);
	fValidateAccuracy = ScoreClassLevelNetworkMatrix(networkMain->clnHead->nNetworkType, 0, networkMain->classHead, networkMain->nMatrix, HIDE_MATRIX, fpFileOut);
	printf("PRE_SIC_Validate: %0.4f\t%s\n", fValidateAccuracy, sTimeBuffer);

	InferCLN_Inference(networkMain->clnHead, networkMain->classHead, inputTestingData, HIDE_DATA, networkMain->nMatrix, networkMain->fInputArray, 1, networkMain->sDrive, networkMain->sTitle, DO_NOT_MARK, NO_THRESHOLD, fpFileOut, sTimeBuffer, MARK_CORRECT_CLASSIFICATION, -1);
	fTestAccuracy = ScoreClassLevelNetworkMatrix(networkMain->clnHead->nNetworkType, 0, networkMain->classHead, networkMain->nMatrix, HIDE_MATRIX, fpFileOut);
	printf("PRE_SIC_Test: %0.4f\t%s\n\n", fTestAccuracy, sTimeBuffer);

	GroupByDifference(inputTrainingData->data, inputTrainingData->nInputCount, 10);
	SortByDifference(inputTrainingData->data, inputTrainingData->nInputCount);

	int *nAdditionArray = NULL;
	int *nMultiplicationArray = NULL;
	int nLayerCount;
	int nLayer;

	SIC_V3(networkMain, inputTrainingData, inputTestingData, &fAccuracy, 1);
	ForwardPropagateAnalyze_Train(networkMain->clnHead->macData, networkMain->clnHead->nMACCount, &nAdditionArray, &nMultiplicationArray, &nLayerCount, 0);

	for (layerCur = networkMain->clnHead->layerHead, nLayer = 0; layerCur != NULL; layerCur = layerCur->next, ++nLayer)
	{
		layerCur->nAdditions = nAdditionArray[nLayer];
		layerCur->nMultiplications = nMultiplicationArray[nLayer];
	}

	DescribeClassLevelNetwork(networkMain->clnHead, NULL);

	InferCLN_Inference(networkMain->clnHead, networkMain->classHead, inputTrainingData, HIDE_DATA, networkMain->nMatrix, networkMain->fInputArray, 1, networkMain->sDrive, networkMain->sTitle, DO_NOT_MARK, NO_THRESHOLD, fpFileOut, sTimeBuffer, MARK_CORRECT_CLASSIFICATION, -1);
	fTrainAccuracy = ScoreClassLevelNetworkMatrix(networkMain->clnHead->nNetworkType, 0, networkMain->classHead, networkMain->nMatrix, HIDE_MATRIX, fpFileOut);
	printf("POST_SIC_Train: %0.4f\t%s\n", fTrainAccuracy, sTimeBuffer);

	InferCLN_Inference(networkMain->clnHead, networkMain->classHead, inputVerifyData, HIDE_DATA, networkMain->nMatrix, networkMain->fInputArray, 1, networkMain->sDrive, networkMain->sTitle, DO_NOT_MARK, NO_THRESHOLD, fpFileOut, sTimeBuffer, MARK_CORRECT_CLASSIFICATION, -1);
	fValidateAccuracy = ScoreClassLevelNetworkMatrix(networkMain->clnHead->nNetworkType, 0, networkMain->classHead, networkMain->nMatrix, HIDE_MATRIX, fpFileOut);
	printf("POST_SIC_Validate: %0.4f\t%s\n", fValidateAccuracy, sTimeBuffer);

	InferCLN_Inference(networkMain->clnHead, networkMain->classHead, inputTestingData, HIDE_DATA, networkMain->nMatrix, networkMain->fInputArray, 1, networkMain->sDrive, networkMain->sTitle, DO_NOT_MARK, NO_THRESHOLD, fpFileOut, sTimeBuffer, MARK_CORRECT_CLASSIFICATION, -1);
	fTestAccuracy = ScoreClassLevelNetworkMatrix(networkMain->clnHead->nNetworkType, 0, networkMain->classHead, networkMain->nMatrix, HIDE_MATRIX, fpFileOut);
	printf("POST_SIC_Test: %0.4f\t%s\n\n", fTestAccuracy, sTimeBuffer);

	if (nReturn == 1)
		nReturn = nTotalWeightCount - networkMain->clnHead->nWeightCount;

	return (0.0f);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void ConvertToDNX(structNetwork *networkMain, float *fInputArray)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structLayer *layerCur;
	structPerceptron *perceptronCurHead;
	structPerceptron *perceptronCur;
	structPerceptron *perceptronDuplicate;
	structSynapse *synapseCur;
	structSynapse *synapseDuplicate;
	structSynapse *synapseCluster;
	float *fWeights;
	int nWeightCount;
	int nInputCount;
	int nWeightStart;
	int nMultipleClusters;
	int *nTempInputArray;
	float **fTempInputArray;

	for (layerCur = networkMain->clnHead->layerHead; layerCur != NULL; layerCur = layerCur->next)
	{
		// Count Weights
		nWeightCount = 0;
		nMultipleClusters = 0;

		if (layerCur->nLayerType == CONV_2D_LAYER || layerCur->nLayerType == CONV_3D_LAYER)
		{
			for (perceptronCurHead = layerCur->perceptronHead; perceptronCurHead != NULL; perceptronCurHead = perceptronCurHead->nextHead)
			{
				for (perceptronDuplicate = perceptronCurHead->next, perceptronCur = perceptronCurHead; perceptronDuplicate != NULL; perceptronDuplicate = perceptronDuplicate->next, perceptronCur = perceptronCur->next)
				{
					perceptronDuplicate->nClusterCount = perceptronCur->nClusterCount;
				}
			}

			for (perceptronCurHead = layerCur->perceptronHead; perceptronCurHead != NULL; perceptronCurHead = perceptronCurHead->nextHead)
			{
				for (perceptronDuplicate = perceptronCurHead->next; perceptronDuplicate != NULL; perceptronDuplicate = perceptronDuplicate->next)
				{
					for (synapseCur = perceptronCurHead->synapseHead->next, synapseDuplicate = perceptronDuplicate->synapseHead->next; synapseCur != NULL; synapseCur = synapseCur->next, synapseDuplicate = synapseDuplicate->next)
					{
						synapseDuplicate->nCluster = synapseCur->nCluster;
					}
				}
			}

			for (perceptronCur = layerCur->perceptronHead; perceptronCur != NULL; perceptronCur = perceptronCur->nextHead)
			{
				// Bias
				++nWeightCount;
				nWeightCount += perceptronCur->nClusterCount;
			}

			fWeights = (float *)calloc(nWeightCount, sizeof(float));

			// Copy Weights and Delete Redundant Synapses
			for (perceptronCurHead = layerCur->perceptronHead, nWeightCount = 0; perceptronCurHead != NULL; perceptronCurHead = perceptronCurHead->nextHead)
			{
				for (perceptronDuplicate = perceptronCurHead->next; perceptronDuplicate != NULL; perceptronDuplicate = perceptronDuplicate->next)
				{
					for (synapseCur = perceptronCurHead->synapseHead->next, synapseDuplicate = perceptronDuplicate->synapseHead->next; synapseCur != NULL; synapseCur = synapseCur->next, synapseDuplicate = synapseDuplicate->next)
					{
						if (synapseDuplicate->nCluster != synapseCur->nCluster)
							printf("Error\n");
					}
				}

				for (perceptronCur = perceptronCurHead; perceptronCur != NULL; perceptronCur = perceptronCur->next)
				{
					if (perceptronCur == perceptronCurHead) // First Kernel Position
					{
						// Bias
						fWeights[nWeightCount++] = *perceptronCur->synapseHead->fWeight;

						for (synapseCur = perceptronCur->synapseHead->next; synapseCur != NULL; synapseCur = synapseCur->next)
						{
							fWeights[nWeightCount++] = *synapseCur->fWeight;

							// Get Input Count
							for (synapseCluster = synapseCur->next, synapseCur->nInputCount = 0; synapseCluster != NULL; synapseCluster = synapseCluster->next)
							{
								if (synapseCluster->nCluster == synapseCur->nCluster)
								{
									++synapseCur->nInputCount;
								}
							}

							synapseCur->nInputArray = (int *)calloc(synapseCur->nInputCount, sizeof(int));
							synapseCur->fInputArray = (float **)calloc(synapseCur->nInputCount, sizeof(float *));

							for (synapseCluster = synapseCur->next, synapseCur->nInputCount = 0; synapseCluster != NULL; synapseCluster = synapseCluster->next)
							{
								if (synapseCluster->nCluster == synapseCur->nCluster)
								{
									if (synapseCluster->perceptronConnectTo == NULL) // input is image
									{
										synapseCur->nInputArray[synapseCur->nInputCount] = synapseCluster->nInputArrayIndex;
										synapseCur->fInputArray[synapseCur->nInputCount++] = synapseCluster->fInput;
									}
									else
									{
										synapseCur->nInputArray[synapseCur->nInputCount] = synapseCluster->perceptronConnectTo->nID;
										synapseCur->fInputArray[synapseCur->nInputCount++] = &synapseCluster->perceptronConnectTo->fOutput;
									}

									synapseCluster = Delete_Synapse(&perceptronCur->synapseHead, synapseCluster->nID);
								}
							}
						}
					}
					else
					{
						for (synapseCur = perceptronCur->synapseHead->next; synapseCur != NULL; synapseCur = synapseCur->next)
						{
							// Get Input Count
							for (synapseCluster = synapseCur->next, synapseCur->nInputCount = 0; synapseCluster != NULL; synapseCluster = synapseCluster->next)
							{
								if (synapseCluster->nCluster == synapseCur->nCluster)
								{
									++synapseCur->nInputCount;
								}
							}

							synapseCur->nInputArray = (int *)calloc(synapseCur->nInputCount, sizeof(int));
							synapseCur->fInputArray = (float **)calloc(synapseCur->nInputCount, sizeof(float *));

							for (synapseCluster = synapseCur->next, synapseCur->nInputCount = 0; synapseCluster != NULL; synapseCluster = synapseCluster->next)
							{
								if (synapseCluster->nCluster == synapseCur->nCluster)
								{
									if (synapseCluster->perceptronConnectTo == NULL) // input is image
									{
										synapseCur->nInputArray[synapseCur->nInputCount] = synapseCluster->nInputArrayIndex;
										synapseCur->fInputArray[synapseCur->nInputCount++] = synapseCluster->fInput;
									}
									else
									{
										synapseCur->nInputArray[synapseCur->nInputCount] = synapseCluster->perceptronConnectTo->nID;
										synapseCur->fInputArray[synapseCur->nInputCount++] = &synapseCluster->perceptronConnectTo->fOutput;
									}

									synapseCluster = Delete_Synapse(&perceptronCur->synapseHead, synapseCluster->nID);
								}
							}
						}
					}
				}
			}

			free(layerCur->fWeightArray);
			layerCur->fWeightArray = fWeights;

			// Assign Weights
			for (perceptronCurHead = layerCur->perceptronHead, nWeightStart = 0; perceptronCurHead != NULL; perceptronCurHead = perceptronCurHead->nextHead)
			{
				for (perceptronCur = perceptronCurHead; perceptronCur != NULL; perceptronCur = perceptronCur->next)
				{
					nWeightCount = nWeightStart;

					// Bias
					perceptronCur->synapseHead->fWeight = &layerCur->fWeightArray[nWeightCount++];
					perceptronCur->nSynapseCount = 1;

					for (synapseCur = perceptronCur->synapseHead->next; synapseCur != NULL; synapseCur = synapseCur->next)
					{
						synapseCur->fWeight = &layerCur->fWeightArray[nWeightCount++];
						++perceptronCur->nSynapseCount;
					}
				}

				nWeightStart = nWeightCount;
			}
		}
		else if (layerCur->nLayerType == FULLY_CONNECTED_LAYER || layerCur->nLayerType == CLASSIFIER_LAYER)
		{
			nMultipleClusters = 0;

			for (perceptronCur = layerCur->perceptronHead; perceptronCur != NULL; perceptronCur = perceptronCur->next)
			{
				// Bias
				++nWeightCount;
				nWeightCount += perceptronCur->nClusterCount;

				for (synapseCur = perceptronCur->synapseHead->next; synapseCur != NULL; synapseCur = synapseCur->next)
					++nWeightCount;

				if (perceptronCur->nClusterCount > 1)
					nMultipleClusters = 1;
			}

			fWeights = (float *)calloc(nWeightCount, sizeof(float));

			for (perceptronCur = layerCur->perceptronHead, nWeightCount = 0; perceptronCur != NULL; perceptronCur = perceptronCur->next)
			{
				// Bias
				fWeights[nWeightCount++] = *perceptronCur->synapseHead->fWeight;

				for (synapseCur = perceptronCur->synapseHead->next; synapseCur != NULL; synapseCur = synapseCur->next)
				{
					nInputCount = synapseCur->nInputCount;
					fWeights[nWeightCount++] = *synapseCur->fWeight;
					nMultipleClusters = 0;

					for (synapseCluster = synapseCur->next; synapseCluster != NULL; synapseCluster = synapseCluster->next)
					{
						if (synapseCluster->nCluster == synapseCur->nCluster)
						{
							if (!synapseCluster->nInputCount)
								++nInputCount;
							else
								nInputCount += synapseCluster->nInputCount;

							nMultipleClusters = 1;
						}
					}

					if (nMultipleClusters)
					{
						nTempInputArray = (int *)calloc(nInputCount, sizeof(int));
						fTempInputArray = (float **)calloc(nInputCount, sizeof(float *));

						for (nInputCount = 0; nInputCount < synapseCur->nInputCount; ++nInputCount)
						{
							nTempInputArray[nInputCount] = synapseCur->nInputArray[nInputCount];
							fTempInputArray[nInputCount] = synapseCur->fInputArray[nInputCount];
						}

						free(synapseCur->nInputArray);
						free(synapseCur->fInputArray);

						for (synapseCluster = synapseCur->next; synapseCluster != NULL; synapseCluster = synapseCluster->next)
						{
							if (synapseCluster->nCluster == synapseCur->nCluster)
							{
								if (synapseCluster->perceptronConnectTo == NULL) // input is image
								{
									nTempInputArray[nInputCount] = synapseCluster->nInputArrayIndex;
									fTempInputArray[nInputCount++] = &fInputArray[synapseCluster->nInputArrayIndex];
								}
								else
								{
									nTempInputArray[nInputCount] = synapseCluster->perceptronConnectTo->nID;
									fTempInputArray[nInputCount++] = &synapseCluster->perceptronConnectTo->fOutput;
								}

								synapseCluster = Delete_Synapse(&perceptronCur->synapseHead, synapseCluster->nID);
							}
						}

						synapseCur->nInputCount = nInputCount;
						synapseCur->nInputArray = nTempInputArray;
						synapseCur->fInputArray = fTempInputArray;
					}
				}
			}

			free(layerCur->fWeightArray);
			layerCur->fWeightArray = fWeights;

			// Assign Weights
			for (perceptronCur = layerCur->perceptronHead, nWeightCount = 0; perceptronCur != NULL; perceptronCur = perceptronCur->next)
			{
				// Bias
				perceptronCur->synapseHead->fWeight = &layerCur->fWeightArray[nWeightCount++];
				perceptronCur->nSynapseCount = 1;

				for (synapseCur = perceptronCur->synapseHead->next; synapseCur != NULL; synapseCur = synapseCur->next)
				{
					synapseCur->fWeight = &layerCur->fWeightArray[nWeightCount++];
					++perceptronCur->nSynapseCount;
				}
			}
		}
		else if (layerCur->nLayerType == MAX_POOLING_LAYER)
		{
		}
	}
}

/*

LeNet5
Layer	Connectivity
Input	28×28 image
1	convolutional, 20 5×5 filters (stride=1), total 11 520 neurons, followed by ReLU
2	max pool, 2×2 window (stride=2), total 2 280 neurons
3	convolutional, 50 5×5 filters (stride=1), total 3 200 neurons, followed by ReLU
4	max pool, 2×2 window (stride=2), total 800 neurons
5	fully connected, 500 neurons and dropout with p=0.5, followed by ReLU
6 (output)	fully connected, 10 neurons and dropout with p=0.5, followed by softmax
P1= 430500 weights, P0= 580 biases




Layer	Connectivity
Input	3×32×32 image
1	convolutional, 128 (3×3) filters (stride=1), zero padding with size 1, total 131 072 neurons, followed by ReLU
2	convolutional, 128 (3×3) filters (stride=1), zero padding with size 1, total 131 072 neurons, followed by ReLU
3	max pool, 2×2 window (stride=2), total 32 768 neurons
4	convolutional, 256 (3×3) filters (stride=1), zero padding with size 1, total 65 536 neurons, followed by ReLU
5	convolutional, 256 (3×3) filters (stride=1), zero padding with size 1, total 65 536 neurons, followed by ReLU
6	max pool, 2×2 window (stride=2), total 16 384 neurons
7	convolutional, 512 (3×3) filters (stride=1), zero padding with size 1, total 32 768 neurons, followed by ReLU
8	convolutional, 512 (3×3) filters (stride=1), zero padding with size 1, total 32 768 neurons, followed by ReLU
9	max pool, 2×2 window (stride=2), total 8 192 neurons
10	fully connected, 1024 neurons and dropout with p=0.5, followed by ReLU
11	fully connected, 1024 neurons and dropout with p=0.5, followed by ReLU
12 (output)	fully connected, 10 neurons and dropout with p=0.5, followed by softmax
P1= 14 022 016 weights, P0= 3 850 biases



	for (size_t i = 0; i < 10000; i++)
	{
		// read label
		unsigned char label;
		ifs.read((char*) &label, 1);
		labels->push_back((int) label);

		// read image
		std::vector<unsigned char> image_c(32*32*3);
		ifs.read((char*) &image_c[0], 32*32*3);
		int width = 32+2*x_padding;
		int height = 32+2*y_padding;
		std::vector<float> image(height*width*3);

		// convert from RGB to BGR
		for (size_t c = 0; c < 3; c++)
		for (size_t y = 0; y < 32; y++)
		for (size_t x = 0; x < 32; x++)
			image[width * (y + y_padding) + x + x_padding + (3-c-1)*width*height] =
			(image_c[y * 32 + x+c*32*32] / 255.0f) * (scale_max - scale_min) + scale_min;

		images->push_back(image);

	}


*/

int counter = 0;
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void SaveNetwork(structNetwork *network, char *sFilePath)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	char newFileName[256];
	if ((strlen(sFilePath)) > 0)
	{
		if ((strstr(sFilePath, ".dna")) != NULL)
		{
			// Increment the counter and create the new file name
			printf("Saving into .dna\n");
			Write_DNA_Network(network, sFilePath);
		}
		else if ((strstr(sFilePath, ".dnx")) != NULL)
		{
			// ConvertToDNX(network, network->fInputArray);
			// AnalyzeCLN(network->clnHead, 0);
			DescribeClassLevelNetwork(network->clnHead, NULL);
			// CreateMACArrayFromDNX_ClassLevelNetworks(network->clnHead);

			printf("Converted to DNX\n");

			Write_DNX_Network(network, sFilePath);
		}
		else
			DisplayMessage("Function Error: SaveNetwork()", PAUSE);
	}

	sprintf(newFileName, "Network_%d.txt", counter);
	printf("Current network saved to %s\n", newFileName);
	Write_TXT_Network(network, newFileName);
	counter++;
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void SIC_V3(structNetwork *network, structInput *inputTrain, structInput *inputTest, float *fAccuracy, int nMaxCluster)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structCLN *cln = network->clnHead;
	structLayer *layerCur = NULL;
	structPerceptron *perceptronCur = NULL;
	float fPercent = 0.0f;
	int nTotalWeights = 0;
	int nCurrentWeights = 0;
	int bReduced;
	int nLoop = 0;
	int nLayer = 0;
	int nLayerCount = 0;
	char sTimeBuffer[32];
	int *nAdditionArray;
	int *nMultiplicationArray;

	GetClassLevelNetworkWeightCount(cln);

	do
	{
		nCurrentWeights = cln->nWeightCount;
		nTotalWeights = 0;

		for (layerCur = cln->layerHead; layerCur != NULL; layerCur = layerCur->next)
		{
			printf("%s\n", layerCur->sLayerName);

			if (layerCur->nLayerType == MAX_POOL_LAYER)
				continue;

			perceptronCur = layerCur->perceptronHead;

			while (perceptronCur != NULL)
			{
				// Copy Perceptron Weights
				CopyPerceptronWeights(perceptronCur, NETWORK_TO_MEMORY);

				for (perceptronCur->nClusterCount = 0, bReduced = 0; perceptronCur->nClusterCount < perceptronCur->nConnectionCount; ++perceptronCur->nClusterCount)
				{
					ClusterPerceptronWeights(perceptronCur);
					InferCLN_Inference(cln, network->classHead, inputTrain, HIDE_DATA, network->nMatrix, network->fInputArray, 1, network->sDrive, network->sTitle, DO_NOT_MARK, NO_THRESHOLD, NULL, sTimeBuffer, BREAK_ON_BAD_CLASSIFICATION, nMaxCluster);
					*fAccuracy = ScoreClassLevelNetworkMatrix(cln->nNetworkType, 0, network->classHead, network->nMatrix, HIDE_MATRIX, NULL);

					if ((bReduced = (*fAccuracy == 1.0f)))
						break;

					// Reset Perceptron Weights
					CopyPerceptronWeights(perceptronCur, MEMORY_TO_NETWORK);
					SortByMissCount(inputTrain->data, inputTrain->nInputCount);
				}

				nTotalWeights += perceptronCur->nClusterCount;

				if (perceptronCur->nWeightCount != perceptronCur->nClusterCount)
				{
					if (perceptronCur->nWeightCount > 0)
						fPercent = ((float)(perceptronCur->nWeightCount - perceptronCur->nClusterCount)) / (float)perceptronCur->nWeightCount;
					else
						fPercent = 0.0f;

					printf("%d\t%0.4f\t%d                     \n", perceptronCur->nIndex, fPercent, nTotalWeights);
				}
				else if (perceptronCur->nWeightCount == 0)
					printf("***** Delete %d\n", perceptronCur->nIndex);

				if (layerCur->nLayerType == CONV_2D_LAYER || layerCur->nLayerType == CONV_3D_LAYER)
					perceptronCur = perceptronCur->nextHead;
				else
					perceptronCur = perceptronCur->next;
			}

			printf("\n");

			ModifyArchitecture(cln);
			GetClassLevelNetworkWeightCount(cln);
			CreateMACArrayFromDNX_ClassLevelNetworks(cln);
		}

		InferCLN_Inference(cln, network->classHead, inputTrain, HIDE_DATA, network->nMatrix, network->fInputArray, 1, network->sDrive, network->sTitle, DO_NOT_MARK, NO_THRESHOLD, NULL, sTimeBuffer, 0, -1);
		*fAccuracy = ScoreClassLevelNetworkMatrix(network->clnHead->nNetworkType, 0, network->classHead, network->nMatrix, HIDE_MATRIX, NULL);
		printf("DNX_Train: %f\t%s\n", *fAccuracy, sTimeBuffer);

		InferCLN_Inference(network->clnHead, network->classHead, inputTest, HIDE_DATA, network->nMatrix, network->fInputArray, 1, network->sDrive, network->sTitle, DO_NOT_MARK, NO_THRESHOLD, NULL, sTimeBuffer, 0, -1);
		*fAccuracy = ScoreClassLevelNetworkMatrix(network->clnHead->nNetworkType, 0, network->classHead, network->nMatrix, HIDE_MATRIX, NULL);
		printf("DNX_Test: %f\t%s\n\n", *fAccuracy, sTimeBuffer);

		sprintf(network->sDNXOutputPath, "sic_%d_%0.4f.dnx", nLoop++, *fAccuracy);
		SaveNetwork(network, network->sDNXOutputPath);

		ForwardPropagateAnalyze_Train(cln->macData, network->clnHead->nMACCount, &nAdditionArray, &nMultiplicationArray, &nLayerCount, 0);

		for (layerCur = cln->layerHead, nLayer = 0; layerCur != NULL; layerCur = layerCur->next, ++nLayer)
		{
			layerCur->nAdditions = nAdditionArray[nLayer];
			layerCur->nMultiplications = nMultiplicationArray[nLayer];
		}

		DescribeClassLevelNetwork(cln, NULL);

	} while (cln->nWeightCount != nCurrentWeights);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void SIC_V4(structNetwork *network, structInput *inputTrain, structInput *inputTest, int nMaxCluster, float fTargetAccuracy)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structCLN *cln = network->clnHead;
	structLayer *layerCur = NULL;
	structPerceptron *perceptronCur = NULL;
	float fAccuracy;
	float fPercent = 0.0f;
	int bReduced;
	int nLoop = 0;
	int nLayer = 0;
	int nLayerCount = 0;
	int nTotalWeights = 0;
	int nPerceptron = 0;
	char sTimeBuffer[32];

	printf("***** SIC *****\n");

	for (layerCur = cln->layerHead; layerCur != NULL; layerCur = layerCur->next)
	{
		printf("%d\n", layerCur->nLayerType);

		if (layerCur->nLayerType == MAX_POOL_LAYER)
			continue;

		perceptronCur = layerCur->perceptronHead;

		while (perceptronCur != NULL)
		{
			++nPerceptron;

			// Copy Perceptron Weights
			CopyPerceptronWeights(perceptronCur, NETWORK_TO_MEMORY);

			for (perceptronCur->nClusterCount = 0, bReduced = 0; perceptronCur->nClusterCount < perceptronCur->nConnectionCount; ++perceptronCur->nClusterCount)
			{
				ClusterPerceptronWeights(perceptronCur);
				// InferCLN_Inference(cln, network->classHead, inputTrain, HIDE_DATA, network->nMatrix, network->fInputArray, 1, network->sDrive, network->sTitle, DO_NOT_MARK, NO_THRESHOLD, NULL, sTimeBuffer, BREAK_ON_BAD_CLASSIFICATION, nMaxCluster);
				InferCLN_Inference(cln, network->classHead, inputTrain, HIDE_DATA, network->nMatrix, network->fInputArray, 1, network->sDrive, network->sTitle, DO_NOT_MARK, NO_THRESHOLD, NULL, sTimeBuffer, 0, nMaxCluster);
				fAccuracy = ScoreClassLevelNetworkMatrix(cln->nNetworkType, 0, network->classHead, network->nMatrix, HIDE_MATRIX, NULL);

				if ((bReduced = (fAccuracy >= fTargetAccuracy)))
					break;

				// Reset Perceptron Weights
				CopyPerceptronWeights(perceptronCur, MEMORY_TO_NETWORK);
				SortByMissCount(inputTrain->data, inputTrain->nInputCount);
			}

			nTotalWeights += perceptronCur->nClusterCount;

			if (perceptronCur->nWeightCount != perceptronCur->nClusterCount)
			{
				if (perceptronCur->nWeightCount > 0)
					fPercent = ((float)(perceptronCur->nWeightCount - perceptronCur->nClusterCount)) / (float)perceptronCur->nWeightCount;
				else
					fPercent = 0.0f;

				printf("%d\t%0.4f\t%d                     \n", perceptronCur->nIndex, fPercent, nTotalWeights);
			}
			else if (perceptronCur->nWeightCount == 0)
				printf("***** Delete %d\n", perceptronCur->nIndex);

			if (layerCur->nLayerType == CONV_2D_LAYER || layerCur->nLayerType == CONV_3D_LAYER)
				perceptronCur = perceptronCur->nextHead;
			else
				perceptronCur = perceptronCur->next;
		}

		nPerceptron = 0;
	}

	printf("\n");

	ModifyArchitecture(cln);
	GetClassLevelNetworkWeightCount(cln);
	CreateMACArrayFromDNX_ClassLevelNetworks(cln);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void CopyPerceptronWeights(structPerceptron *perceptron, int nMode)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structSynapse *synapseCur;

	if (nMode == NETWORK_TO_MEMORY)
	{
		for (synapseCur = perceptron->synapseHead, perceptron->nConnectionCount = -1; synapseCur != NULL; synapseCur = synapseCur->next, ++perceptron->nConnectionCount)
			synapseCur->fTempWeight = *synapseCur->fWeight;
	}
	else
	{
		for (synapseCur = perceptron->synapseHead, perceptron->nConnectionCount = -1; synapseCur != NULL; synapseCur = synapseCur->next, ++perceptron->nConnectionCount)
			*synapseCur->fWeight = synapseCur->fTempWeight;
	}
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void ModifyArchitecture(structCLN *cln)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structLayer *layerNew = NULL;
	structLayer *layerCur = NULL;
	structPerceptron *perceptronCurHead = NULL;
	structPerceptron *perceptronCur = NULL;
	structSynapse *synapseCur = NULL;
	structSynapse *synapseConsolidate = NULL;
	int bRemoved;

	for (layerCur = cln->layerHead; layerCur != NULL; layerCur = layerCur->next)
	{
		if (layerCur->nLayerType == MAX_POOL_LAYER)
			continue;

		for (perceptronCurHead = layerCur->perceptronHead; perceptronCurHead != NULL; perceptronCurHead = perceptronCurHead->nextHead)
		{
			for (perceptronCur = perceptronCurHead; perceptronCur != NULL; perceptronCur = perceptronCur->next)
			{
				bRemoved = 0;

				for (synapseCur = perceptronCur->synapseHead; synapseCur != NULL; synapseCur = synapseCur->next)
				{
					if (synapseCur->nInputIndex == -99)
						continue;

					for (synapseConsolidate = synapseCur->next; synapseConsolidate != NULL; synapseConsolidate = synapseConsolidate->next)
					{
						if (*synapseCur->fWeight == *synapseConsolidate->fWeight)
						{
							bRemoved = 1;

							float **fInputArray;
							int *nIndexArray;
							int nInputCount;
							int i, j;

							nInputCount = synapseCur->nInputCount + synapseConsolidate->nInputCount + 1;

							if (nInputCount > 0)
							{
								nIndexArray = (int *)calloc(nInputCount, sizeof(int));
								fInputArray = (float **)calloc(nInputCount, sizeof(float *));

								for (i = 0, j = 0; i < synapseCur->nInputCount; ++i, ++j)
								{
									nIndexArray[j] = synapseCur->nInputArray[i];
									fInputArray[j] = synapseCur->fInputArray[i];
								}

								for (i = 0; i < synapseConsolidate->nInputCount; ++i, ++j)
								{
									nIndexArray[j] = synapseConsolidate->nInputArray[i];
									fInputArray[j] = synapseConsolidate->fInputArray[i];
								}

								if (synapseConsolidate->nInputArrayIndex == -999)
									nIndexArray[j] = synapseConsolidate->perceptronConnectTo->nID;
								else
									nIndexArray[j] = synapseConsolidate->nInputArrayIndex;

								fInputArray[j] = synapseConsolidate->fInput;

								free(synapseCur->nInputArray);
								free(synapseCur->fInputArray);

								synapseCur->nInputCount = ++j;
								synapseCur->nInputArray = nIndexArray;
								synapseCur->fInputArray = fInputArray;
							}

							// Mark for Removal
							synapseConsolidate->nInputIndex = -99;
						}
					}
				}

				while (bRemoved)
				{
					bRemoved = 0;

					for (synapseCur = perceptronCur->synapseHead; synapseCur != NULL; synapseCur = synapseCur->next)
					{
						if (synapseCur->nInputIndex == -99)
						{
							Delete_Synapse(&perceptronCur->synapseHead, synapseCur->nID);
							bRemoved = 1;

							break;
						}
					}
				}
			}
		}
	}
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void DeleteZeroWeights(structCLN *cln)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structLayer *layerCur = NULL;
	structPerceptron *perceptronCurHead = NULL;
	structPerceptron *perceptronCur = NULL;
	structSynapse *synapseCur = NULL;
	int bRemoved;

	for (layerCur = cln->layerHead; layerCur != NULL; layerCur = layerCur->next)
	{
		if (layerCur->nLayerType == MAX_POOL_LAYER)
			continue;

		for (perceptronCurHead = layerCur->perceptronHead; perceptronCurHead != NULL; perceptronCurHead = perceptronCurHead->nextHead)
		{
			for (perceptronCur = perceptronCurHead; perceptronCur != NULL; perceptronCur = perceptronCur->next)
			{
				bRemoved = 1;

				while (bRemoved)
				{
					bRemoved = 0;

					for (synapseCur = perceptronCur->synapseHead; synapseCur != NULL; synapseCur = synapseCur->next)
					{
						printf("Printing Synapse Current Weight: %lf\n", *synapseCur->fWeight);
						if (*synapseCur->fWeight == 0.0f)
						{
							Delete_Synapse(&perceptronCur->synapseHead, synapseCur->nID);
							bRemoved = 1;
							break;
						}
					}
				}
			}
		}
	}
}
