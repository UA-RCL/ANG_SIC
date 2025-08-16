#include "main.h"

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
int ParseLine_Config(char *sBuffer, structConfigParameters *parameterData)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	int		nLength=(int)strlen(sBuffer);
	int		nReturn = 0;
	int		i, j;

	for (i=0, j=0; i<nLength; ++i)
	{
		if (sBuffer[i] == ';')
		{
			parameterData->sParameter[j++] = '\0';

			for (++i, j = 0; i<nLength; ++i)
			{
				if (sBuffer[i] != '\t' && sBuffer[i] != '\r' && sBuffer[i] != '\n')
				{
					parameterData->sValue[j++] = sBuffer[i];
					nReturn = 1;
				}
			}
		}
		else
		{
			parameterData->sParameter[j++] = sBuffer[i];
		}
	}

	parameterData->sValue[j++] = '\0';

	return(nReturn);
}


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void ReadFile_Config(structNetwork *networkMain, char *sFilePath)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	FILE					*pFile;
	structConfigParameters	*parameterData;
	char					sLayerType[32];
	char					sTemp[256];
	char					sBuffer[1024];
	int						*nLayerTypeArray;
	int						*nLayerMapCountArray;
	int						*nLayerRowCountArray;
	int						*nLayerColumnCountArray;
	int						*nLayerRowStrideArray;
	int						*nLayerColumnStrideArray;
	int						*nLayerNeuronsArray;
	int						nLayerCount;
	int						nCount;
	int						i;

	nLayerTypeArray = (int *)calloc(100, sizeof(int));
	nLayerMapCountArray = (int *)calloc(100, sizeof(int));
	nLayerRowCountArray = (int *)calloc(100, sizeof(int));
	nLayerColumnCountArray = (int *)calloc(100, sizeof(int));
	nLayerRowStrideArray = (int *)calloc(100, sizeof(int));
	nLayerColumnStrideArray = (int *)calloc(100, sizeof(int));
	nLayerNeuronsArray = (int *)calloc(100, sizeof(int));
	nLayerCount = 0;

	nCount = -1;
	if ((parameterData=(structConfigParameters *)calloc(100, sizeof(structConfigParameters))) == NULL)
	{
		HoldDisplay("Memory Error: ReadFile_Config()\n");
	}


	if ((pFile = FOpenMakeDirectory(sFilePath, "rb")) == NULL)
	{
		printf("Error ReadFile_Config(): Could not find the config file -- %s\n\n", sFilePath);
		while (1);
	}

	fgets(sBuffer, 1024, pFile);
	ParseLine_Config(sBuffer, &parameterData[++nCount]);

	if (!STRICMP(parameterData[nCount].sParameter, "MODE"))
	{
		if (!STRICMP(parameterData[nCount].sValue, "BUILD_COMPLETE_NETWORK"))
			networkMain->nOpMode = BUILD_COMPLETE_NETWORK;
		else if (!STRICMP(parameterData[nCount].sValue, "REBUILD_COMPLETE_NETWORK"))
			networkMain->nOpMode = REBUILD_COMPLETE_NETWORK;
		else if (!STRICMP(parameterData[nCount].sValue, "REBUILD_CLASS_LEVEL_NETWORK"))
			networkMain->nOpMode = REBUILD_CLASS_LEVEL_NETWORK;
		else if (!STRICMP(parameterData[nCount].sValue, "BUILD_CLASS_LEVEL_NETWORK"))
			networkMain->nOpMode = BUILD_CLASS_LEVEL_NETWORK;
		else if (!STRICMP(parameterData[nCount].sValue, "NEUROGENESIS"))
			networkMain->nOpMode = NEUROGENESIS;
		else if (!STRICMP(parameterData[nCount].sValue, "REBUILD_NEUROGENESIS"))
			networkMain->nOpMode = REBUILD_NEUROGENESIS;
		else if (!STRICMP(parameterData[nCount].sValue, "ANALYZE_INPUT"))
			networkMain->nOpMode = ANALYZE_INPUT;
		else
			HoldDisplay("Mode Error\n");

		while (!feof(pFile))
		{
			fgets(sBuffer, 1024, pFile);

			if (strlen(sBuffer) == 0)
			{
				++nCount;
				break;
			}

			ParseLine_Config(sBuffer, &parameterData[++nCount]);

			if (!STRICMP(parameterData[nCount].sParameter, "DATA TYPE"))
			{
				sprintf(networkMain->sDataSource, "%s", parameterData[nCount].sValue);
			}
			else if (!STRICMP(parameterData[nCount].sParameter, "DATA DIR"))
			{
				sprintf(networkMain->sDrive, "%s", parameterData[nCount].sValue);
			}
			else if (!STRICMP(parameterData[nCount].sParameter, "TRAIN DATA"))
			{
				if (!STRICMP(networkMain->sDataSource, "ISAR"))
					sprintf(networkMain->sTrainingFilePath, "%s\\%s", networkMain->sDrive, parameterData[nCount].sValue);
				else
					sprintf(networkMain->sTrainingFilePath, "%s", parameterData[nCount].sValue);
			}
			else if (!STRICMP(parameterData[nCount].sParameter, "TEST DATA"))
			{
				sprintf(networkMain->sTestingFilePath, "%s", parameterData[nCount].sValue);
			}
			else if (!STRICMP(parameterData[nCount].sParameter, "TRAIN-VERIFY SPLIT"))
			{
				networkMain->nTrainVerifySplit=atoi(parameterData[nCount].sValue);
			}
			else if (!STRICMP(parameterData[nCount].sParameter, "NEUROGENESIS UPPER SIGMA"))
			{
				networkMain->fMaximumThreshold=(float)atof(parameterData[nCount].sValue);
			}
			else if (!STRICMP(parameterData[nCount].sParameter, "NEUROGENESIS LOWER SIGMA"))
			{
				networkMain->fMinimumThreshold = (float)atof(parameterData[nCount].sValue);
			}
			else if (!STRICMP(parameterData[nCount].sParameter, "GLOBAL LEARNING RATE"))
			{
				networkMain->fLearningRate = (float)atof(parameterData[nCount].sValue);
			}
			else if (!STRICMP(parameterData[nCount].sParameter, "INITIAL ERROR"))
			{
				networkMain->fInitialError = (float)atof(parameterData[nCount].sValue);
			}
			else if (!STRICMP(parameterData[nCount].sParameter, "THRESHOLD"))
			{
				networkMain->fThreshold = (float)atof(parameterData[nCount].sValue);
			}
			else if (!STRICMP(parameterData[nCount].sParameter, "BACKPROP THRESHOLD PERCENT"))
			{
				networkMain->fThresholdPercent = (float)atof(parameterData[nCount].sValue);
			}
			else if (!STRICMP(parameterData[nCount].sParameter, "TARGET LEARNING RATE"))
			{
				networkMain->fTargetWeight = (float)atof(parameterData[nCount].sValue);
			}
			else if (!STRICMP(parameterData[nCount].sParameter, "PRIMING CYCLES"))
			{
				networkMain->nPrimingCycles = atoi(parameterData[nCount].sValue);
			}
			else if (!STRICMP(parameterData[nCount].sParameter, "TRAINING CYCLES"))
			{
				networkMain->nTrainingCycles = atoi(parameterData[nCount].sValue);
			}
			else if (!STRICMP(parameterData[nCount].sParameter, "CONVOLUTION SWEEP"))
			{
				sscanf(parameterData[nCount].sValue, "%d,%d,%d,%d,%d,%d,%d,%d\n", &networkMain->nKernelCountStart, &networkMain->nKernelCountEnd, &networkMain->nSubNetCount, &networkMain->nBuildSort, &networkMain->nBuildThreshold, &networkMain->nTrainSort, &networkMain->nTrainThreshold, &networkMain->nPostTrain);
			}
			else if (!STRICMP(parameterData[nCount].sParameter, "ADJ GLOBAL LEARNING RATE"))
			{
				networkMain->bAdjustGlobalLearningRate = atoi(parameterData[nCount].sValue);
			}
			else if (!STRICMP(parameterData[nCount].sParameter, "LEARNING RATE MINIMUM"))
			{
				networkMain->fLearningRateMinimum = (float)atof(parameterData[nCount].sValue);
			}
			else if (!STRICMP(parameterData[nCount].sParameter, "LEARNING RATE MAXIMUM"))
			{
				networkMain->fLearningRateMaximum = (float)atof(parameterData[nCount].sValue);
			}
			else if (!STRICMP(parameterData[nCount].sParameter, "ADJ PERCEPTRON LEARNING RATES"))
			{
				networkMain->bAdjustPerceptronLearningRate = atoi(parameterData[nCount].sValue);
			}
			else if (!STRICMP(parameterData[nCount].sParameter, "ADJ PERCEPTRON THRESHOLDS"))
			{
				networkMain->bAdjustThreshold = atoi(parameterData[nCount].sValue);
			}
			else if (!STRICMP(parameterData[nCount].sParameter, "LEARNING RATE INITAILIZATION"))
			{
				networkMain->bLearnRateInitialization = atoi(parameterData[nCount].sValue);
			}
			else if (!STRICMP(parameterData[nCount].sParameter, "TRAINING RESORT"))
			{
				networkMain->nTrainSort = atoi(parameterData[nCount].sValue);
			}
			else if (!STRICMP(parameterData[nCount].sParameter, "NO PROGRESS RESORT COUNT"))
			{
				networkMain->nNoProgressResortCount = atoi(parameterData[nCount].sValue);
			}
			else if (!STRICMP(parameterData[nCount].sParameter, "TRAINING RESPLIT"))
			{
				networkMain->nTrainResplit = atoi(parameterData[nCount].sValue);
			}
			else if (!STRICMP(parameterData[nCount].sParameter, "NO PROGRESS RESPLIT COUNT"))
			{
				networkMain->nNoProgressResplitCount = atoi(parameterData[nCount].sValue);
			}
			else if (!STRICMP(parameterData[nCount].sParameter, "NETWORK FILE PATH"))
			{
				strcpy(networkMain->sNetworkFilePath, parameterData[nCount].sValue);
			}
			else if (!STRICMP(parameterData[nCount].sParameter, "INPUT STRUCTURE PATH"))
			{
				sprintf(networkMain->sInputStructurePath, parameterData[nCount].sValue);
			}
			else if (!STRICMP(parameterData[nCount].sParameter, "OUTPUT STRUCTURE PATH"))
			{
				sprintf(networkMain->sOutputStructurePath, parameterData[nCount].sValue);
			}
			else if (!STRICMP(parameterData[nCount].sParameter, "DNA OUTPUT PATH"))
			{
				sprintf(networkMain->sDNAOutputPath, parameterData[nCount].sValue);
			}
			else if (!STRICMP(parameterData[nCount].sParameter, "DNX OUTPUT PATH"))
			{
				sprintf(networkMain->sDNXOutputPath, parameterData[nCount].sValue);
			}
			else if (!STRICMP(parameterData[nCount].sParameter, "SIC"))
				networkMain->nSIC = 1;
			else if (!STRICMP(parameterData[nCount].sParameter, "SIC_PRUNE"))
				networkMain->nSIC = 2;
			else if (!STRICMP(parameterData[nCount].sParameter, "PRUNE_SIC"))
				networkMain->nSIC = 3;
			else if (!STRICMP(parameterData[nCount].sParameter, "PRUNE"))
				networkMain->nSIC = 4;
			
			else if (!STRICMP(parameterData[nCount].sParameter, "PRUNE MODE"))
			{
				if (!STRICMP(parameterData[nCount].sValue, "AFTER"))
					networkMain->nPruneNetwork = PRUNE_AFTER;
				else if (!STRICMP(parameterData[nCount].sValue, "PRUNE EACH CYCLE"))
					networkMain->nPruneNetwork = PRUNE_EACH_CYCLE;
				else if (!STRICMP(parameterData[nCount].sValue, "PRUNE INTERVAL CYCLE"))
					networkMain->nPruneNetwork = PRUNE_INTERVAL_CYCLE;
				else if (!STRICMP(parameterData[nCount].sValue, "PRUNE AFTER ZERO"))
					networkMain->nPruneNetwork = PRUNE_AFTER_ZERO;
				else
					networkMain->nPruneNetwork = 0;
			}
			else if (!STRICMP(parameterData[nCount].sParameter, "PRUNE FC THRESHOLD"))
			{
				networkMain->fPruneFCThreshold = (float)atof(parameterData[nCount].sValue);
			}
			else if (!STRICMP(parameterData[nCount].sParameter, "PRUNE CONV THRESHOLD"))
			{
				networkMain->fPruneConvThreshold = (float)atof(parameterData[nCount].sValue);
			}
			else if (!STRICMP(parameterData[nCount].sParameter, "PRUNE INTERVAL"))
			{
				networkMain->nPruneInterval = atoi(parameterData[nCount].sValue);
			}
			else if (!STRICMP(parameterData[nCount].sParameter, "NO PROGRESS STOP COUNT"))
			{
				networkMain->nNoProgressCount = atoi(parameterData[nCount].sValue);
			}
			else if (!STRICMP(parameterData[nCount].sParameter, "EXECUTE TRAIN INFERENCE"))
			{
				networkMain->nTrainInferenceExecute = atoi(parameterData[nCount].sValue);
			}
			else if (!STRICMP(parameterData[nCount].sParameter, "EXECUTE VALIDATE INFERENCE"))
			{
				networkMain->nValidateInferenceExecute = atoi(parameterData[nCount].sValue);
			}
			else if (!STRICMP(parameterData[nCount].sParameter, "EXECUTE TEST INFERENCE"))
			{
				networkMain->nTestInferenceExecute = atoi(parameterData[nCount].sValue);
			}
			else if (!STRICMP(parameterData[nCount].sParameter, "LAYER TYPE"))
			{
				sprintf(sLayerType, "%s", parameterData[nCount].sValue);

				if (feof(pFile))
					break;

				if (!strcmp(sLayerType, "INPUT_LAYER"))
				{
					nLayerTypeArray[nLayerCount] = INPUT_LAYER;
					
					fgets(sBuffer, 1024, pFile);
					ParseLine_Config(sBuffer, &parameterData[++nCount]);
					nLayerRowCountArray[nLayerCount] = atoi(parameterData[nCount].sValue);

					fgets(sBuffer, 1024, pFile);
					ParseLine_Config(sBuffer, &parameterData[++nCount]);
					nLayerColumnCountArray[nLayerCount] = atoi(parameterData[nCount].sValue);
				}
				else if (!strcmp(sLayerType, "SINGLE_CONV_LAYER"))
				{
					nLayerTypeArray[nLayerCount] = SINGLE_CONV_LAYER;
					
					fgets(sBuffer, 1024, pFile);
					ParseLine_Config(sBuffer, &parameterData[++nCount]);
					nLayerMapCountArray[nLayerCount] = atoi(parameterData[nCount].sValue);

					fgets(sBuffer, 1024, pFile);
					ParseLine_Config(sBuffer, &parameterData[++nCount]);
					nLayerRowCountArray[nLayerCount] = atoi(parameterData[nCount].sValue);

					fgets(sBuffer, 1024, pFile);
					ParseLine_Config(sBuffer, &parameterData[++nCount]);
					nLayerColumnCountArray[nLayerCount] = atoi(parameterData[nCount].sValue);

					fgets(sBuffer, 1024, pFile);
					ParseLine_Config(sBuffer, &parameterData[++nCount]);
					nLayerRowStrideArray[nLayerCount] = atoi(parameterData[nCount].sValue);

					fgets(sBuffer, 1024, pFile);
					ParseLine_Config(sBuffer, &parameterData[++nCount]);
					nLayerColumnStrideArray[nLayerCount] = atoi(parameterData[nCount].sValue);
				}
				else if (!strcmp(sLayerType, "MAX_POOLING_LAYER"))
				{
					nLayerTypeArray[nLayerCount] = MAX_POOLING_LAYER;
					
					fgets(sBuffer, 1024, pFile);
					ParseLine_Config(sBuffer, &parameterData[++nCount]);
					nLayerRowCountArray[nLayerCount] = atoi(parameterData[nCount].sValue);

					fgets(sBuffer, 1024, pFile);
					ParseLine_Config(sBuffer, &parameterData[++nCount]);
					nLayerColumnCountArray[nLayerCount] = atoi(parameterData[nCount].sValue);

					fgets(sBuffer, 1024, pFile);
					ParseLine_Config(sBuffer, &parameterData[++nCount]);
					nLayerRowStrideArray[nLayerCount] = atoi(parameterData[nCount].sValue);

					fgets(sBuffer, 1024, pFile);
					ParseLine_Config(sBuffer, &parameterData[++nCount]);
					nLayerColumnStrideArray[nLayerCount] = atoi(parameterData[nCount].sValue);
				}
				else if (!strcmp(sLayerType, "MULTIPLE_CONV_LAYER"))
				{
					nLayerTypeArray[nLayerCount] = MULTIPLE_CONV_LAYER;

					fgets(sBuffer, 1024, pFile);
					ParseLine_Config(sBuffer, &parameterData[++nCount]);
					nLayerMapCountArray[nLayerCount] = atoi(parameterData[nCount].sValue);

					fgets(sBuffer, 1024, pFile);
					ParseLine_Config(sBuffer, &parameterData[++nCount]);
					nLayerRowCountArray[nLayerCount] = atoi(parameterData[nCount].sValue);

					fgets(sBuffer, 1024, pFile);
					ParseLine_Config(sBuffer, &parameterData[++nCount]);
					nLayerColumnCountArray[nLayerCount] = atoi(parameterData[nCount].sValue);

					fgets(sBuffer, 1024, pFile);
					ParseLine_Config(sBuffer, &parameterData[++nCount]);
					nLayerRowStrideArray[nLayerCount] = atoi(parameterData[nCount].sValue);

					fgets(sBuffer, 1024, pFile);
					ParseLine_Config(sBuffer, &parameterData[++nCount]);
					nLayerColumnStrideArray[nLayerCount] = atoi(parameterData[nCount].sValue);
				}
				else if (!strcmp(sLayerType, "FULLY_CONNECTED_LAYER"))
				{
					nLayerTypeArray[nLayerCount] = FULLY_CONNECTED_LAYER;

					fgets(sBuffer, 1024, pFile);
					ParseLine_Config(sBuffer, &parameterData[++nCount]);
					nLayerNeuronsArray[nLayerCount] = atoi(parameterData[nCount].sValue);
				}
				else if (!strcmp(sLayerType, "CLASSIFIER_LAYER"))
				{
					nLayerTypeArray[nLayerCount] = CLASSIFIER_LAYER;

					fgets(sBuffer, 1024, pFile);
					ParseLine_Config(sBuffer, &parameterData[++nCount]);
					nLayerNeuronsArray[nLayerCount] = atoi(parameterData[nCount].sValue);
				}
				else
				{
					HoldDisplay("Layer Description Error\n");
				}

				++nLayerCount;
			}
			else
			{
				sprintf(sBuffer, "Config Description Error: %s\n", parameterData[nCount].sParameter);
				HoldDisplay(sBuffer);
			}

			sBuffer[0] = '\0';
		}

		


		if ((networkMain->parameterData = (structConfigParameters *)calloc(nCount, sizeof(structConfigParameters))) == NULL)
		{
			HoldDisplay("Memory Error: ReadFile_Config()\n");
		}

		for (networkMain->nParameterCount=0; networkMain->nParameterCount<nCount; ++networkMain->nParameterCount)
		{
			strcpy(networkMain->parameterData[networkMain->nParameterCount].sParameter, parameterData[networkMain->nParameterCount].sParameter);
			strcpy(networkMain->parameterData[networkMain->nParameterCount].sValue, parameterData[networkMain->nParameterCount].sValue);
		}

		free(parameterData);
	}
	else
	{
		fseek(pFile, SEEK_SET, 0);


		fscanf(pFile, "%s\n", networkMain->sDataSource);
		fscanf(pFile, "%s\n", networkMain->sDrive);

		if (!strcmp(networkMain->sDataSource, "ISAR"))
		{
			fscanf(pFile, "%s\n", sTemp);
			sprintf(networkMain->sTrainingFilePath, "%s\\%s", networkMain->sDrive, sTemp);
		}
		else
		{
			fscanf(pFile, "%s\n", networkMain->sTrainingFilePath);
			fscanf(pFile, "%s\n", networkMain->sTestingFilePath);
		}

		fscanf(pFile, "%d\n", &networkMain->nTrainVerifySplit);
		fscanf(pFile, "%f\n", &networkMain->fMaximumThreshold);
		fscanf(pFile, "%f\n", &networkMain->fMinimumThreshold);
		fscanf(pFile, "%f\n", &networkMain->fLearningRate);
		fscanf(pFile, "%f\n", &networkMain->fInitialError);
		fscanf(pFile, "%f\n", &networkMain->fThreshold);
		fscanf(pFile, "%f\n", &networkMain->fTargetWeight);
		fscanf(pFile, "%d\n", &networkMain->nPrimingCycles);
		fscanf(pFile, "%d\n", &networkMain->nTrainingCycles);

		fscanf(pFile, "%d,%d,%d,%d,%d,%d,%d,%d\n", &networkMain->nKernelCountStart, &networkMain->nKernelCountEnd, &networkMain->nSubNetCount, &networkMain->nBuildSort, &networkMain->nBuildThreshold, &networkMain->nTrainSort, &networkMain->nTrainThreshold, &networkMain->nPostTrain);
		fscanf(pFile, "%d,%f,%f,%d,%d\n", &networkMain->bAdjustGlobalLearningRate, &networkMain->fLearningRateMinimum, &networkMain->fLearningRateMaximum, &networkMain->bAdjustPerceptronLearningRate, &networkMain->bAdjustThreshold);

		nLayerCount = 0;
		while (!feof(pFile))
		{
			fscanf(pFile, "%s\n", sLayerType);

			if (!strcmp(sLayerType, "INPUT_LAYER"))
			{
				nLayerTypeArray[nLayerCount] = INPUT_LAYER;
				fscanf(pFile, "%d %d\n", &nLayerRowCountArray[nLayerCount], &nLayerColumnCountArray[nLayerCount]);
			}
			else if (!strcmp(sLayerType, "SINGLE_CONV_LAYER"))
			{
				nLayerTypeArray[nLayerCount] = SINGLE_CONV_LAYER;
				fscanf(pFile, "%d %d %d %d %d\n", &nLayerMapCountArray[nLayerCount], &nLayerRowCountArray[nLayerCount], &nLayerColumnCountArray[nLayerCount], &nLayerRowStrideArray[nLayerCount], &nLayerColumnStrideArray[nLayerCount]);
			}
			else if (!strcmp(sLayerType, "MULTIPLE_CONV_LAYER"))
			{
				nLayerTypeArray[nLayerCount] = MULTIPLE_CONV_LAYER;
				fscanf(pFile, "%d %d %d %d %d\n", &nLayerMapCountArray[nLayerCount], &nLayerRowCountArray[nLayerCount], &nLayerColumnCountArray[nLayerCount], &nLayerRowStrideArray[nLayerCount], &nLayerColumnStrideArray[nLayerCount]);
			}
			else if (!strcmp(sLayerType, "FULLY_CONNECTED_LAYER"))
			{
				nLayerTypeArray[nLayerCount] = FULLY_CONNECTED_LAYER;
				fscanf(pFile, "%d\n", &nLayerNeuronsArray[nLayerCount]);
			}
			else if (!strcmp(sLayerType, "CLASSIFIER_LAYER"))
			{
				nLayerTypeArray[nLayerCount] = CLASSIFIER_LAYER;
			}
			else
			{
				HoldDisplay("Layer Description Error\n");
			}

			++nLayerCount;
		}
	}

	fclose(pFile);

	networkMain->nLayerCount = nLayerCount;

	networkMain->architecture = (structArchitecture *)calloc(networkMain->nLayerCount, sizeof(structArchitecture));
	for (i=0; i<networkMain->nLayerCount; ++i)
	{
		networkMain->architecture[i].nID=i;
		networkMain->architecture[i].nLayerType = nLayerTypeArray[i];
		
		if (networkMain->architecture[i].nLayerType == FULLY_CONNECTED_LAYER || networkMain->architecture[i].nLayerType == CLASSIFIER_LAYER)
		{
			networkMain->architecture[i].nKernelCount = nLayerNeuronsArray[i];
		}
		else
		{
			networkMain->architecture[i].nKernelCount = nLayerMapCountArray[i];
			networkMain->architecture[i].nRowKernelSize = nLayerRowCountArray[i];
			networkMain->architecture[i].nColumnKernelSize = nLayerColumnCountArray[i];
			networkMain->architecture[i].nStrideRow = nLayerRowStrideArray[i];
			networkMain->architecture[i].nStrideColumn = nLayerColumnStrideArray[i];
		}
	}

	free(nLayerTypeArray);
	free(nLayerMapCountArray);
	free(nLayerRowCountArray);
	free(nLayerColumnCountArray);
	free(nLayerRowStrideArray);
	free(nLayerColumnStrideArray);
	free(nLayerNeuronsArray);

}

