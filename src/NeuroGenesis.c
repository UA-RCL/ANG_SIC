
#include "main.h"

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
structLayer *NeuroGenesis(structNetwork *networkMain, structInput *inputData, structInput *inputTestingData, structInput **inputTrainingData, structInput **inputVerifyData, float fLearningRate, float fThreshold, float fMinimumThreshold, float fMaximumThreshold, structMAC **macData, int nMACCount, int nBuildMode, float *fTestAccuracy, FILE *fpFileOut)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structCLN			*clnCur = NULL;
	structLayer			*layerNew=NULL;
	structLayer			*layerCur;
	structClass			*classCur = NULL;
	structPerceptron	*perceptronCur = NULL;
	structPerceptron	*perceptronCurHead = NULL;
	structSynapse		*synapseCur = NULL;
	float				fTrainAccuracy;
	float				fValidateAccuracy;
	float				fTargetAccuracy=1.0f;
	int					nPerceptronCount = 0;
	char				sTimeBuffer[32];
	char				sFilePath[256];
	int					nCurrentLayerType;
	int					i;

	if (!nBuildMode)
	{
		// Create Seed Network
		
		clnCur=CreateSeedNetwork_ConstructNetworks(networkMain);

		CalculateOutputSize(networkMain->clnHead);
		GetClassLevelNetworkWeightCount(networkMain->clnHead);

		DescribeClassLevelNetwork(clnCur, NULL);

		// Prime Network
		PrimeSeedNetwork_ConstructNetworks(networkMain, clnCur, networkMain->classHead, inputData, inputTrainingData, inputVerifyData, inputTestingData, networkMain->fInputArray, networkMain->nPrimingCycles, networkMain->fTargetWeight, RANDOMIZE, CALCULATE_LEARNING_RATE, networkMain->fLearningRate, HIDE_MATRIX, NULL, 1); // FIXED_LEARNING_RATE


		int nLoop = 1;
		int nSIC = 0;
		int nLayer = 1;
		int nPrevWeightCount = 1;
		int nLayerCount = 1;
		int* nAdditionArray = NULL;
		int* nMultiplicationArray = NULL;

		MarkInputData(networkMain, *inputTrainingData, *inputVerifyData, inputTestingData);
		fTargetAccuracy = networkMain->clnHead->fTrainAccuracy;
		
		do
		{
			nPrevWeightCount = networkMain->clnHead->nWeightCount;
			
			GroupByDifference((*inputTrainingData)->data, (*inputTrainingData)->nInputCount, 10);
			SortByDifference((*inputTrainingData)->data, (*inputTrainingData)->nInputCount);

			if (nSIC == 1 || (nSIC == 2 && nLoop == 1))
				SIC_V4(networkMain, *inputTrainingData, inputTestingData, 1, fTargetAccuracy);

			if (nSIC == 2 || (nSIC == 3 && nLoop == 1) || nSIC == 4)
				PrunePerceptronWeightsSinglePass(networkMain, *inputTrainingData, inputTestingData, 1);

			if (nSIC == 3)
				SIC_V4(networkMain, *inputTrainingData, inputTestingData, 1, fTargetAccuracy);

			ForwardPropagateAnalyze_Train(networkMain->clnHead->macData, networkMain->clnHead->nMACCount, &nAdditionArray, &nMultiplicationArray, &nLayerCount, 0);

			for (layerCur = networkMain->clnHead->layerHead, nLayer = 0; layerCur != NULL; layerCur = layerCur->next, ++nLayer)
			{
				layerCur->nAdditions = nAdditionArray[nLayer];
				layerCur->nMultiplications = nMultiplicationArray[nLayer];
			}

			DescribeClassLevelNetwork(networkMain->clnHead, NULL);

			nLoop = 1;
		} while (networkMain->clnHead->nWeightCount != nPrevWeightCount);


		InferCLN_Inference(clnCur, networkMain->classHead, *inputTrainingData, HIDE_DATA, networkMain->nMatrix, networkMain->fInputArray, 1, networkMain->sDrive, networkMain->sTitle, DO_NOT_MARK, NO_THRESHOLD, fpFileOut, sTimeBuffer, 0, -1);
		fTrainAccuracy = ScoreClassLevelNetworkMatrix(clnCur->nNetworkType, 0, networkMain->classHead, networkMain->nMatrix, HIDE_MATRIX, fpFileOut);
		printf("ANG_Prime_Train: %f\t%s\n", fTrainAccuracy, sTimeBuffer);

		InferCLN_Inference(clnCur, networkMain->classHead, *inputVerifyData, HIDE_DATA, networkMain->nMatrix, networkMain->fInputArray, 1, networkMain->sDrive, networkMain->sTitle, DO_NOT_MARK, NO_THRESHOLD, fpFileOut, sTimeBuffer, 0, -1);
		fValidateAccuracy = ScoreClassLevelNetworkMatrix(clnCur->nNetworkType, 0, networkMain->classHead, networkMain->nMatrix, HIDE_MATRIX, fpFileOut);
		printf("ANG_Prime_Validate: %f\t%s\n", fValidateAccuracy, sTimeBuffer);

		InferCLN_Inference(clnCur, networkMain->classHead, inputTestingData, HIDE_DATA, networkMain->nMatrix, networkMain->fInputArray, 1, networkMain->sDrive, networkMain->sTitle, DO_NOT_MARK, NO_THRESHOLD, fpFileOut, sTimeBuffer, 0, -1);
		*fTestAccuracy = ScoreClassLevelNetworkMatrix(clnCur->nNetworkType, 0, networkMain->classHead, networkMain->nMatrix, HIDE_MATRIX, fpFileOut);
		printf("ANG_Prime_Test: %f\t%s\n\n", *fTestAccuracy, sTimeBuffer);

		sprintf(sFilePath, "seed.dna");
		SaveNetwork(networkMain, sFilePath);
	}
	else
	{
		RebuildCompleteNetwork_ConstructNetworks(networkMain, fpFileOut, SHOW_DATA, *inputTrainingData);
		clnCur = networkMain->clnHead;

		// Find Layer Boundries
		for (layerCur = clnCur->layerHead, clnCur->nLayerCount = 0; layerCur != NULL; layerCur = layerCur->next, ++clnCur->nLayerCount);

		if ((clnCur->nStartArray = (int*)calloc(clnCur->nLayerCount, sizeof(int))) == NULL)
			exit(0);

		if ((clnCur->nEndArray = (int*)calloc(clnCur->nLayerCount, sizeof(int))) == NULL)
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


		clnCur->nNetworkType = COMPLETE_NETWORK;
		GetClassLevelNetworkWeightCount(clnCur);
		CreateMACArray_ClassLevelNetworks(clnCur);
	}
	
	// Delete Classifier
	networkMain->layerInput = Delete_Layer(&clnCur->layerHead, clnCur->perceptronClassifier->nLayerID);

	// Create MAC Array
	GetClassLevelNetworkWeightCount(clnCur);
	CreateMACArray_ClassLevelNetworks(clnCur);

	// Calculate the average output of each perceptron in the last layer
	GetLastLayerOutputAverages_Neurogenesis(clnCur, *inputTrainingData, networkMain->classHead, networkMain->layerInput, networkMain->fInputArray);

	// Create a new layer to receive the new perceptrons
	layerNew=Create_Layer(&clnCur->layerHead, FULLY_CONNECTED_LAYER, 0, 0, networkMain->nClassCount * 2);

	// Add perceptrons to new layer
	GrowNewLayer_Neurogenesis(networkMain, clnCur, layerNew, *inputTrainingData, &nPerceptronCount, fLearningRate, fThreshold, fMinimumThreshold, fMaximumThreshold);

	// Make selected connections
	CreateConnectionLayer_ClassLevelNetworks(clnCur, (*inputTrainingData)->nClassCount, CLASSIFIER_LAYER, 0.001f, RANDOMIZE, networkMain->nRowCount, networkMain->nColumnCount, &networkMain->nPerceptronID, &networkMain->nSynapseID, networkMain->fInputArray);

	DescribeClassLevelNetwork(clnCur, NULL);
	if (fpFileOut != NULL)
		DescribeClassLevelNetwork(clnCur, fpFileOut);

	// Create MAC Array	GetClassLevelNetworkWeightCount(clnCur);
	GetClassLevelNetworkWeightCount(clnCur);

	CreateMACArray_ClassLevelNetworks(clnCur);

	for (perceptronCur = layerNew->perceptronHead; perceptronCur != NULL; perceptronCur = perceptronCur->next)
		perceptronCur->nLayerID = layerNew->nID;

	//Train Network
	//*fTestAccuracy = PrimeSeedNetwork_ConstructNetworks(networkMain, clnCur, networkMain->classHead, inputData, inputTrainingData, inputVerifyData, inputTestingData, networkMain->fInputArray, networkMain->nTrainingCycles, networkMain->fTargetWeight, 0, FIXED_LEARNING_RATE, networkMain->fLearningRate, SHOW_MATRIX, fpFileOut, 0);
	

////////////////////////////////////

		// Find Layer Boundries
	for (layerCur = clnCur->layerHead, clnCur->nLayerCount = 0; layerCur != NULL; layerCur = layerCur->next, ++clnCur->nLayerCount);

	if ((clnCur->nStartArray = (int*)calloc(clnCur->nLayerCount, sizeof(int))) == NULL)
		exit(0);

	if ((clnCur->nEndArray = (int*)calloc(clnCur->nLayerCount, sizeof(int))) == NULL)
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



////////////////////////////////////






	Network_Train(networkMain, clnCur, inputTrainingData, inputVerifyData, inputTestingData, inputData, networkMain->nTrainingCycles, fpFileOut, EXECUTE_TEST_INFERENCE);


	InferCLN_Inference(clnCur, networkMain->classHead, *inputTrainingData, HIDE_DATA, networkMain->nMatrix, networkMain->fInputArray, 1, networkMain->sDrive, networkMain->sTitle, DO_NOT_MARK, NO_THRESHOLD, fpFileOut, sTimeBuffer, 0, -1);
	fTrainAccuracy = ScoreClassLevelNetworkMatrix(clnCur->nNetworkType, 0, networkMain->classHead, networkMain->nMatrix, HIDE_MATRIX, fpFileOut);
	printf("ANG_Train: %f\t%s\n", fTrainAccuracy, sTimeBuffer);

	InferCLN_Inference(clnCur, networkMain->classHead, *inputVerifyData, HIDE_DATA, networkMain->nMatrix, networkMain->fInputArray, 1, networkMain->sDrive, networkMain->sTitle, DO_NOT_MARK, NO_THRESHOLD, fpFileOut, sTimeBuffer, 0, -1);
	fValidateAccuracy = ScoreClassLevelNetworkMatrix(clnCur->nNetworkType, 0, networkMain->classHead, networkMain->nMatrix, HIDE_MATRIX, fpFileOut);
	printf("ANG_Validate: %f\t%s\n", fValidateAccuracy, sTimeBuffer);

	InferCLN_Inference(clnCur, networkMain->classHead, inputTestingData, HIDE_DATA, networkMain->nMatrix, networkMain->fInputArray, 1, networkMain->sDrive, networkMain->sTitle, DO_NOT_MARK, NO_THRESHOLD, fpFileOut, sTimeBuffer, 0, -1);
	*fTestAccuracy = ScoreClassLevelNetworkMatrix(clnCur->nNetworkType, 0, networkMain->classHead, networkMain->nMatrix, HIDE_MATRIX, fpFileOut);
	printf("ANG_Test: %f\t%s\n\n", *fTestAccuracy, sTimeBuffer);

	return(layerNew);
}



/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void GetLastLayerOutputAverages_Neurogenesis(structCLN *cln, structInput *inputData, structClass *classHead, structLayer *layerInput, float *fInputArray)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structClass			*classCur;
	structPerceptron	*perceptronHeadCur;
	structPerceptron	*perceptronCur;
	int					nInputIndex;
	int					i;
	
	
	cln->nClassCount = inputData->nClassCount;
	cln->fOutputArray = (float **)calloc(cln->nClassCount, sizeof(float *));


	for (classCur = classHead; classCur != NULL; classCur = classCur->next)
	{
		cln->fOutputArray[classCur->nID] = (float *)calloc(layerInput->nOutputArraySize, sizeof(float));

		for (nInputIndex = 0; nInputIndex < inputData->nInputCount; ++nInputIndex)
		{
			if (inputData->data[nInputIndex].nLabelID == classCur->nID)
			{
				for (i = 0; i < inputData->nSize; ++i)
							fInputArray[i] = inputData->data[nInputIndex].fIntensity[i];
			
				ForwardPropagate_Train(cln->macData, cln->nMACCount, NULL);

			
				for (perceptronHeadCur = layerInput->perceptronHead, i = 0; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead)
				{
					for (perceptronCur = perceptronHeadCur; perceptronCur != NULL; perceptronCur = perceptronCur->next, ++i)
					{
						cln->fOutputArray[classCur->nID][i] += perceptronCur->fOutput;
					}
				}
			}
		}

		for (i = 0; i < layerInput->nOutputArraySize; ++i)
		{
			if (inputData->nClassMemberCount[classCur->nID] != 0)
				cln->fOutputArray[classCur->nID][i] /= (float)inputData->nClassMemberCount[classCur->nID];
			else
				cln->fOutputArray[classCur->nID][i] = 0.0f;
		}
	}
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void GetExtremeInputClassMembers_Neurogenesis(structNetwork *networkMain, structLayer *layerInput, structInput *inputData, float **fOutputArray, int nClassID, structMAC **macData, int nMACCount)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structPerceptron	*perceptronHeadCur;
	structPerceptron	*perceptronCur;
	float				fError;
	float				fTemp;
	float				fA, fB;
	float				fMaxError = -999999.0f;
	float				fMinError = 999999.0f;
	int					nSize = networkMain->nRowCount * networkMain->nColumnCount;
	int					nInputIndex;
	int					i;

	for (nInputIndex = 0; nInputIndex < inputData->nInputCount; ++nInputIndex)
	{
		if (inputData->data[nInputIndex].nLabelID == nClassID && !inputData->data[nInputIndex].bTrained)
		{
			for (i = 0; i < nSize; ++i)
				networkMain->fInputArray[i] = inputData->data[nInputIndex].fIntensity[i];

			ForwardPropagate_Train(*macData, nMACCount, NULL);

			for (perceptronHeadCur = layerInput->perceptronHead, fTemp = 0.0, i = 0; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead)
			{
				for (perceptronCur = perceptronHeadCur; perceptronCur != NULL; perceptronCur = perceptronCur->next, ++i)
				{
					fA = fOutputArray[nClassID][i];
					fB = perceptronCur->fOutput;

					fTemp += ((fA - fB) * (fA - fB));
				}
			}

			fError = (1.0f / (layerInput->nOutputArraySize*layerInput->nOutputArraySize))*fTemp;
			//input->data[j].fError = ComputeSSIM_InputData(fPixelTotal[classCur->nID], input->data[j].fIntensity, input->nSize);

			if (fError < fMinError)
			{
				fMinError = fError;
				inputData->nAverageIDArray[nClassID] = nInputIndex;
			}

			if (fError > fMaxError)
			{
				fMaxError = fError;
				inputData->nMaxIDArray[nClassID] = nInputIndex;
			}
		}
	}
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void AddExtremePerceptron_Neurogenesis(structNetwork *networkMain, structLayer *layerCur, structLayer **layerNew, structInput *inputData, int nInputIndex, int *nPerceptronCount, float fLearningRate, float fThreshold, float fMinimumThreshold, float fMaximumThreshold, structMAC **macData, int nMACCount)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structPerceptron	*perceptronCurHead;
	structPerceptron	*perceptronNew;
	structPerceptron	*perceptronCur;
	float				fValue;
	float				fSum = 0.0f;
	float				fStandardDeviation = 0.0f;
	float				fAverage;
	float				fSD;
	int					nSize = networkMain->nRowCount * networkMain->nColumnCount;
	int					i;

	for (i = 0; i < nSize; ++i)
		networkMain->fInputArray[i] = inputData->data[nInputIndex].fIntensity[i];

	ForwardPropagate_Train(*macData, nMACCount, NULL);

	for (perceptronCurHead = layerCur->perceptronHead, i = 0; perceptronCurHead != NULL; perceptronCurHead = perceptronCurHead->nextHead)
	{
		for (perceptronCur = perceptronCurHead; perceptronCur != NULL; perceptronCur = perceptronCur->next, ++i)
		{
			fSum += perceptronCur->fOutput;
		}
	}
	
	
	fAverage = fSum / i;

	for (perceptronCurHead = layerCur->perceptronHead; perceptronCurHead != NULL; perceptronCurHead = perceptronCurHead->nextHead)
		for (perceptronCur = perceptronCurHead; perceptronCur != NULL; perceptronCur = perceptronCur->next)
			fStandardDeviation += (float)pow(perceptronCur->fOutput - fAverage, 2.0f);

	fSD = (float)sqrt(fStandardDeviation / (float)layerCur->nOutputArraySize);


	++(*layerNew)->nWeightCount;

	perceptronNew = (structPerceptron *)calloc(1, sizeof(structPerceptron));
	perceptronNew->nID = networkMain->nPerceptronID++;

	perceptronNew->nLayerType = FULLY_CONNECTED_LAYER;
	perceptronNew->nIndex = *nPerceptronCount;
	perceptronNew->fLearningRate = fLearningRate;

	perceptronNew->fThreshold = fThreshold;

	perceptronNew->fSDUpper = fSD * fMaximumThreshold;
	perceptronNew->fSDLower = fSD * fMinimumThreshold;
	perceptronNew->nConnectionCount = 0;

	for (perceptronCurHead = layerCur->perceptronHead, i=0; perceptronCurHead != NULL; perceptronCurHead = perceptronCurHead->nextHead)
	{
		for (perceptronCur = perceptronCurHead; perceptronCur != NULL; perceptronCur = perceptronCur->next, ++i)
		{
			fValue = (float)fabs(perceptronCur->fOutput);

			if (fValue >= perceptronNew->fSDLower && fValue < perceptronNew->fSDUpper)
			{
				++(*layerNew)->nWeightCount;
				++perceptronNew->nConnectionCount;
			}
		}
	}


	for (perceptronCurHead = layerCur->perceptronHead, perceptronNew->nConnectionCount = 0; perceptronCurHead != NULL; perceptronCurHead = perceptronCurHead->nextHead)
	{
		for (perceptronCur = perceptronCurHead; perceptronCur != NULL; perceptronCur = perceptronCur->next, ++perceptronNew->nConnectionCount);
	}


	perceptronNew->nConnectionArray = (int *)calloc(perceptronNew->nConnectionCount, sizeof(int));
	perceptronNew->nConnectionCount = 0;

	for (perceptronCurHead = layerCur->perceptronHead, i = 0; perceptronCurHead != NULL; perceptronCurHead = perceptronCurHead->nextHead)
	{
		for (perceptronCur = perceptronCurHead; perceptronCur != NULL; perceptronCur = perceptronCur->next, ++i)
		{
			fValue = (float)fabs(perceptronCur->fOutput);

			if (fValue >= perceptronNew->fSDLower && fValue < perceptronNew->fSDUpper)
			{
				perceptronNew->nConnectionArray[perceptronNew->nConnectionCount++] = perceptronCur->nID;
			}
		}
	}


	AddNew_Perceptron(&(*layerNew)->perceptronHead, perceptronNew);
}



/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void ConnectExtremePerceptrons_Neurogenesis(structNetwork *networkMain, structLayer *layerCur)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structLayer			*layerInput;
	structPerceptron	*perceptronCurHead;
	structPerceptron	*perceptronCur;
	structPerceptron	*perceptronPrevCur;
	structSynapse		*synapseNew;
	float				fSum = 0.0f;
	float				fStandardDeviation = 0.0f;
	int					nSynapseIndex;
	int					i;

	nSynapseIndex = 0;
	layerInput = layerCur->prev;

	for (perceptronCur = layerCur->perceptronHead; perceptronCur != NULL; perceptronCur = perceptronCur->next)
	{
		if (perceptronCur->synapseHead != NULL)
			continue;
		
		//Add Bias Synapse ////////////////////////////////////////////////////////////////////////
		synapseNew = (structSynapse *)calloc(1, sizeof(structSynapse));
		synapseNew->nID = networkMain->nSynapseID++;
		synapseNew->nIndex = nSynapseIndex++;
		synapseNew->fWeight = &layerCur->fWeightArray[synapseNew->nIndex];
		synapseNew->fInput = (float *)calloc(1, sizeof(float));
		*(synapseNew->fInput) = 1; // Bias
		synapseNew->nInputArrayIndex = -1;

		AddNew_Synapse(&perceptronCur->synapseHead, synapseNew);

		for (perceptronCurHead = layerInput->perceptronHead; perceptronCurHead != NULL; perceptronCurHead = perceptronCurHead->nextHead)
		{
			for (perceptronPrevCur = perceptronCurHead; perceptronPrevCur != NULL; perceptronPrevCur = perceptronPrevCur->next)
			{
				for (i=0; i<perceptronCur->nConnectionCount; ++i)
				{
					if (perceptronPrevCur->nID == perceptronCur->nConnectionArray[i])
					{
						synapseNew = (structSynapse *)calloc(1, sizeof(structSynapse));

						synapseNew->nID = networkMain->nSynapseID++;
						synapseNew->nIndex = nSynapseIndex++;

						synapseNew->fWeight = &layerCur->fWeightArray[synapseNew->nIndex];
						synapseNew->perceptronConnectTo = perceptronPrevCur;
						synapseNew->fInput = &synapseNew->perceptronConnectTo->fOutput;
						synapseNew->nInputArrayIndex = -999;

						AddNew_Synapse(&perceptronCur->synapseHead, synapseNew);

						break;
					}
				}
			}
		}
	}
}


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void GrowNewLayer_Neurogenesis(structNetwork *networkMain, structCLN *cln, structLayer *layer, structInput *inputData, int *nPerceptronCount, float fLearningRate, float fThreshold, float fMinimumThreshold, float fMaximumThreshold)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structClass			*classCur = NULL;
	structPerceptron	*perceptronCur = NULL;
	structPerceptron	*perceptronCurHead = NULL;
	structSynapse		*synapseCur = NULL;
	float				*ptr = NULL;
	int					i;

	for (classCur = networkMain->classHead; classCur != NULL; classCur = classCur->next)
	{
		GetExtremeInputClassMembers_Neurogenesis(networkMain, networkMain->layerInput, inputData, cln->fOutputArray, classCur->nID, &cln->macData, cln->nMACCount);
		AddExtremePerceptron_Neurogenesis(networkMain, networkMain->layerInput, &layer, inputData, inputData->nAverageIDArray[classCur->nID], nPerceptronCount, fLearningRate, fThreshold, fMinimumThreshold, fMaximumThreshold, &cln->macData, cln->nMACCount);
		AddExtremePerceptron_Neurogenesis(networkMain, networkMain->layerInput, &layer, inputData, inputData->nMaxIDArray[classCur->nID], nPerceptronCount, fLearningRate, fThreshold, fMinimumThreshold, fMaximumThreshold, &cln->macData, cln->nMACCount);
		inputData->data[inputData->nAverageIDArray[classCur->nID]].bTrained = 1;
		inputData->data[inputData->nMaxIDArray[classCur->nID]].bTrained = 1;
		printf("%d\t%d\n", inputData->nAverageIDArray[classCur->nID], inputData->nMaxIDArray[classCur->nID]);
	}


	layer->nConnectionCount = layer->nWeightCount;
	layer->nOutputArraySize = layer->nPerceptronCount;


	if (layer->fWeightArray == NULL)
	{
		layer->fWeightArray = (float *)malloc(layer->nWeightCount * sizeof(float));

		for (i = 0; i < layer->nWeightCount; ++i)
			layer->fWeightArray[i] = MULTIPLIER * UNIFORM_PLUS_MINUS_ONE;
	}
	else
	{
		ptr = (float *)realloc(layer->fWeightArray, (layer->nWeightCount * sizeof(float)));
		layer->fWeightArray = ptr;

		for (perceptronCurHead = layer->perceptronHead; perceptronCurHead != NULL; perceptronCurHead = perceptronCurHead->nextHead)
		{
			for (perceptronCur = perceptronCurHead; perceptronCur != NULL; perceptronCur = perceptronCur->next)
			{
				for (synapseCur = perceptronCur->synapseHead; synapseCur != NULL; synapseCur = synapseCur->next)
				{
					synapseCur->fWeight = &layer->fWeightArray[synapseCur->nIndex];
				}
			}
		}
	}

	ConnectExtremePerceptrons_Neurogenesis(networkMain, layer);


}
