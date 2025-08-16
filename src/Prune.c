
#include "main.h"


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void PrunePerceptronWeights(structNetwork * network, structInput * inputTrain, structInput * inputTest, float* fAccuracy, int nMaxCluster)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structCLN* cln = network->clnHead;
	structLayer* layerCur = NULL;
	structPerceptron* perceptronCur = NULL;
	structSynapse* synapseCur = NULL;
	float				fPercent = 0.0f;
	float* fWeightArray = NULL;
	int					nTotalWeights = 0;
	int					nWeightCount = 0;
	int					nCurrentWeights = 0;
	int					nCount = 0;
	int					nPruneCount = 0;
	char				sTimeBuffer[32];

	int					nMidStart = 0;
	int					nMidEnd = 0;
	int					nUniqueCount = 0;
	int					nMid = 0;
	int					nPruneIndex = 0;
	int					nLoop = 0;

	GetClassLevelNetworkWeightCount(cln);
	nTotalWeights = 0;

	do
	{
		DescribeClassLevelNetwork(cln, NULL);

		nCurrentWeights = cln->nWeightCount;

		for (layerCur = cln->layerHead; layerCur != NULL; layerCur = layerCur->next)
		{
			if (layerCur->nLayerType == MAX_POOL_LAYER)
				continue;

			perceptronCur = layerCur->perceptronHead;

			while (perceptronCur != NULL)
			{
				if (perceptronCur->synapseHead != NULL)
				{
					// Copy Perceptron Weights
					CopyPerceptronWeights(perceptronCur, NETWORK_TO_MEMORY);

					// Create Array of Weights
					for (synapseCur = perceptronCur->synapseHead->next, nCount = 0; synapseCur != NULL; synapseCur = synapseCur->next, ++nCount);

					fWeightArray = (float*)calloc(nCount, sizeof(float));
					nTotalWeights += nCount;

					for (synapseCur = perceptronCur->synapseHead->next, nCount = 0; synapseCur != NULL; synapseCur = synapseCur->next)
						fWeightArray[nCount++] = fabsf(*synapseCur->fWeight);

					// Order Least to Greatest
					SortFloatAscend(fWeightArray, nCount);


					nMidStart = 0;
					nMidEnd = nCount;
					nMid = (nMidStart + nMidEnd) / 2;
					nPruneIndex = -1;

					while (nMidStart != nMidEnd)
					{
						if (fWeightArray[nMid] != 0.0f)
						{
							nPruneCount = 0;
							for (synapseCur = perceptronCur->synapseHead->next; synapseCur != NULL; synapseCur = synapseCur->next)
							{
								if (fabs(*synapseCur->fWeight) < fWeightArray[nMid])
								{
									++nPruneCount;
									*synapseCur->fWeight = 0.0f;
								}
							}

							InferCLN_Inference(cln, network->classHead, inputTrain, HIDE_DATA, network->nMatrix, network->fInputArray, 1, network->sDrive, network->sTitle, DO_NOT_MARK, NO_THRESHOLD, NULL, sTimeBuffer, BREAK_ON_BAD_CLASSIFICATION, nMaxCluster);
							*fAccuracy = ScoreClassLevelNetworkMatrix(cln->nNetworkType, 0, network->classHead, network->nMatrix, HIDE_MATRIX, NULL);

							if (*fAccuracy == 1.0f)
							{
								// Copy Perceptron Weights
								CopyPerceptronWeights(perceptronCur, NETWORK_TO_MEMORY);
								++nTotalWeights;

								nPruneIndex = nMid;
								if (nMidStart == nMid)
									break;

								nMidStart = nMid;
							}
							else
							{
								// Reset Perceptron Weights
								CopyPerceptronWeights(perceptronCur, MEMORY_TO_NETWORK);
								SortByMissCount(inputTrain->data, inputTrain->nInputCount);

								if (nMidEnd == nMid)
									break;

								nMidEnd = nMid;
							}

							nMid = (nMidStart + nMidEnd) / 2;
						}
						else
							++nMid;
					}

					nWeightCount += (nCount - nPruneCount);
					fPercent = (float)(nWeightCount - nTotalWeights) / (float)nTotalWeights;

					if (layerCur->nLayerType == CONV_2D_LAYER || layerCur->nLayerType == CONV_3D_LAYER)
						printf("%6d\t%6d\t%6d\t%6d\t%0.4f\n", perceptronCur->nHeadIndex, layerCur->nLayerType, nTotalWeights, nWeightCount, fPercent);
					else
						printf("%6d\t%6d\t%6d\t%6d\t%0.4f\n", perceptronCur->nIndex, layerCur->nLayerType, nTotalWeights, nWeightCount, fPercent);

					free(fWeightArray);
				}

				if (layerCur->nLayerType == CONV_2D_LAYER || layerCur->nLayerType == CONV_3D_LAYER)
					perceptronCur = perceptronCur->nextHead;
				else
					perceptronCur = perceptronCur->next;
			}
		}

		DeleteZeroWeights(cln);
		GetClassLevelNetworkWeightCount(cln);
		CreateMACArrayFromDNX_ClassLevelNetworks(cln);

		InferCLN_Inference(cln, network->classHead, inputTrain, HIDE_DATA, network->nMatrix, network->fInputArray, 1, network->sDrive, network->sTitle, DO_NOT_MARK, NO_THRESHOLD, NULL, sTimeBuffer, 0, -1);
		*fAccuracy = ScoreClassLevelNetworkMatrix(network->clnHead->nNetworkType, 0, network->classHead, network->nMatrix, HIDE_MATRIX, NULL);
		printf("DNX_Train: %f\t%s\n", *fAccuracy, sTimeBuffer);

		InferCLN_Inference(network->clnHead, network->classHead, inputTest, HIDE_DATA, network->nMatrix, network->fInputArray, 1, network->sDrive, network->sTitle, DO_NOT_MARK, NO_THRESHOLD, NULL, sTimeBuffer, 0, -1);
		*fAccuracy = ScoreClassLevelNetworkMatrix(network->clnHead->nNetworkType, 0, network->classHead, network->nMatrix, HIDE_MATRIX, NULL);
		printf("DNX_Test: %f\t%s\n\n", *fAccuracy, sTimeBuffer);

		sprintf(network->sDNXOutputPath, "sic_%d_%0.4f.dnx", nLoop++, *fAccuracy);
		SaveNetwork(network, network->sDNXOutputPath);

		printf("---> %d\t%d\n", cln->nWeightCount, nCurrentWeights);

	} while (cln->nWeightCount != nCurrentWeights);

}




/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void PrunePerceptronWeightsSinglePass(structNetwork* network, structInput* inputTrain, structInput* inputTest, int nMaxCluster)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structCLN* cln = network->clnHead;
	structLayer* layerCur = NULL;
	structPerceptron* perceptronCur = NULL;
	structSynapse* synapseCur = NULL;
	float				fPercent = 0.0f;
	float				fAccuracy;
	float* fWeightArray = NULL;
	int					nTotalWeights = 0;
	int					nWeightCount = 0;
	int					nCurrentWeights = 0;
	int					nCount = 0;
	int					nPruneCount = 0;
	char				sTimeBuffer[32];

	int					nMidStart = 0;
	int					nMidEnd = 0;
	int					nUniqueCount = 0;
	int					nMid = 0;
	int					nPruneIndex = 0;
	int					nLoop = 0;

	printf("***** Prune *****\n");

	
	nTotalWeights = 0;

	for (layerCur = cln->layerHead; layerCur != NULL; layerCur = layerCur->next)
	{
		if (layerCur->nLayerType == MAX_POOL_LAYER)
			continue;

		perceptronCur = layerCur->perceptronHead;

		while (perceptronCur != NULL)
		{
			if (perceptronCur->synapseHead != NULL)
			{
				// Copy Perceptron Weights
				CopyPerceptronWeights(perceptronCur, NETWORK_TO_MEMORY);

				// Create Array of Weights
				for (synapseCur = perceptronCur->synapseHead->next, nCount = 0; synapseCur != NULL; synapseCur = synapseCur->next, ++nCount);

				fWeightArray = (float*)calloc(nCount, sizeof(float));
				nTotalWeights += nCount;

				for (synapseCur = perceptronCur->synapseHead->next, nCount = 0; synapseCur != NULL; synapseCur = synapseCur->next)
					fWeightArray[nCount++] = fabsf(*synapseCur->fWeight);

				// Order Least to Greatest
				SortFloatAscend(fWeightArray, nCount);


				nMidStart = 0;
				nMidEnd = nCount;
				nMid = (nMidStart + nMidEnd) / 2;
				nPruneIndex = -1;

				while (nMidStart != nMidEnd)
				{
					if (fWeightArray[nMid] != 0.0f)
					{
						nPruneCount = 0;
						for (synapseCur = perceptronCur->synapseHead->next; synapseCur != NULL; synapseCur = synapseCur->next)
						{
							if (fabs(*synapseCur->fWeight) < fWeightArray[nMid])
							{
								++nPruneCount;
								*synapseCur->fWeight = 0.0f;
							}
						}

						InferCLN_Inference(cln, network->classHead, inputTrain, HIDE_DATA, network->nMatrix, network->fInputArray, 1, network->sDrive, network->sTitle, DO_NOT_MARK, NO_THRESHOLD, NULL, sTimeBuffer, BREAK_ON_BAD_CLASSIFICATION, nMaxCluster);
						fAccuracy = ScoreClassLevelNetworkMatrix(cln->nNetworkType, 0, network->classHead, network->nMatrix, HIDE_MATRIX, NULL);

						if (fAccuracy == 1.0f)
						{
							// Copy Perceptron Weights
							CopyPerceptronWeights(perceptronCur, NETWORK_TO_MEMORY);
							++nTotalWeights;

							nPruneIndex = nMid;
							if (nMidStart == nMid)
								break;

							nMidStart = nMid;
						}
						else
						{
							// Reset Perceptron Weights
							CopyPerceptronWeights(perceptronCur, MEMORY_TO_NETWORK);
							SortByMissCount(inputTrain->data, inputTrain->nInputCount);

							if (nMidEnd == nMid)
								break;

							nMidEnd = nMid;
						}

						nMid = (nMidStart + nMidEnd) / 2;
					}
					else
						++nMid;
				}

				nWeightCount += (nCount - nPruneCount);
				fPercent = (float)(nWeightCount - nTotalWeights) / (float)nTotalWeights;

				if (layerCur->nLayerType == CONV_2D_LAYER || layerCur->nLayerType == CONV_3D_LAYER)
					printf("%6d\t%6d\t%6d\t%6d\t%0.4f\n", perceptronCur->nHeadIndex, layerCur->nLayerType, nTotalWeights, nWeightCount, fPercent);
				else
					printf("%6d\t%6d\t%6d\t%6d\t%0.4f\n", perceptronCur->nIndex, layerCur->nLayerType, nTotalWeights, nWeightCount, fPercent);

				free(fWeightArray);
			}

			if (layerCur->nLayerType == CONV_2D_LAYER || layerCur->nLayerType == CONV_3D_LAYER)
				perceptronCur = perceptronCur->nextHead;
			else
				perceptronCur = perceptronCur->next;
		}
	}

	DeleteZeroWeights(cln);
	GetClassLevelNetworkWeightCount(cln);
	CreateMACArrayFromDNX_ClassLevelNetworks(cln);
}







/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
int PruneWeights(structLayer **layerHead, float fConvThreshold, float fFCThreshold)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structPerceptron	*perceptronHeadCur;
	structPerceptron	*perceptronCur;
	structSynapse		*synapseCur;
	structLayer			*layerCur;
	int					nWeightCount = 0;
	int					nID = 0;

	for (layerCur = *layerHead; layerCur != NULL; layerCur = layerCur->next)
	{
		if (layerCur->nLayerType == SINGLE_CONV_LAYER || layerCur->nLayerType == MULTIPLE_CONV_LAYER)
		{
			for (perceptronHeadCur = layerCur->perceptronHead; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead)
			{
				for (perceptronCur = perceptronHeadCur; perceptronCur != NULL; perceptronCur = perceptronCur->next)
				{
					if (perceptronCur->synapseHead != NULL)
					{
						for (synapseCur = perceptronCur->synapseHead; synapseCur != NULL; synapseCur = synapseCur->next)
						{					
							if (fabs(*synapseCur->fWeight) <= fabs(fConvThreshold))
							{
								synapseCur = Delete_Synapse(&perceptronCur->synapseHead, synapseCur->nID);
								if (synapseCur == NULL)
									break;
							}
							else
								++nWeightCount;
						}
					}
				}
			}
		}
		else if ((layerCur->nLayerType == FULLY_CONNECTED_LAYER || layerCur->nLayerType == CLASSIFIER_LAYER)) //  
		{
			for (perceptronHeadCur = layerCur->perceptronHead; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead)
			{
				for (perceptronCur = perceptronHeadCur; perceptronCur != NULL; perceptronCur = perceptronCur->next)
				{
					if (perceptronCur->synapseHead != NULL)
					{
						for (synapseCur = perceptronCur->synapseHead; synapseCur != NULL; synapseCur = synapseCur->next)
						{							
							if (fabs(*synapseCur->fWeight) <= fabs(fFCThreshold))
							{
								synapseCur = Delete_Synapse(&perceptronCur->synapseHead, synapseCur->nID);
								if (synapseCur == NULL)
									break;
							}
							else
								++nWeightCount;
						}
					}
				}
			}
		}
	}

	return(nWeightCount);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
int SetPruneWeightsZero(structLayer **layerHead, float fConvThreshold, float fFCThreshold)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structPerceptron	*perceptronHeadCur;
	structPerceptron	*perceptronCur;
	structSynapse		*synapseCur;
	structLayer			*layerCur;
	int					nWeightCount = 0;
	int					nID = 0;
	int					nRemoveCount = 0;

	for (layerCur = *layerHead; layerCur != NULL; layerCur = layerCur->next)
	{
		if (layerCur->nLayerType == SINGLE_CONV_LAYER || layerCur->nLayerType == MULTIPLE_CONV_LAYER)
		{
			for (perceptronHeadCur = layerCur->perceptronHead; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead)
			{
				for (perceptronCur = perceptronHeadCur; perceptronCur != NULL; perceptronCur = perceptronCur->next)
				{
					if (perceptronCur->synapseHead != NULL)
					{
						for (synapseCur = perceptronCur->synapseHead; synapseCur != NULL; synapseCur = synapseCur->next)
						{					
							if (fabs(*synapseCur->fWeight) <= fabs(fConvThreshold))
							{
								//synapseCur = Delete_Synapse(&perceptronCur->synapseHead, synapseCur->nID);
								
								*synapseCur->fWeight = 0.0f;
								++nRemoveCount;
								
								if (synapseCur == NULL)
									break;
							}
							else
								++nWeightCount;
						}
					}
				}
			}
		}
		else if ((layerCur->nLayerType == FULLY_CONNECTED_LAYER || layerCur->nLayerType == CLASSIFIER_LAYER)) //  
		{
			for (perceptronHeadCur = layerCur->perceptronHead; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead)
			{
				for (perceptronCur = perceptronHeadCur; perceptronCur != NULL; perceptronCur = perceptronCur->next)
				{
					if (perceptronCur->synapseHead != NULL)
					{
						for (synapseCur = perceptronCur->synapseHead; synapseCur != NULL; synapseCur = synapseCur->next)
						{							
							if (fabs(*synapseCur->fWeight) <= fabs(fFCThreshold))
							{
								//synapseCur = Delete_Synapse(&perceptronCur->synapseHead, synapseCur->nID);
								
								*synapseCur->fWeight = 0.0f;
								++nRemoveCount;
								
								if (synapseCur == NULL)
									break;
							}
							else
								++nWeightCount;
						}
					}
				}
			}
		}
	}

	return(nWeightCount);
}


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
int SetMACPruneWeightsZero(structMAC *macData, int nMACCount, float fConvThreshold, float fFCThreshold)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	int		nWeightCount = 0;
	int		p, q;

	for (p = 0; p < nMACCount; ++p)
	{
		if (macData[p].nLayerType == FULLY_CONNECTED_LAYER || macData[p].nLayerType == CLASSIFIER_LAYER)
		{
			//if (fabs(*(macData[p]).fWeight[0]) <= fabs(fFCThreshold))
			//	*(macData[p]).fWeight[0] = 0.0f;

			for (q = 1; q < macData[p].nCount; ++q)
			{
				if (fabs(*(macData[p]).fWeight[q]) <= fabs(fFCThreshold))
					*(macData[p]).fWeight[q] = 0.0f;
			}
		}
		else if (macData[p].nLayerType == SINGLE_CONV_LAYER || macData[p].nLayerType == MULTIPLE_CONV_LAYER)
		{
			//if (fabs(*(macData[p]).fWeight[0]) <= fabs(fConvThreshold))
			//	*(macData[p]).fWeight[0] = 0.0f;

			for (q = 1; q < macData[p].nCount; ++q)
			{
				if (fabs(*(macData[p]).fWeight[q]) <= fabs(fConvThreshold))
					*(macData[p]).fWeight[q] = 0.0f;
			}
		}
		
	}

	return(nWeightCount);
}



/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
int PruneWeights_V2(structNetwork *network, structLayer **layerHead, structInput *inputValidateData, structInput *inputTrainData)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structPerceptron	*perceptronHeadCur;
	structPerceptron	*perceptronCur;
	structSynapse		*synapseCur;
	structLayer			*layerCur;
	float				*fWeights;
	float				fValidateAccuracy;
	float				fTrainAccuracy;
	float				fAccuracy;
	int					nWeightCount = 0;
	int					nRemoved = 0;
	int					nID = 0;
	int					nIndex;
	int					i, j;
	int					nUpper;
	int					nLower;
	char				sTimeBuffer[32];
	
	structData			*data;
	structData			temp;

	fWeights = (float *)calloc(network->clnHead->nWeightCount, sizeof(float));

	CopyWeights(network->clnHead, NETWORK_TO_MEMORY);

	InferCLN_Inference(network->clnHead, network->classHead, inputValidateData, HIDE_DATA, network->nMatrix, network->fInputArray, 1, network->sDrive, network->sTitle, DO_NOT_MARK, NO_THRESHOLD, NULL, sTimeBuffer, 0, -1);
	fValidateAccuracy = ScoreClassLevelNetworkMatrix(network->clnHead->nNetworkType, 0, network->classHead, network->nMatrix, HIDE_MATRIX, NULL);
	printf("Validate Accuracy: %f\n\n", fValidateAccuracy);

	InferCLN_Inference(network->clnHead, network->classHead, inputTrainData, HIDE_DATA, network->nMatrix, network->fInputArray, 1, network->sDrive, network->sTitle, DO_NOT_MARK, NO_THRESHOLD, NULL, sTimeBuffer, 0, -1);
	fTrainAccuracy = ScoreClassLevelNetworkMatrix(network->clnHead->nNetworkType, 0, network->classHead, network->nMatrix, HIDE_MATRIX, NULL);
	printf("Train Accuracy: %f\n\n", fTrainAccuracy);
	

	for (layerCur = *layerHead; layerCur != NULL; layerCur = layerCur->next)
	{
		if (layerCur->nLayerType == SINGLE_CONV_LAYER || layerCur->nLayerType == MULTIPLE_CONV_LAYER)
		{
			for (perceptronCur = layerCur->perceptronHead, nWeightCount = 0; perceptronCur != NULL; perceptronCur = perceptronCur->nextHead)
			{
				if (perceptronCur->synapseHead != NULL)
				{
					for (synapseCur = perceptronCur->synapseHead->next; synapseCur != NULL; synapseCur = synapseCur->next, ++nWeightCount);
				}
			}

			data = (structData *)calloc(nWeightCount, sizeof(structData));

			for (perceptronCur = layerCur->perceptronHead, nWeightCount = 0; perceptronCur != NULL; perceptronCur = perceptronCur->nextHead)
			{
				if (perceptronCur->synapseHead != NULL)
				{
					for (synapseCur = perceptronCur->synapseHead->next; synapseCur != NULL; synapseCur = synapseCur->next, ++nWeightCount)
					{
						data[nWeightCount].fOutput = fabsf(*synapseCur->fWeight);
					}
				}
			}

			for (i = 0; i < nWeightCount - 1; i++)
			{
				for (j = 0; j < nWeightCount - i - 1; j++)
				{
					if (data[j].fOutput > data[j + 1].fOutput)
					{
						temp = data[j];
						data[j] = data[j + 1];
						data[j + 1] = temp;
					}
				}
			}

			nUpper = nWeightCount;
			nLower = 0;
			nIndex = (nUpper + nLower) / 2;

			while(nIndex != nUpper && nIndex != nLower)
			{
				// Reset Weights
				CopyWeights(network->clnHead, MEMORY_TO_NETWORK);

				for (perceptronCur = layerCur->perceptronHead, nWeightCount = 0; perceptronCur != NULL; perceptronCur = perceptronCur->nextHead)
				{
					if (perceptronCur->synapseHead != NULL)
					{
						for (synapseCur = perceptronCur->synapseHead->next; synapseCur != NULL; synapseCur = synapseCur->next)
						{
							if (fabsf(*synapseCur->fWeight) <= data[nIndex].fOutput)
							{
								*synapseCur->fWeight = 0.0f;
								++nRemoved;
							}
						}
					}
				}

				InferCLN_Inference(network->clnHead, network->classHead, inputValidateData, HIDE_DATA, network->nMatrix, network->fInputArray, 1, network->sDrive, network->sTitle, DO_NOT_MARK, NO_THRESHOLD, NULL, sTimeBuffer, 0, -1);
				fAccuracy = ScoreClassLevelNetworkMatrix(network->clnHead->nNetworkType, 0, network->classHead, network->nMatrix, HIDE_MATRIX, NULL);
				printf("Pruned Accuracy: %f\t%d\t%d\t%d\n", fAccuracy - fTrainAccuracy, nIndex, nLower, nUpper);

				if (fAccuracy < fValidateAccuracy)
					nUpper = nIndex;
				else
				{
					InferCLN_Inference(network->clnHead, network->classHead, inputTrainData, HIDE_DATA, network->nMatrix, network->fInputArray, 1, network->sDrive, network->sTitle, DO_NOT_MARK, NO_THRESHOLD, NULL, sTimeBuffer, 0, -1);
					fAccuracy = ScoreClassLevelNetworkMatrix(network->clnHead->nNetworkType, 0, network->classHead, network->nMatrix, HIDE_MATRIX, NULL);
					printf("Pruned Accuracy: %f\t%d\t%d\t%d\n", fAccuracy - fTrainAccuracy, nIndex, nLower, nUpper);

					if (fAccuracy < fTrainAccuracy)
						nUpper = nIndex;
					else
						nLower = nIndex;
				}

				nIndex = (nUpper + nLower) / 2;
			}

			printf("\n\n");
			CopyWeights(network->clnHead, NETWORK_TO_MEMORY);

			free(data);
		}
		else
		{
			for (perceptronHeadCur = layerCur->perceptronHead, nWeightCount = 0; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead)
			{
				for (perceptronCur = perceptronHeadCur; perceptronCur != NULL; perceptronCur = perceptronCur->next)
				{
					if (perceptronCur->synapseHead != NULL)
					{
						for (synapseCur = perceptronCur->synapseHead->next; synapseCur != NULL; synapseCur = synapseCur->next, ++nWeightCount);
					}
				}
			}

			data = (structData *)calloc(nWeightCount, sizeof(structData));

			for (perceptronHeadCur = layerCur->perceptronHead, nWeightCount = 0; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead)
			{
				for (perceptronCur = perceptronHeadCur; perceptronCur != NULL; perceptronCur = perceptronCur->next)
				{
					if (perceptronCur->synapseHead != NULL)
					{
						for (synapseCur = perceptronCur->synapseHead->next; synapseCur != NULL; synapseCur = synapseCur->next, ++nWeightCount)
						{
							data[nWeightCount].fOutput = fabsf(*synapseCur->fWeight);
						}
					}
				}
			}

			for (i = 0; i < nWeightCount - 1; i++)
			{
				for (j = 0; j < nWeightCount - i - 1; j++)
				{
					if (data[j].fOutput > data[j + 1].fOutput)
					{
						temp = data[j];
						data[j] = data[j + 1];
						data[j + 1] = temp;
					}
				}
			}

			nUpper = nWeightCount;
			nLower = 0;
			nIndex = (nUpper + nLower) / 2;

			while (nIndex != nUpper && nIndex != nLower)
			{
				// Reset Weights
				CopyWeights(network->clnHead, MEMORY_TO_NETWORK);

				for (perceptronHeadCur = layerCur->perceptronHead, nRemoved = 0; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead)
				{
					for (perceptronCur = perceptronHeadCur; perceptronCur != NULL; perceptronCur = perceptronCur->next)
					{
						if (perceptronCur->synapseHead != NULL)
						{
							for (synapseCur = perceptronCur->synapseHead->next; synapseCur != NULL; synapseCur = synapseCur->next)
							{
								if (fabsf(*synapseCur->fWeight) <= data[nIndex].fOutput)
								{
									*synapseCur->fWeight = 0.0f;
									++nRemoved;
								}
							}
						}
					}
				}

				InferCLN_Inference(network->clnHead, network->classHead, inputValidateData, HIDE_DATA, network->nMatrix, network->fInputArray, 1, network->sDrive, network->sTitle, DO_NOT_MARK, NO_THRESHOLD, NULL, sTimeBuffer, 0, -1);
				fAccuracy = ScoreClassLevelNetworkMatrix(network->clnHead->nNetworkType, 0, network->classHead, network->nMatrix, HIDE_MATRIX, NULL);
				printf("Pruned Accuracy: %f\t%d\t%d\t%d\n", fAccuracy - fTrainAccuracy, nIndex, nLower, nUpper);

				if (fAccuracy < fValidateAccuracy)
					nUpper = nIndex;
				else
				{
					InferCLN_Inference(network->clnHead, network->classHead, inputTrainData, HIDE_DATA, network->nMatrix, network->fInputArray, 1, network->sDrive, network->sTitle, DO_NOT_MARK, NO_THRESHOLD, NULL, sTimeBuffer, 0, -1);
					fAccuracy = ScoreClassLevelNetworkMatrix(network->clnHead->nNetworkType, 0, network->classHead, network->nMatrix, HIDE_MATRIX, NULL);
					printf("Pruned Accuracy: %f\t%d\t%d\t%d\n", fAccuracy - fTrainAccuracy, nIndex, nLower, nUpper);

					if (fAccuracy < fTrainAccuracy)
						nUpper = nIndex;
					else
						nLower = nIndex;
				}

				nIndex = (nUpper + nLower) / 2;
			}

			printf("\n\n");
			CopyWeights(network->clnHead, NETWORK_TO_MEMORY);

			free(data);
		}
	}

	free(fWeights);

	CreateMACArray_ClassLevelNetworks(network->clnHead);

	InferCLN_Inference(network->clnHead, network->classHead, inputTrainData, HIDE_DATA, network->nMatrix, network->fInputArray, 1, network->sDrive, network->sTitle, DO_NOT_MARK, NO_THRESHOLD, NULL, sTimeBuffer, 0, -1);
	fAccuracy = ScoreClassLevelNetworkMatrix(network->clnHead->nNetworkType, 0, network->classHead, network->nMatrix, HIDE_MATRIX, NULL);
	printf("Pre Connection Deletion: %f\n", fAccuracy);

	nRemoved = 0;
	for (layerCur = *layerHead; layerCur != NULL; layerCur = layerCur->next)
	{
		for (perceptronHeadCur = layerCur->perceptronHead, nWeightCount = 0; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead)
		{
			for (perceptronCur = perceptronHeadCur; perceptronCur != NULL; perceptronCur = perceptronCur->next)
			{
				if (perceptronCur->synapseHead->next != NULL)  // Do not remove bias
				{
					for (synapseCur = perceptronCur->synapseHead->next; synapseCur != NULL; synapseCur = synapseCur->next)
					{
						if (*synapseCur->fWeight == 0.0f)
						{
							synapseCur = Delete_Synapse(&perceptronCur->synapseHead, synapseCur->nID);
							++nRemoved;
							
							if (synapseCur == NULL)
								break;
						}
						else
							++nWeightCount;
					}
				}
			}
		}
	}

	CreateMACArray_ClassLevelNetworks(network->clnHead);

	InferCLN_Inference(network->clnHead, network->classHead, inputTrainData, HIDE_DATA, network->nMatrix, network->fInputArray, 1, network->sDrive, network->sTitle, DO_NOT_MARK, NO_THRESHOLD, NULL, sTimeBuffer, 0, -1);
	fAccuracy = ScoreClassLevelNetworkMatrix(network->clnHead->nNetworkType, 0, network->classHead, network->nMatrix, HIDE_MATRIX, NULL);
	printf("Post Connection Deletion: %f\n", fAccuracy);


	return(nWeightCount);
}


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void RealignWeights(structCLN *cln)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structLayer			*layerCur;
	structPerceptron	*perceptronCurHead;
	structPerceptron	*perceptronCur;
	structSynapse		*synapseCur;
	structSynapse		*synapseSource;
	float				*fWeightArray;
	int					nWeightCount = 0;
	int					nCount = 0;

	for (layerCur = cln->layerHead; layerCur != NULL; layerCur = layerCur->next)
	{
		if (layerCur->nLayerType == SINGLE_CONV_LAYER || layerCur->nLayerType == MULTIPLE_CONV_LAYER)
		{
			for (perceptronCurHead = layerCur->perceptronHead, nWeightCount = 0; perceptronCurHead != NULL; perceptronCurHead = perceptronCurHead->nextHead)
			{
				for (synapseCur = perceptronCurHead->synapseHead; synapseCur != NULL; synapseCur = synapseCur->next, ++nWeightCount);
			}

			if ((fWeightArray = (float *)calloc(nWeightCount, sizeof(float))) == NULL)
				exit(0);
		
			nCount = 0;
			for (perceptronCurHead = layerCur->perceptronHead; perceptronCurHead != NULL; perceptronCurHead = perceptronCurHead->nextHead)
			{
				nWeightCount=nCount;
				for (synapseCur = perceptronCurHead->synapseHead; synapseCur != NULL; synapseCur = synapseCur->next, ++nWeightCount)
				{
					fWeightArray[nWeightCount] = *synapseCur->fWeight;
					synapseCur->nIndex = nWeightCount;
					synapseCur->fWeight = &fWeightArray[synapseCur->nIndex];
				}

				for (perceptronCur = perceptronCurHead->next; perceptronCur != NULL; perceptronCur = perceptronCur->next)
				{
					for (synapseCur = perceptronCur->synapseHead, synapseSource= perceptronCurHead->synapseHead; synapseCur != NULL; synapseCur = synapseCur->next, synapseSource= synapseSource->next)
					{
						synapseCur->fWeight = synapseSource->fWeight;
						synapseCur->nIndex = synapseSource->nIndex;
					}
				}

				nCount = nWeightCount;
			}

			layerCur->nWeightCount = nWeightCount;
			free(layerCur->fWeightArray);
			layerCur->fWeightArray = fWeightArray;
		}
		else
		{
			for (perceptronCurHead = layerCur->perceptronHead, nWeightCount = 0; perceptronCurHead != NULL; perceptronCurHead = perceptronCurHead->nextHead)
			{
				for (perceptronCur = perceptronCurHead; perceptronCur != NULL; perceptronCur = perceptronCur->next)
				{
					for (synapseCur = perceptronCur->synapseHead; synapseCur != NULL; synapseCur = synapseCur->next, ++nWeightCount);
				}
			}

			if ((fWeightArray = (float *)calloc(nWeightCount, sizeof(float))) == NULL)
				exit(0);
			
			for (perceptronCurHead = layerCur->perceptronHead, nWeightCount = 0; perceptronCurHead != NULL; perceptronCurHead = perceptronCurHead->nextHead)
			{
				for (perceptronCur = perceptronCurHead; perceptronCur != NULL; perceptronCur = perceptronCur->next)
				{
					for (synapseCur = perceptronCur->synapseHead; synapseCur != NULL; synapseCur = synapseCur->next, ++nWeightCount)
					{
						fWeightArray[nWeightCount] = *synapseCur->fWeight;
						synapseCur->nIndex = nWeightCount;
						synapseCur->fWeight = &fWeightArray[synapseCur->nIndex];
					}
				}
			}

			layerCur->nWeightCount = nWeightCount;
			free(layerCur->fWeightArray);
			layerCur->fWeightArray = fWeightArray;
		}
	}

	CreateMACArray_ClassLevelNetworks(cln);
}
