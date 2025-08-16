#include "main.h"

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
int CreateCLN_ClassLevelNetworks(structCLN	**cln, int nLayerCount, structArchitecture *architecture, int nNetworkType, int nLabelID, int *nClassLevelNetworkCount, float fLearningRate, float fInitialError, float fThreshold, int nRowCount, int nColumnCount, int *nPerceptronID, int *nSynapseID, float *fInputArray, int nMode)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	int			nWindowSize=0;
	int			i;

	if (((*cln) = (structCLN *)calloc(1, sizeof(structCLN))) == NULL)
		exit(0);

	(*cln)->nNetworkType = nNetworkType;
	(*cln)->nID = (*nClassLevelNetworkCount)++;
	(*cln)->fLearningRate = fLearningRate;
	(*cln)->fInitialError = fInitialError;
	(*cln)->fThreshold = fThreshold;
	(*cln)->nLabelID = nLabelID;
	(*cln)->nSize = nRowCount * nColumnCount;
	(*cln)->nPerceptronLayerCount = 0;

	//%%%%%%%%%% Build Network %%%%%%%%%%
	for (i = 0; i<nLayerCount; ++i)
	{
		printf("nLayerCount: %d nLayerType: %d ...", nLayerCount - i, architecture[i].nLayerType);
		
		if (architecture[i].nLayerType == MAX_POOLING_LAYER)
		{
			CreateMaxPoolLayer_ClassLevelNetworks((*cln), &architecture[i], fLearningRate, nMode, nRowCount, nColumnCount, nPerceptronID, nSynapseID, fInputArray);
			printf(" MAX_POOLING_LAYER");
		}
		else if (architecture[i].nLayerType == SINGLE_CONV_LAYER)
		{
			Create2DConvolveLayer_ClassLevelNetworks((*cln), &architecture[i], fLearningRate, nMode, nRowCount, nColumnCount, nPerceptronID, nSynapseID, fInputArray);
			printf(" SINGLE_CONV_LAYER");
		}
		else if (architecture[i].nLayerType == MULTIPLE_CONV_LAYER)
		{
			Create3DConvolveLayer_ClassLevelNetworks((*cln), &architecture[i], fLearningRate, nMode, nRowCount, nColumnCount, nPerceptronID, nSynapseID, fInputArray);
			printf(" MULTIPLE_CONV_LAYER");
		}
		else if (architecture[i].nLayerType == FULLY_CONNECTED_LAYER)
		{
			CreateConnectionLayer_ClassLevelNetworks((*cln), architecture[i].nKernelCount, FULLY_CONNECTED_LAYER, fLearningRate, nMode, nRowCount, nColumnCount, nPerceptronID, nSynapseID, fInputArray);
			
			architecture[i].nOutputRows = architecture[i].nKernelCount;
			architecture[i].nOutputColumns = 1;
			printf(" FULLY_CONNECTED_LAYER");
		}
		else if (architecture[i].nLayerType == CLASSIFIER_LAYER)
		{
			if(nNetworkType == COMPLETE_NETWORK)
				CreateConnectionLayer_ClassLevelNetworks((*cln), nLabelID, CLASSIFIER_LAYER, fLearningRate, nMode, nRowCount, nColumnCount, nPerceptronID, nSynapseID, fInputArray);
			else if(nNetworkType == CLASS_NETWORK)
				CreateConnectionLayer_ClassLevelNetworks((*cln), 2, CLASSIFIER_LAYER, fLearningRate, nMode, nRowCount, nColumnCount, nPerceptronID, nSynapseID, fInputArray);
			printf(" CLASSIFIER_LAYER");
		}

		printf("\n");
		
		// printf("nID = %d\n", architecture[i].nID);
		// printf("nLayerType = %d\n", architecture[i].nLayerType);
		// printf("fAccuracy = %f\n", architecture[i].fAccuracy);
		// printf("fPercentResponse = %f\n", architecture[i].fPercentResponse);
		// printf("nKernelCount = %d\n", architecture[i].nKernelCount);
		// printf("nRowKernelSize = %d\n", architecture[i].nRowKernelSize);
		// printf("nColumnKernelSize = %d\n", architecture[i].nColumnKernelSize);
		// printf("nStrideRow = %d\n", architecture[i].nStrideRow);
		// printf("nStrideColumn = %d\n", architecture[i].nStrideColumn);
		// printf("nWeightCount = %d\n", architecture[i].nWeightCount);
		// printf("bKeep = %d\n", architecture[i].bKeep);

		// printf("nInputRowCount = %d\n", architecture[i].nInputRowCount);
		// printf("nInputColumnCount = %d\n", architecture[i].nInputColumnCount);
		// printf("fLearningRate = %f\n", architecture[i].fLearningRate);
		// printf("nPaddingMode = %d\n", architecture[i].nPaddingMode);
		// printf("nActivationMode = %d\n", architecture[i].nActivationMode);
		// printf("nNumberFormat = %d\n", architecture[i].nNumberFormat);

		// if(architecture[i].next != NULL) {
		// 	printf("Next nID = %d\n", architecture[i].next->nID);
		// }

		// if(architecture[i].prev != NULL) {
		// 	printf("Prev nID = %d\n", architecture[i].prev->nID);
		// }
	}

	CreateMACArray_ClassLevelNetworks((*cln));

	return(nWindowSize);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
int CreateCLN_ClassLevelNetworks_v2(structCLN **cln,  structArchitecture *archHead, int nNetworkType, int nLabelID, int *nClassLevelNetworkCount, float fLearningRate, float fInitialError, float fThreshold, int nRowCount, int nColumnCount, int *nPerceptronID, int *nSynapseID, float *fInputArray, int nMode)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structArchitecture	*architecture;
	int			nWindowSize = 0;

	if (((*cln) = (structCLN *)calloc(1, sizeof(structCLN))) == NULL)
		exit(0);

	(*cln)->nNetworkType = nNetworkType;
	(*cln)->nID = (*nClassLevelNetworkCount)++;
	(*cln)->fLearningRate = fLearningRate;
	(*cln)->fInitialError = fInitialError;
	(*cln)->fThreshold = fThreshold;
	(*cln)->nLabelID = nLabelID;
	(*cln)->nSize = nRowCount * nColumnCount;
	(*cln)->nPerceptronLayerCount = 0;

	//%%%%%%%%%% Build Network %%%%%%%%%%
	for (architecture = archHead; architecture != NULL; architecture = architecture->next)
	{
		if (architecture->nLayerType == SINGLE_CONV_LAYER)
		{
			Create2DConvolveLayer_ClassLevelNetworks((*cln), architecture, fLearningRate, nMode, nRowCount, nColumnCount, nPerceptronID, nSynapseID, fInputArray);
		}
		else if (architecture->nLayerType == MULTIPLE_CONV_LAYER)
		{
			Create3DConvolveLayer_ClassLevelNetworks((*cln), architecture, fLearningRate, nMode, nRowCount, nColumnCount, nPerceptronID, nSynapseID, fInputArray);
		}
		else if (architecture->nLayerType == FULLY_CONNECTED_LAYER)
		{
			CreateConnectionLayer_ClassLevelNetworks((*cln), architecture->nKernelCount, FULLY_CONNECTED_LAYER, fLearningRate, nMode, nRowCount, nColumnCount, nPerceptronID, nSynapseID, fInputArray);

			architecture->nOutputRows = architecture->nKernelCount;
			architecture->nOutputColumns = 1;
		}
		else if (architecture->nLayerType == CLASSIFIER_LAYER)
		{
			if (nNetworkType == COMPLETE_NETWORK)
				CreateConnectionLayer_ClassLevelNetworks((*cln), nLabelID, CLASSIFIER_LAYER, fLearningRate, nMode, nRowCount, nColumnCount, nPerceptronID, nSynapseID, fInputArray);
			else if (nNetworkType == CLASS_NETWORK)
				CreateConnectionLayer_ClassLevelNetworks((*cln), 2, CLASSIFIER_LAYER, fLearningRate, nMode, nRowCount, nColumnCount, nPerceptronID, nSynapseID, fInputArray);
		}
	}

	GetClassLevelNetworkWeightCount((*cln));
	CreateMACArray_ClassLevelNetworks((*cln));

	return(nWindowSize);
}


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void CreateMaxPoolLayer_ClassLevelNetworks(structCLN* cln, structArchitecture* architecture, float fLearningRate, int nRandomMode, int nRowCount, int nColumnCount, int* nPerceptronID, int* nSynapseID, float* fInputArray)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structLayer* layerNew = NULL;
	structLayer* layerPrev = NULL;
	structPerceptron* perceptronNew = NULL;
	structPerceptron* perceptronNewHead = NULL;
	structPerceptron* perceptronHeadCur = NULL;
	structPerceptron* perceptronCur = NULL;
	structSynapse* synapseNew = NULL;
	structSynapse* synapseTail = NULL;
	structPerceptron** connections = NULL;
	int					nInputRow;
	int					nInputColumn;
	int					nKernelRow;
	int					nKernelColumn;
	int					nInputRowOffset;
	int					nInputColumnOffset;
	int					nInputOffset;
	int					nKernelOffset;
	int					nInputIndex;
	int					nKernelIndex;
	int					nRowDiff;
	int					nColumnDiff;
	int					nCount;
	int					nSynapseIndex = 0;
	int					nInputMapCount = 0;
	int					nConnectionCount = 0;
	int					nOutputArraySize = 0;
	int					bCorrect = 0;
	int					nFilter = 0;
	int					nMap = 0;
	int					nFilterOffset = 0;
	int					nMapOffset = 0;

	if ((layerNew = (structLayer*)calloc(1, sizeof(structLayer))) == NULL)
		exit(0);

	layerNew->nLayerType = MAX_POOLING_LAYER;
	layerNew->nKernelRowCount = architecture->nRowKernelSize;
	layerNew->nKernelColumnCount = architecture->nColumnKernelSize;
	layerNew->nStrideRow = architecture->nStrideRow;
	layerNew->nStrideColumn = architecture->nStrideColumn;
	layerNew->nPaddingMode = BACK_UP;
	layerNew->nActivationMode = MTANH;
	layerNew->nNumberFormat = FLOAT_POINT;

	if (cln->layerHead == NULL) // Input
	{
		layerNew->nInputRowCount = nRowCount;
		layerNew->nInputColumnCount = nColumnCount;
	}
	else
	{
		layerPrev = CalculateOutputSize(cln);

		layerNew->prev = layerPrev;
		layerNew->nInputRowCount = layerPrev->nOutputRowCount;
		layerNew->nInputColumnCount = layerPrev->nOutputColumnCount;
	}

	CalculateLayerOutputSize(layerNew);


	if (cln->layerHead != NULL)
	{
		if ((connections = (structPerceptron**)calloc(layerPrev->nOutputArraySize, sizeof(structPerceptron*))) == NULL)
			exit(0);

		for (perceptronHeadCur = layerPrev->perceptronHead, nCount = 0; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead)
		{
			for (perceptronCur = perceptronHeadCur; perceptronCur != NULL; perceptronCur = perceptronCur->next, ++nCount)
			{
				connections[perceptronCur->nIndex] = perceptronCur;
			}
		}
	}
	else
	{
		layerNew->nInputRowCount = nRowCount;
		layerNew->nInputColumnCount = nColumnCount;
		layerNew->nInputMapCount = 1;
	}

	nRowDiff = layerNew->nInputRowCount - layerNew->nKernelRowCount;
	nColumnDiff = layerNew->nInputColumnCount - layerNew->nKernelColumnCount;

	nCount = 0;
	architecture->nOutputRows = 0;
	architecture->nOutputColumns = 0;
	nConnectionCount = 0;
	nOutputArraySize = 0;

	nMapOffset = layerNew->nInputRowCount * layerNew->nInputColumnCount * layerNew->nKernelCount;
	nFilterOffset = layerNew->nInputRowCount * layerNew->nInputColumnCount;


	for (nFilter = 0; nFilter < layerNew->nKernelCount; ++nFilter)
	{
		perceptronNewHead = NULL;
	
		architecture->nOutputRows = 0;

			for (nInputRow = 0; nInputRow < (layerNew->nInputRowCount); nInputRow += layerNew->nStrideRow)
			{
				++architecture->nOutputRows;

				if (nInputRow + layerNew->nKernelRowCount >= layerNew->nInputRowCount)
					nInputRowOffset = nRowDiff * layerNew->nInputColumnCount;
				else
					nInputRowOffset = nInputRow * layerNew->nInputColumnCount;

				for (nInputColumn = 0, architecture->nOutputColumns = 0; nInputColumn < layerNew->nInputColumnCount; nInputColumn += layerNew->nStrideColumn)
				{
					++architecture->nOutputColumns;

					if (nInputColumn + layerNew->nKernelColumnCount >= layerNew->nInputColumnCount)
						nInputColumnOffset = nInputRowOffset + nColumnDiff;
					else
						nInputColumnOffset = nInputRowOffset + nInputColumn;

					perceptronNew = (structPerceptron*)calloc(1, sizeof(structPerceptron));
					perceptronNew->nID = (*nPerceptronID)++;
					perceptronNew->nLayerType = MAX_POOLING_LAYER;
					perceptronNew->nIndex = nCount++;
					perceptronNew->fLearningRate = fLearningRate;

					for (nKernelRow = 0; nKernelRow < layerNew->nKernelRowCount; ++nKernelRow)
					{
						nKernelOffset = nKernelRow * layerNew->nKernelColumnCount;
						nInputOffset = nKernelRow * layerNew->nInputColumnCount + nInputColumnOffset;

						for (nKernelColumn = 0; nKernelColumn < layerNew->nKernelColumnCount; ++nKernelColumn)
						{
							nKernelIndex = nKernelOffset + nKernelColumn;
							
							for (nMap=0; nMap< layerNew->nInputMapCount; ++nMap)
							{
								nInputIndex = nInputOffset + nKernelColumn + (nMap * nMapOffset) + (nFilter * nFilterOffset);

								printf("%d\t%d\n", nKernelIndex, nInputIndex);

								if (layerPrev == NULL)
								{
									synapseNew = (structSynapse*)calloc(1, sizeof(structSynapse));
									synapseNew->nID = (*nSynapseID)++;
									synapseNew->nIndex = nSynapseIndex++;
									synapseNew->fWeight = NULL;
									synapseNew->nInputArrayIndex = nInputIndex;

									synapseNew->fInput = &fInputArray[synapseNew->nInputArrayIndex];

									synapseNew->perceptronConnectTo = NULL;

									if (perceptronNew->synapseHead == NULL)
										AddNew_Synapse(&perceptronNew->synapseHead, synapseNew);
									else
										synapseTail->next = synapseNew;

									synapseTail = synapseNew;
									++nConnectionCount;
								}
								else
								{
									synapseNew = (structSynapse*)calloc(1, sizeof(structSynapse));
									synapseNew->nID = (*nSynapseID)++;
									synapseNew->nIndex = nSynapseIndex++;
									synapseNew->fWeight = NULL;
									synapseNew->nInputArrayIndex = -999;
									synapseNew->perceptronConnectTo = connections[nInputIndex];
									synapseNew->fInput = &synapseNew->perceptronConnectTo->fOutput;

									if (perceptronNew->synapseHead == NULL)
										AddNew_Synapse(&perceptronNew->synapseHead, synapseNew);
									else
										synapseTail->next = synapseNew;

									synapseTail = synapseNew;
									++nConnectionCount;
								}
							}
						}
					}

					perceptronNew->nWeightCount = 0;
					AddNew_Perceptron(&perceptronNewHead, perceptronNew);
					++nOutputArraySize;

					if (nInputColumn >= nColumnDiff)
						break;
				}

				if (nInputRow >= nRowDiff)
					break;
			}
		//}

		perceptronNewHead->nHeadIndex = nInputMapCount++;
		AddToLayerV2_Perceptron(&layerNew->perceptronHead, perceptronNewHead);
	}


	// Construction Check
	if (layerNew->nOutputRowCount == architecture->nOutputRows)
	{
		if (layerNew->nOutputColumnCount == architecture->nOutputColumns)
		{
			if (layerNew->nConnectionCount == nConnectionCount)
			{
				if (layerNew->nOutputArraySize == nOutputArraySize)
				{
					bCorrect = 1;
				}
			}
		}
	}

	if (!bCorrect)
	{
		//printf("\nError: CreateMaxPoolLayer_ClassLevelNetworks()\n");
		//while (1);
	}

	AddNew_Layer(&cln->layerHead, layerNew);

	for (perceptronCur = layerNew->perceptronHead; perceptronCur != NULL; perceptronCur = perceptronCur->next)
		perceptronCur->nLayerID = layerNew->nID;

	if (connections != NULL)
		free(connections);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void Create2DConvolveLayer_ClassLevelNetworks(structCLN* cln, structArchitecture* architecture, float fLearningRate, int nRandomMode, int nRowCount, int nColumnCount, int* nPerceptronID, int* nSynapseID, float* fInputArray)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structLayer			*layerNew = NULL;
	structLayer			*layerPrev = NULL;
	structPerceptron	*perceptronNew = NULL;
	structPerceptron	*perceptronTail = NULL;
	structPerceptron	*perceptronNewHead = NULL;
	structPerceptron	*perceptronCur = NULL;
	structPerceptron	*perceptronHeadCur = NULL;
	structSynapse		*synapseNew = NULL;
	structSynapse		*synapseTail = NULL;
	structPerceptron	**connections = NULL;
	int					nFilter;
	int					nInputRow;
	int					nInputColumn;
	int					nKernelRow;
	int					nKernelColumn;
	int					nInputRowOffset;
	int					nInputColumnOffset;
	int					nInputOffset;
	int					nKernelOffset;
	int					nInputIndex;
	int					nKernelIndex;
	int					nRowDiff;
	int					nColumnDiff;
	int					nCount;
	int					nSynapseIndex;
	int					nWeightsPerFilter;
	int					nInputMapCount = 0;
	int					nMap = 0;
	int					nConnectionCount = 0;
	int					nOutputArraySize = 0;
	int					nFilterOffset = 0;
	int					bCorrect = 0;

	if ((layerNew = (structLayer*)calloc(1, sizeof(structLayer))) == NULL)
		exit(0);

	layerNew->nLayerType = SINGLE_CONV_LAYER;
	layerNew->nKernelCount = architecture->nKernelCount;
	layerNew->nKernelCount = architecture->nKernelCount;
	layerNew->nKernelRowCount = architecture->nRowKernelSize;
	layerNew->nKernelColumnCount = architecture->nColumnKernelSize;
	layerNew->nStrideRow = architecture->nStrideRow;
	layerNew->nStrideColumn = architecture->nStrideColumn;
	layerNew->nPaddingMode = BACK_UP;
	layerNew->nActivationMode = MTANH;
	layerNew->nNumberFormat = FLOAT_POINT;

	if (cln->layerHead == NULL) // Input
	{
		layerNew->nInputRowCount = nRowCount;
		layerNew->nInputColumnCount = nColumnCount;

	}
	else
	{
		layerPrev = CalculateOutputSize(cln);

		layerNew->prev = layerPrev;
		layerNew->nInputRowCount = layerPrev->nOutputRowCount;
		layerNew->nInputColumnCount = layerPrev->nOutputColumnCount;

	}

	CalculateLayerOutputSize(layerNew);


	if (cln->layerHead != NULL)
	{
		nCount = 0;
		for (perceptronHeadCur = layerPrev->perceptronHead; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead)
		{
			for (perceptronCur = perceptronHeadCur; perceptronCur != NULL; perceptronCur = perceptronCur->next)
			{
				if(perceptronCur->nIndex > nCount)
				nCount = perceptronCur->nIndex;
			}
		}
		
		++nCount;

		// Create Connect to Array
		//if ((connections = (structPerceptron**)calloc(layerPrev->nOutputArraySize, sizeof(structPerceptron*))) == NULL)
		//	exit(0);
		if ((connections = (structPerceptron**)calloc(nCount, sizeof(structPerceptron*))) == NULL)
			exit(0);

		nCount = 0;
		for (perceptronHeadCur = layerPrev->perceptronHead; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead)
		{
			for (perceptronCur = perceptronHeadCur; perceptronCur != NULL; perceptronCur = perceptronCur->next)
			{
				connections[perceptronCur->nIndex] = perceptronCur;
				nCount++;
			}
		}

	}

	nRowDiff = layerNew->nInputRowCount - layerNew->nKernelRowCount;
	nColumnDiff = layerNew->nInputColumnCount - layerNew->nKernelColumnCount;

	// Create Weight Array
	nWeightsPerFilter = layerNew->nKernelRowCount * layerNew->nKernelColumnCount + 1;
	layerNew->nWeightCount = layerNew->nKernelCount * nWeightsPerFilter * layerNew->nInputMapCount;
	layerNew->fWeightArray = (float*)calloc(layerNew->nWeightCount, sizeof(float));
	cln->nWeightCount += layerNew->nWeightCount;


	nCount = 0;
	architecture->nOutputRows = 0;
	architecture->nOutputColumns = 0;
	nConnectionCount = 0;
	nOutputArraySize = 0;



	for (nFilter = 0; nFilter < layerNew->nKernelCount; ++nFilter)
	{
		nFilterOffset = nFilter * layerNew->nInputMapCount * nWeightsPerFilter;

		for (int nMap=0; nMap < layerNew->nInputMapCount; ++nMap)
		{
			perceptronNewHead = NULL;
			architecture->nOutputRows = 0;

			for (nInputRow = 0; nInputRow < layerNew->nInputRowCount; nInputRow += layerNew->nStrideRow)
			{
				++architecture->nOutputRows;

				if (nInputRow + layerNew->nKernelRowCount >= layerNew->nInputRowCount)
					nInputRowOffset = nRowDiff * layerNew->nInputColumnCount;
				else
					nInputRowOffset = nInputRow * layerNew->nInputColumnCount;

				for (nInputColumn = 0, architecture->nOutputColumns = 0; nInputColumn < layerNew->nInputColumnCount; nInputColumn += layerNew->nStrideColumn)
				{
					++architecture->nOutputColumns;

					if (nInputColumn + layerNew->nKernelColumnCount >= layerNew->nInputColumnCount)
						nInputColumnOffset = nInputRowOffset + nColumnDiff;
					else
						nInputColumnOffset = nInputRowOffset + nInputColumn;

					perceptronNew = (structPerceptron*)calloc(1, sizeof(structPerceptron));
					perceptronNew->nID = (*nPerceptronID)++;
					perceptronNew->nLayerType = SINGLE_CONV_LAYER;
				
					perceptronNew->nIndex = nCount++;
					perceptronNew->fLearningRate = fLearningRate;

					//nSynapseIndex = nFilter * nWeightsPerFilter;
					nSynapseIndex = nMap * nWeightsPerFilter + nFilterOffset;

					//Add Bias Synapse ////////////////////////////////////////////////////////////////////////
					perceptronNew->synapseHead = (structSynapse*)calloc(1, sizeof(structSynapse));
					perceptronNew->synapseHead->nID = (*nSynapseID)++;
					perceptronNew->synapseHead->nIndex = nSynapseIndex++;
					perceptronNew->synapseHead->nInputArrayIndex = -1;
					perceptronNew->synapseHead->fWeight = &layerNew->fWeightArray[perceptronNew->synapseHead->nIndex];
					perceptronNew->synapseHead->fInput = (float*)calloc(1, sizeof(float));
					*(perceptronNew->synapseHead->fInput) = 1; // Bias

					
					

					synapseTail = perceptronNew->synapseHead;

					for (nKernelRow = 0; nKernelRow < layerNew->nKernelRowCount; ++nKernelRow)
					{
						nKernelOffset = nKernelRow * layerNew->nKernelColumnCount;
						nInputOffset = nKernelRow * layerNew->nInputColumnCount + nInputColumnOffset;

						for (nKernelColumn = 0; nKernelColumn < layerNew->nKernelColumnCount; ++nKernelColumn)
						{
							nInputIndex = nInputOffset + nKernelColumn + (nMap * layerNew->nInputRowCount * layerNew->nInputColumnCount);
							nKernelIndex = nKernelOffset + nKernelColumn;

							if (layerPrev == NULL)
							{
								synapseNew = (structSynapse*)calloc(1, sizeof(structSynapse));
								synapseNew->nID = (*nSynapseID)++;
								//printf("Synapse:\nnID: %d\n", synapseNew->nID);
								synapseNew->nIndex = nSynapseIndex++;
								//printf("nIndex: %d\n", synapseNew->nIndex);
								synapseNew->fWeight = &layerNew->fWeightArray[synapseNew->nIndex];
								//printf("fWeight: %f\n", layerNew->fWeightArray[synapseNew->nIndex]);
								synapseNew->nInputArrayIndex = nInputIndex;
								//printf("nInputArrayIndex: %d\n", synapseNew->nInputArrayIndex);
								synapseNew->fInput = &fInputArray[synapseNew->nInputArrayIndex];
								//printf("fInput: %f\n", fInputArray[synapseNew->nInputArrayIndex]);
								synapseNew->perceptronConnectTo = NULL;
								++nConnectionCount;
								synapseTail->next = synapseNew;
								synapseTail = synapseNew;
								//printf("\n");
							}
							else
							{
								synapseNew = (structSynapse*)calloc(1, sizeof(structSynapse));
								synapseNew->nID = (*nSynapseID)++;
								synapseNew->nIndex = nSynapseIndex++;
								synapseNew->fWeight = &layerNew->fWeightArray[synapseNew->nIndex];
								synapseNew->nInputArrayIndex = -999;
								synapseNew->perceptronConnectTo = connections[nInputIndex];
								synapseNew->fInput = &synapseNew->perceptronConnectTo->fOutput;

								++nConnectionCount;
								synapseTail->next = synapseNew;
								synapseTail = synapseNew;
							}
						}
					}

					perceptronNew->nWeightCount = nSynapseIndex;
				
					++nOutputArraySize;
					
					if (perceptronNewHead == NULL)
					{
						perceptronNewHead = perceptronNew;
					}
					else
					{
						perceptronTail->next = perceptronNew;
					}
					
					perceptronTail = perceptronNew;

					if (nInputColumn >= nColumnDiff)
						break;
				}

				if (nInputRow >= nRowDiff)
					break;
			}

		perceptronNewHead->nHeadIndex = nInputMapCount++;
		AddToLayerV2_Perceptron(&layerNew->perceptronHead, perceptronNewHead);
		}
	}

	layerNew->nOutputRowCount = architecture->nOutputRows;
	layerNew->nOutputColumnCount = architecture->nOutputColumns;
	layerNew->nConnectionCount = nConnectionCount;
	layerNew->nOutputArraySize = nOutputArraySize;

	// Construction Check
	if (layerNew->nOutputRowCount == architecture->nOutputRows)
	{
		if (layerNew->nOutputColumnCount == architecture->nOutputColumns)
		{
			if (layerNew->nConnectionCount == nConnectionCount)
			{
				if (layerNew->nOutputArraySize == nOutputArraySize)
				{
					bCorrect = 1;
				}
			}
		}
	}
	

	if (!bCorrect)
	{
		printf("\nError: Create2DConvolveLayer_ClassLevelNetworks()\n");
		while (1);
	}

	AddNew_Layer(&cln->layerHead, layerNew);

	for (perceptronCur = layerNew->perceptronHead; perceptronCur != NULL; perceptronCur = perceptronCur->next)
		perceptronCur->nLayerID = layerNew->nID;

	if (connections != NULL)
		free(connections);
}



/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void Create3DConvolveLayer_ClassLevelNetworks(structCLN *cln, structArchitecture *architecture, float fLearningRate, int nRandomMode, int nRowCount, int nColumnCount, int *nPerceptronID, int *nSynapseID, float *fInputArray)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structLayer			*layerNew;
	structLayer			*layerPrev;
	structPerceptron	*perceptronSubSetCur = NULL;
	structPerceptron	*perceptronNew = NULL;
	structPerceptron	*perceptronNewHead = NULL;
	structPerceptron	*perceptronCur = NULL;
	structSynapse		*synapseNew = NULL;
	int					nSynapseIndex;
	int					nIndex;
	int					nCount;
	int					nXStep;
	int					nYStep;
	int					nRowStep;
	int					nColumnStep;
	int					nRowStepDelta;
	int					nColumnStepDelta;
	int					i, j, k, l;
	int					nSize = architecture->nRowKernelSize * architecture->nColumnKernelSize;
	int					nFC;
	int					nRows;
	int					nColumns;
	int					nInputMapCount;
	int					nMemoryOffset;
	int					nMapX;
	int					nMapY;
	int					nIDOffset;
	int					nOffset;
	int					nOffsetStep;
	int					nSubSetOverlap = architecture->nRowKernelSize - architecture->nStrideRow;

	if ((layerNew = (structLayer *)calloc(1, sizeof(structLayer))) == NULL)
		exit(0);

	for (layerPrev = cln->layerHead; layerPrev->next != NULL; layerPrev = layerPrev->next);

	nRows = layerPrev->nOutputRowCount;
	nColumns = layerPrev->nOutputColumnCount;
	nOffsetStep = (nRows * nColumns);

	layerNew->nLayerType = MULTIPLE_CONV_LAYER;
	layerNew->nPerceptronCount = 0;
	layerNew->nInputRowCount = nRows;
	layerNew->nInputColumnCount = nColumns;
	layerNew->nKernelCount = architecture->nKernelCount;
	layerNew->nKernelRowCount = architecture->nRowKernelSize;
	layerNew->nKernelColumnCount = architecture->nColumnKernelSize;
	layerNew->nStrideRow = architecture->nStrideRow;
	layerNew->nStrideColumn = architecture->nStrideColumn;
	layerNew->nPaddingMode = BACK_UP;
	layerNew->nActivationMode = MTANH;
	layerNew->nNumberFormat = FLOAT_POINT;
	layerNew->nInputMapCount = layerPrev->nKernelCount;

	for (perceptronSubSetCur = layerPrev->perceptronHead, nInputMapCount = 0; perceptronSubSetCur != NULL; perceptronSubSetCur = perceptronSubSetCur->nextHead, ++nInputMapCount);


	if (nRows <= layerNew->nKernelRowCount) // Fully Connected
	{
		nXStep = 0;
		layerNew->nKernelRowCount = nRows;
		nRowStep = nRows;
		nRowStepDelta = nRows;
	}
	else
	{
		if (nSubSetOverlap >= layerNew->nKernelRowCount)
			nSubSetOverlap = 0;

		nXStep = layerNew->nKernelRowCount - nSubSetOverlap;


		nRowStep = nRows - nSubSetOverlap;
		nRowStepDelta = nRowStep - nXStep;
	}


	if (nColumns <= layerNew->nKernelColumnCount) // Fully Connected
	{
		nYStep = 0;
		layerNew->nKernelColumnCount = 1;
		nColumnStep = 1;
		nColumnStepDelta = 1;
	}
	else
	{
		if (nSubSetOverlap >= layerNew->nKernelColumnCount)
			nSubSetOverlap = 0;

		nYStep = layerNew->nKernelColumnCount - nSubSetOverlap;
		nColumnStep = nColumns - nSubSetOverlap;
		nColumnStepDelta = nColumnStep - nYStep;
	}


	// Create Weight Array
	nMemoryOffset = (nInputMapCount * (layerNew->nKernelRowCount * layerNew->nKernelColumnCount) + 1);
	layerNew->nWeightCount = layerNew->nKernelCount * nMemoryOffset;
	layerNew->fWeightArray = (float *)calloc(layerNew->nWeightCount, sizeof(float));
	cln->nWeightCount += layerNew->nWeightCount;

	// Build Layer
	nInputMapCount = 0;
	nMapX = 0;
	nMapY = 0;
	nIDOffset = layerPrev->perceptronHead->nID;
	for (nFC = 0; nFC<layerNew->nKernelCount; ++nFC)
	{
		nCount = 0;
		i = 0;

		nMapX = 0;
		while (i < nRowStep)
		{
			nMapY = 0;
			j = 0;
			++nMapX;
			while (j < nColumnStep)
			{
				++nMapY;

				perceptronNew = (structPerceptron *)calloc(1, sizeof(structPerceptron));
				perceptronNew->nID = (*nPerceptronID)++;

				perceptronNew->nLayerType = MULTIPLE_CONV_LAYER;
				perceptronNew->nIndex = nCount++;
				perceptronNew->fLearningRate = fLearningRate;

				nSynapseIndex = nFC * nMemoryOffset;

				perceptronNew->fBias = layerNew->fWeightArray[nSynapseIndex];

				//Add Bias Synapse ////////////////////////////////////////////////////////////////////////
				synapseNew = (structSynapse *)calloc(1, sizeof(structSynapse));
				synapseNew->nID = (*nSynapseID)++;
				synapseNew->nInputArrayIndex = -1;

				synapseNew->fInput = (float *)calloc(1, sizeof(float));
				*(synapseNew->fInput) = 1; // Bias
				synapseNew->nIndex = nSynapseIndex++;
				synapseNew->fWeight = &layerNew->fWeightArray[synapseNew->nIndex];

				++layerNew->nConnectionCount;
				AddNew_Synapse(&perceptronNew->synapseHead, synapseNew);
				////////////////////////////////////////////////////////////////////////

				for (k = 0; k<layerNew->nKernelRowCount; ++k)
				{
					for (l = 0; l<layerNew->nKernelColumnCount; ++l)
					{
						for (perceptronSubSetCur = layerPrev->perceptronHead, nOffset = 0; perceptronSubSetCur != NULL; perceptronSubSetCur = perceptronSubSetCur->nextHead, nOffset += nOffsetStep)
						{
							synapseNew = (structSynapse *)calloc(1, sizeof(structSynapse));

							nIndex = (((i + k) * nRows) + (j + l)) + nOffset;

							for (perceptronCur = perceptronSubSetCur; perceptronCur != NULL; perceptronCur = perceptronCur->next)
							{
								if ((perceptronCur->nID - nIDOffset) == nIndex)
								{
									synapseNew = (structSynapse *)calloc(1, sizeof(structSynapse));
									synapseNew->nID = (*nSynapseID)++;
									synapseNew->nIndex = nSynapseIndex++;
									synapseNew->fWeight = &layerNew->fWeightArray[synapseNew->nIndex];
									synapseNew->nInputArrayIndex = -999;
									synapseNew->fInput = &perceptronCur->fOutput;
									synapseNew->perceptronConnectTo = perceptronCur;

									AddNew_Synapse(&perceptronNew->synapseHead, synapseNew);

									break;
								}
							}

						}
					}
				}

				++layerNew->nPerceptronCount;
				perceptronNew->nWeightCount = nSynapseIndex;
				AddNewV2_Perceptron(&perceptronNewHead, perceptronNew);

				if ((j != nColumnStepDelta) && (j + nYStep + nSubSetOverlap > (nColumns - nYStep)))
					j = nColumnStepDelta;
				else
					j += nYStep;
			}

			if ((i != nRowStepDelta) && (i + nXStep + nSubSetOverlap > (nRows - nXStep)))
				i = nRowStepDelta;
			else
				i += nXStep;
		}

		perceptronNewHead->nDimX = nMapX;
		perceptronNewHead->nDimY = nMapY;
		perceptronNewHead->nIndex = nInputMapCount;
		perceptronNewHead->nHeadIndex = nInputMapCount;
		perceptronNewHead->nWeightCount = nSynapseIndex;

		AddToLayer_Perceptron(&layerNew->perceptronHead, perceptronNewHead, nInputMapCount++);
	}

	layerNew->nConnectionCount = (nSynapseIndex - 1);
	layerNew->nOutputArraySize = perceptronNewHead->nDimX * perceptronNewHead->nDimY * layerNew->nKernelCount;

	architecture->nOutputRows = perceptronNewHead->nDimY * layerNew->nKernelCount;
	architecture->nOutputColumns = perceptronNewHead->nDimX;

	layerNew->nOutputRowCount = architecture->nOutputRows;
	layerNew->nOutputColumnCount = architecture->nOutputColumns;

	AddNew_Layer(&cln->layerHead, layerNew);

	for (perceptronCur = layerNew->perceptronHead; perceptronCur != NULL; perceptronCur = perceptronCur->next)
		perceptronCur->nLayerID = layerNew->nID;

	return;
}




/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void AddNew_ClassLevelNetworks(structCLN **head, structCLN *newCLN)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structCLN *cur = *head;

	newCLN->nID = 0;

	if (*head == NULL)
	{
		newCLN->nID = 0;
		*head = newCLN;
	}
	else
	{
		while (cur->next != NULL)
		{
			cur = cur->next;
			newCLN->nID = cur->nID;
		}

		++newCLN->nID;
		cur->next = newCLN;
		newCLN->prev = cur;
	}

	return;
}


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
structCLN *DeleteCLN_ClassLevelNetworks(structCLN **head, int nID)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structCLN	*cur = NULL;
	structCLN	*prev = NULL;
	int			i;

	if ((*head)->nID == nID)
	{
		if ((*head)->next != NULL)
			cur = (*head)->next;

		DeleteAll_Layer(&(*head)->layerHead);
		free((*head)->macData);

		for (i = 0; i < (*head)->nClassCount; ++i)
			free((*head)->fOutputArray[i]);
		free((*head)->fOutputArray);

		free((*head)->nStartArray);
		free((*head)->nEndArray);

		(*head) = cur;

		return(*head);
	}
	else
	{
		for (cur = *head; cur != NULL; prev = cur, cur = cur->next)
		{
			if (cur->nID == nID)
			{
				DeleteAll_Layer(&cur->layerHead);
				free(cur->macData);
				
				for (i = 0; i < cur->nClassCount; ++i)
					free(cur->fOutputArray[i]);
				free(cur->fOutputArray);

				free((*head)->nStartArray);
				free((*head)->nEndArray);

				if (prev == NULL)
				{
					*head = cur->next;

					return(*head);
				}
				else
				{
					prev->next = cur->next;

					return(prev);
				}
			}
		}
	}

	return(NULL);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void DeleteCLN_V2_ClassLevelNetworks(structCLN **head, int nID)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structCLN	*cur = NULL;
	structCLN	*prev = NULL;
	int			i;

	if ((*head)->nID == nID)
	{
		DeleteAll_Layer(&(*head)->layerHead);

		free((*head)->macData);

		for (i = 0; i < (*head)->nClassCount; ++i)
			free((*head)->fOutputArray[i]);
		free((*head)->fOutputArray);

		free((*head)->nStartArray);
		free((*head)->nEndArray);

		if ((*head)->next != NULL)
			cur = (*head)->next;


		(*head) = cur;
	}
	else
	{
		for (cur = *head; cur != NULL; prev = cur, cur = cur->next)
		{
			if (cur->nID == nID)
			{
				DeleteAll_Layer(&cur->layerHead);

				free(cur->macData);
				
				for (i = 0; i < cur->nClassCount; ++i)
					free(cur->fOutputArray[i]);
				free(cur->fOutputArray);

				free((*head)->nStartArray);
				free((*head)->nEndArray);

				if (prev == NULL)
				{
					*head = cur->next;
				}
				else
				{
					prev->next = cur->next;
				}
			}
		}
	}

	return;
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void DeleteAll_ClassLevelNetworks(structCLN **head)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structCLN	*cur = *head;
	structCLN	*next = NULL;
	int			i;

	while (cur != NULL)
	{
		DeleteAll_Layer(&cur->layerHead);
		FreeArray_MAC(&cur->macData, &cur->nMACCount);

		for (i = 0; i < cur->nClassCount; ++i)
			free(cur->fOutputArray[i]);
		free(cur->fOutputArray);

		free((*head)->nStartArray);
		free((*head)->nEndArray);

		next = cur->next;
		cur = next;
	}

	*head = NULL;
}


/*

structLayer			*layerHead;
structLayer			*layerClassifier;
structPerceptron	*perceptronClassifier;
structMAC			*macData;

float				fLearningRate;
float				fInitialError;
float				fThreshold;
float				fThresholdPercent;
float				fAccuracy;
int					nID;
int					nNetworkType;
int					nLabelID;
int					nSize;
int					nPerceptronLayerCount;
int					nWeightCount;
int					nMACCount;
int					nClassCount;
int					bKeep;

int					bAdjustGlobalLearningRate;
float				fLearningRateMinimum;
float				fLearningRateMaximum;
int					bAdjustPerceptronLearningRate;
int					bAdjustThreshold;
float				fRatioAverage;
float				fPercentBackProp;
int					*nStartArray;
int					*nEndArray;
int					nLayerCount;
float				**fOutputArray;

struct structCLN		*next;
struct structCLN		*prev;


*/

















/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void CreateMACArray_ClassLevelNetworks(structCLN *cln)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structLayer			*layerCur;
	structPerceptron	*perceptronHeadCur = NULL;
	structPerceptron	*perceptronHead = NULL;
	structPerceptron	*perceptronCur;
	structSynapse		*synapseCur;
	structSynapse		*synapseCluster;
	int					nPerceptronCount = 0;
	int					nSynapseCount = 0;
	int					nKernel = 0;
	int					nInputMapSize;
	int					nRowDiff;
	int					nColumnDiff;
	int					nWeightsPerKernel;
	int					nWeightCount;
	int					nKernelInputOffset;
	int					nExactRowPosition;
	int					nWindowSize;
	int					nKernelOffset;
	int					nInputRow;
	int					nInputRowOffset;
	int					nInputColumn;
	int					nInputColumnOffset;
	int					nKernelRow;
	int					nInputOffset;
	int					nKernelRowOffset;
	int					nKernelColumn;
	int					nKernelColumnOffset;
	int					nKernelColumnInputOffset;
	int					nInputMap;
	int					nInputIndex;
	int					nKernelIndex;
	int					nOutputCount;
	int					nClassifierID;
	int					nInputCount;


	//Delete Old MAC array
	FreeArray_MAC(&cln->macData, &cln->nMACCount);

	for (layerCur = cln->layerHead, nPerceptronCount = 0; layerCur != NULL; layerCur = layerCur->next)
	{
		if (layerCur->perceptronHead != NULL)
		{
			for (perceptronHeadCur = layerCur->perceptronHead; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead)
			{
				for (perceptronCur = perceptronHeadCur; perceptronCur != NULL; perceptronCur = perceptronCur->next, ++nPerceptronCount)
				{
					for (synapseCur = perceptronCur->synapseHead; synapseCur != NULL; synapseCur = synapseCur->next, ++nSynapseCount);
				}
			}
		}
		else
		{
			nPerceptronCount += layerCur->nOutputArraySize;
			nSynapseCount += layerCur->nConnectionCount;
		}
	}

	if ((cln->macData = (structMAC *)calloc(nPerceptronCount, sizeof(structMAC))) == NULL)
	{
		HoldDisplay("memory Error: void CreateArray(structNetwork *networkMain, structMAC **macData)\n");
	}

	nClassifierID = 0;
	for (layerCur = cln->layerHead, nPerceptronCount = 0; layerCur != NULL; layerCur = layerCur->next)
	{
		if (layerCur->perceptronHead != NULL)
		{
			for (perceptronHeadCur = layerCur->perceptronHead; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead)
			{
				for (perceptronCur = perceptronHeadCur; perceptronCur != NULL; perceptronCur = perceptronCur->next, ++nPerceptronCount)
				{
					for (synapseCur = perceptronCur->synapseHead, nSynapseCount = 0; synapseCur != NULL; synapseCur = synapseCur->next, ++nSynapseCount);

					if ((cln->macData[nPerceptronCount].fInputArray = (float ***)calloc(nSynapseCount, sizeof(float **))) == NULL)
					{
						HoldDisplay("memory Error: void CreateArray(structNetwork *networkMain, structMAC **macData)\n");
					}
					if ((cln->macData[nPerceptronCount].fInputSum = (float **)calloc(nSynapseCount, sizeof(float *))) == NULL)
					{
						HoldDisplay("memory Error: void CreateArray(structNetwork *networkMain, structMAC **macData)\n");
					}
					
					
					
					if ((cln->macData[nPerceptronCount].fInput = (float **)calloc(nSynapseCount, sizeof(float *))) == NULL)
					{
						HoldDisplay("memory Error: void CreateArray(structNetwork *networkMain, structMAC **macData)\n");
					}

					if ((cln->macData[nPerceptronCount].fWeight = (float **)calloc(nSynapseCount, sizeof(float *))) == NULL)
					{
						HoldDisplay("memory Error: void CreateArray(structNetwork *networkMain, structMAC **macData)\n");
					}

					if ((cln->macData[nPerceptronCount].fConnectToDifferential = (float **)calloc(nSynapseCount, sizeof(float *))) == NULL)
					{
						HoldDisplay("memory Error: void CreateArray(structNetwork *networkMain, structMAC **macData)\n");
					}

					if ((cln->macData[nPerceptronCount].nConnectFromID = (int *)calloc(nSynapseCount, sizeof(int))) == NULL)
					{
						HoldDisplay("memory Error: void CreateArray(structNetwork *networkMain, structMAC **macData)\n");
					}

					if ((cln->macData[nPerceptronCount].fAverage = (float *)calloc(nSynapseCount, sizeof(float))) == NULL)
					{
						HoldDisplay("memory Error: void CreateArray(structNetwork *networkMain, structMAC **macData)\n");
					}

					if ((cln->macData[nPerceptronCount].nAverageCount = (int *)calloc(nSynapseCount, sizeof(int))) == NULL)
					{
						HoldDisplay("memory Error: void CreateArray(structNetwork *networkMain, structMAC **macData)\n");
					}

					if ((cln->macData[nPerceptronCount].fSumSquares = (float *)calloc(nSynapseCount, sizeof(float))) == NULL)
					{
						HoldDisplay("memory Error: void CreateArray(structNetwork *networkMain, structMAC **macData)\n");
					}

					if ((cln->macData[nPerceptronCount].nInputCount = (int *)calloc(nSynapseCount, sizeof(int))) == NULL)
					{
						HoldDisplay("memory Error: void CreateArray(structNetwork *networkMain, structMAC **macData)\n");
					}

					for (synapseCur = perceptronCur->synapseHead, cln->macData[nPerceptronCount].nCount = 0; synapseCur != NULL; synapseCur = synapseCur->next)
					{
						if (synapseCur->bAdjust == 1)
							continue;

						cln->macData[nPerceptronCount].fWeight[cln->macData[nPerceptronCount].nCount] = synapseCur->fWeight;
						cln->macData[nPerceptronCount].fInput[cln->macData[nPerceptronCount].nCount] = synapseCur->fInput;
						
						if (synapseCur != perceptronCur->synapseHead && perceptronCur->nClusterCount > 1)
						{
							for (synapseCluster = perceptronCur->synapseHead->next, nInputCount=0; synapseCluster != NULL; synapseCluster = synapseCluster->next)
							{
								if ((synapseCluster != synapseCur) && (synapseCluster->nCluster == synapseCur->nCluster))
									++nInputCount;
							}

							if (nInputCount > 0)
							{
								cln->macData[nPerceptronCount].nInputCount[cln->macData[nPerceptronCount].nCount] = nInputCount;
								
								if ((cln->macData[nPerceptronCount].fInputArray[cln->macData[nPerceptronCount].nCount] = (float **)calloc(nInputCount, sizeof(float*))) == NULL)
								{
									HoldDisplay("memory Error: void CreateArray(structNetwork *networkMain, structMAC **macData)\n");
								}
								
								for (synapseCluster = perceptronCur->synapseHead->next, nInputCount = 0; synapseCluster != NULL; synapseCluster = synapseCluster->next)
								{
									if ((synapseCluster != synapseCur) && (synapseCluster->nCluster == synapseCur->nCluster))
									{
										cln->macData[nPerceptronCount].fInputArray[cln->macData[nPerceptronCount].nCount][nInputCount++] = synapseCluster->fInput;
										synapseCluster->bAdjust = 1;
									}
								}
							}
						}
						

						if (synapseCur->perceptronConnectTo != NULL)
						{
							cln->macData[nPerceptronCount].fConnectToDifferential[cln->macData[nPerceptronCount].nCount] = &synapseCur->perceptronConnectTo->fDifferential;
							cln->macData[nPerceptronCount].nConnectFromID[cln->macData[nPerceptronCount].nCount] = synapseCur->perceptronConnectTo->nID;
						}
						else
						{
							cln->macData[nPerceptronCount].fConnectToDifferential[cln->macData[nPerceptronCount].nCount] = NULL;
							cln->macData[nPerceptronCount].nConnectFromID[cln->macData[nPerceptronCount].nCount] = -1;
						}

						++cln->macData[nPerceptronCount].nCount;
					}

					cln->macData[nPerceptronCount].fFeedBackWeight = -MULTIPLIER * UNIFORM_PLUS_MINUS_ONE;
					cln->macData[nPerceptronCount].fOutput = &perceptronCur->fOutput;
					cln->macData[nPerceptronCount].fDifferential = &perceptronCur->fDifferential;
					cln->macData[nPerceptronCount].fLearningRate = &perceptronCur->fLearningRate;
					cln->macData[nPerceptronCount].nLayerType = layerCur->nLayerType;
					cln->macData[nPerceptronCount].nKernelID = perceptronHeadCur->nIndex;

					if (cln->macData[nPerceptronCount].nLayerType == CLASSIFIER_LAYER)
						cln->macData[nPerceptronCount].nID = nClassifierID++;
				}
			}
		}
		else
		{
			nInputMapSize = layerCur->nInputRowCount * layerCur->nInputColumnCount;
			nRowDiff = layerCur->nInputRowCount - layerCur->nKernelRowCount;
			nColumnDiff = layerCur->nInputColumnCount - layerCur->nKernelColumnCount;
			nWeightsPerKernel = layerCur->nKernelRowCount * layerCur->nKernelColumnCount * layerCur->nInputMapCount + 1;
			nWeightCount = layerCur->nKernelCount * nWeightsPerKernel;
			nKernelInputOffset = layerCur->nKernelColumnCount * layerCur->nInputMapCount;
			nExactRowPosition = nRowDiff * layerCur->nInputColumnCount;   // Calculate exact row position for last kernel
			nWindowSize = CalculateWindowSize(layerCur->nInputColumnCount, layerCur->nKernelColumnCount, layerCur->nStrideColumn);
			nOutputCount = 0;

			for (nKernel = 0; nKernel < layerCur->nKernelCount; ++nKernel)
			{
				nKernelOffset = nKernel * nWeightsPerKernel;

				nInputRow = 0;
				while ((nInputRow - layerCur->nStrideRow) < nRowDiff)
				{
					if ((nInputRow + layerCur->nKernelRowCount) > layerCur->nInputRowCount)		// Kernel exceeds input boundry
						nInputRowOffset = nExactRowPosition;				// Set exact row position for last kernel
					else
						nInputRowOffset = nInputRow * layerCur->nInputColumnCount;

					nInputColumn = 0;
					while ((nInputColumn - layerCur->nStrideColumn) < nColumnDiff)
					{
						if ((nInputColumn + layerCur->nKernelColumnCount) > layerCur->nInputColumnCount)	// Kernel exceeds input boundry
							nInputColumnOffset = nInputRowOffset + nColumnDiff;			// Calculate exact position for last kernel
						else
							nInputColumnOffset = nInputRowOffset + nInputColumn;


						////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////						
						if ((cln->macData[nPerceptronCount].fInput = (float **)calloc(nWeightsPerKernel, sizeof(float *))) == NULL)
						{
							HoldDisplay("memory Error: void CreateArray(structNetwork *networkMain, structMAC **macData)\n");
						}

						if ((cln->macData[nPerceptronCount].fWeight = (float **)calloc(nWeightsPerKernel, sizeof(float *))) == NULL)
						{
							HoldDisplay("memory Error: void CreateArray(structNetwork *networkMain, structMAC **macData)\n");
						}

						if ((cln->macData[nPerceptronCount].fConnectToDifferential = (float **)calloc(nWeightsPerKernel, sizeof(float *))) == NULL)
						{
							HoldDisplay("memory Error: void CreateArray(structNetwork *networkMain, structMAC **macData)\n");
						}

						if ((cln->macData[nPerceptronCount].fAverage = (float *)calloc(nSynapseCount, sizeof(float))) == NULL)
						{
							HoldDisplay("memory Error: void CreateArray(structNetwork *networkMain, structMAC **macData)\n");
						}

						if ((cln->macData[nPerceptronCount].fSumSquares = (float *)calloc(nSynapseCount, sizeof(float))) == NULL)
						{
							HoldDisplay("memory Error: void CreateArray(structNetwork *networkMain, structMAC **macData)\n");
						}

						if ((cln->macData[nPerceptronCount].nAverageCount = (int *)calloc(nSynapseCount, sizeof(int))) == NULL)
						{
							HoldDisplay("memory Error: void CreateArray(structNetwork *networkMain, structMAC **macData)\n");
						}



						nKernelIndex = nKernelOffset;
						nKernelIndex = 0;
						cln->macData[nPerceptronCount].fWeight[cln->macData[nPerceptronCount].nCount] = &layerCur->fWeightArray[nKernelIndex]; // synapseCur->fWeight;
						cln->macData[nPerceptronCount].fInput[cln->macData[nPerceptronCount].nCount] = NULL; // synapseCur->fInput;

						for (nKernelRow = 0, ++cln->macData[nPerceptronCount].nCount; nKernelRow < layerCur->nKernelRowCount; ++nKernelRow)
						{
							nInputOffset = (nKernelRow * layerCur->nInputColumnCount) + nInputColumnOffset;
							nKernelRowOffset = (nKernelRow * nKernelInputOffset) + nKernelOffset;

							for (nKernelColumn = 0; nKernelColumn < layerCur->nKernelColumnCount; ++nKernelColumn)
							{
								nKernelColumnOffset = (nKernelColumn * layerCur->nInputMapCount) + nKernelRowOffset;
								nKernelColumnInputOffset = nInputOffset + nKernelColumn;

								for (nInputMap = 0; nInputMap < layerCur->nInputMapCount; ++nInputMap, ++cln->macData[nPerceptronCount].nCount)
								{
									nInputIndex = (nInputMap * nInputMapSize) + nKernelColumnInputOffset;
									nKernelIndex = nInputMap + nKernelColumnOffset + 1;

									cln->macData[nPerceptronCount].fWeight[cln->macData[nPerceptronCount].nCount] = &layerCur->fWeightArray[nKernelIndex]; // synapseCur->fWeight;
									cln->macData[nPerceptronCount].fInput[cln->macData[nPerceptronCount].nCount] = layerCur->fInputArray[nInputIndex]; // synapseCur->fInput;

									//if (layerCur->prev != NULL)
									//	cln->macData[nPerceptronCount].fConnectToDifferential[cln->macData[nPerceptronCount].nCount] = &layerCur->prev->fDifferentialArray[nKernelIndex];
								}
							}
						}

						cln->macData[nPerceptronCount].fOutput = &layerCur->fOutputArray[nOutputCount];
						//cln->macData[nPerceptronCount].fDifferential = &layerCur->fDifferentialArray[nOutputCount];
	//					cln->macData[nPerceptronCount].fLearningRate = &layerCur->//fLearningRateArray[nOutputCount];
						cln->macData[nPerceptronCount].nLayerType = layerCur->nLayerType;
						cln->macData[nPerceptronCount].nKernelID = nKernel;

						if (cln->macData[nPerceptronCount].nLayerType == CLASSIFIER_LAYER)
							cln->macData[nPerceptronCount].nID = nClassifierID++;

						nInputColumn += layerCur->nStrideColumn;
						++nPerceptronCount;
						++nOutputCount;
					}

					nInputRow += layerCur->nStrideRow;
				}
			}
		}
	}

	cln->nMACCount = nPerceptronCount;
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
float GetMedian_ClassLevelNetworks(structCLN *clnHead)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structCLN	*clnCur;
	float		fMedian = 0.0f;
	float		*fArray;
	float		fTemp;
	int			nCount;
	int			i, j;

	for (clnCur = clnHead, nCount = 0; clnCur != NULL; clnCur = clnCur->next)
	{
		if (clnCur->bKeep == 1)
			++nCount;
	}

	if ((fArray = (float *)calloc(nCount, sizeof(float))) == NULL)
	{
		HoldDisplay("GetMedian_ClassLevelNetworks memory Error\n");
	}

	for (clnCur = clnHead, nCount = 0; clnCur != NULL; clnCur = clnCur->next)
	{
		if (clnCur->bKeep == 1)
			fArray[nCount++] = clnCur->fAccuracy;
	}


	for (i = 0; i < nCount - 1; i++)
	{
		for (j = 0; j < nCount - i - 1; j++)
		{
			if (fArray[j] > fArray[j + 1])
			{
				fTemp = fArray[j];
				fArray[j] = fArray[j + 1];
				fArray[j + 1] = fTemp;
			}
		}
	}

	j = nCount / 2;

	if (!(nCount % 2))
	{
		if (fArray[j - 1] == fArray[j])
			fMedian = fArray[j - 1];
		else
			fMedian = (fArray[j - 1] + fArray[j]) / 2.0f;
	}
	else
		fMedian = fArray[j];

	return(fMedian);
}



/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void InitializeWeights_ClassLevelNetworksMAC(structMAC *macData, int nMACCount)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	int		p, q;
	
	for (p = 0; p < nMACCount; ++p)
	{
		if (macData[p].nLayerType != MAX_POOLING_LAYER)
		{
			*(macData[p]).fWeight[0] = MULTIPLIER * UNIFORM_PLUS_MINUS_ONE;

			for (q = 1; q < macData[p].nCount; ++q)
				*(macData[p]).fWeight[q] = MULTIPLIER * UNIFORM_PLUS_MINUS_ONE;
		}
	}
}



/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void InitializeWeights_ClassLevelNetworks(structCLN *cln, int nMode, float fSameWeight)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structLayer			*layerCur;
	structPerceptron	*perceptronHeadCur = NULL;
	structPerceptron	*perceptronHead = NULL;
	int					i;

	for (layerCur = cln->layerHead; layerCur != NULL; layerCur = layerCur->next)
	{
		if (layerCur->nLayerType == MAX_POOL_LAYER)
			continue;
		
		
		if (nMode == SAME_WEIGHTS)
		{
			for (i = 0; i < layerCur->nWeightCount; ++i)
			{
				layerCur->fWeightArray[i] = fSameWeight;
			}
		}
		else
		{
			if (nMode == ORDERED && (layerCur->nLayerType == SINGLE_CONV_LAYER || layerCur->nLayerType == MULTIPLE_CONV_LAYER))
			{
				CreateGaborArray_GaborFilter(layerCur);
			}
			else
			{
				for (i = 0; i < layerCur->nWeightCount; ++i)
				{
					layerCur->fWeightArray[i] = MULTIPLIER * UNIFORM_PLUS_MINUS_ONE;
				}
			}
		}
	}

	return;
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void InitializeWeightsOld_ClassLevelNetworks(structCLN *cln, float *fRandomWeightarray, int nMode)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structLayer			*layerCur;
	structPerceptron	*perceptronHeadCur = NULL;
	structPerceptron	*perceptronHead = NULL;
	structPerceptron	*perceptronCur;
	structSynapse		*synapseCur;
	int					nWeightIndex;

	nWeightIndex = 0;

	for (layerCur = cln->layerHead; layerCur != NULL; layerCur = layerCur->next)
	{
		if ((layerCur->nLayerType == SINGLE_CONV_LAYER || layerCur->nLayerType == MULTIPLE_CONV_LAYER))
		{
			for (perceptronHeadCur = layerCur->perceptronHead; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead)
			{
				for (perceptronCur = perceptronHeadCur; perceptronCur != NULL; perceptronCur = perceptronCur->next)
				{
					perceptronCur->fLearningRate = cln->fLearningRate;

					for (synapseCur = perceptronCur->synapseHead; synapseCur != NULL; synapseCur = synapseCur->next)
					{
						*(synapseCur->fWeight) = MULTIPLIER * UNIFORM_PLUS_MINUS_ONE;
					}
				}
			}

			if (nMode == ORDERED)
				CreateGaborArray_GaborFilter(layerCur);
		}
		else if ((layerCur->nLayerType == FULLY_CONNECTED_LAYER || layerCur->nLayerType == CLASSIFIER_LAYER))
		{
			for (perceptronHeadCur = layerCur->perceptronHead; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead)
			{
				for (perceptronCur = perceptronHeadCur; perceptronCur != NULL; perceptronCur = perceptronCur->next)
				{
					perceptronCur->fLearningRate = cln->fLearningRate;

					for (synapseCur = perceptronCur->synapseHead; synapseCur != NULL; synapseCur = synapseCur->next)
					{
						*(synapseCur->fWeight) = MULTIPLIER * UNIFORM_PLUS_MINUS_ONE;
					}
				}
			}
		}
		else if (layerCur->nLayerType == COMBINING_CLASSIFIER_LAYER)
		{
			for (perceptronHeadCur = layerCur->perceptronHead; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead)
			{
				for (perceptronCur = perceptronHeadCur; perceptronCur != NULL; perceptronCur = perceptronCur->next)
				{
					perceptronCur->fLearningRate = cln->fLearningRate;

					for (synapseCur = perceptronCur->synapseHead; synapseCur != NULL; synapseCur = synapseCur->next)
					{
						*(synapseCur->fWeight) = MULTIPLIER * UNIFORM_PLUS_MINUS_ONE;
					}
				}
			}
		}
	}

	return;
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void SetLearningRates_ClassLevelNetworks(structCLN *cln, float fLearningRateMin, float fLearningRateMax)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structLayer			*layerCur;
	structPerceptron	*perceptronHeadCur = NULL;
	structPerceptron	*perceptronHead = NULL;
	structPerceptron	*perceptronCur;

	for (layerCur = cln->layerHead; layerCur != NULL; layerCur = layerCur->next)
	{
		for (perceptronHeadCur = layerCur->perceptronHead; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead)
		{
			for (perceptronCur = perceptronHeadCur; perceptronCur != NULL; perceptronCur = perceptronCur->next)
			{
				if (fLearningRateMin == fLearningRateMax)
					perceptronCur->fLearningRate = fLearningRateMax;
				else
				{
					perceptronCur->fLearningRate = fLearningRateMax * UNIFORM_ZERO_THRU_ONE;

					if (perceptronCur->fLearningRate < fLearningRateMin)
						perceptronCur->fLearningRate = fLearningRateMin;
				}
			}
		}
	}

}


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void WriteSynapseWeight_ClassLevelNetworks(structCLN *cln, int nID, int nImageID, FILE *pFile)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structLayer			*layerCur;
	structPerceptron	*perceptronHeadCur = NULL;
	structPerceptron	*perceptronHead = NULL;
	structPerceptron	*perceptronCur;
	structSynapse		*synapseCur = NULL;
	int					bFound;

	for (layerCur = cln->layerHead, bFound = 0; layerCur != NULL && !bFound; layerCur = layerCur->next)
	{
		for (perceptronHeadCur = layerCur->perceptronHead; perceptronHeadCur != NULL && !bFound; perceptronHeadCur = perceptronHeadCur->nextHead)
		{
			for (perceptronCur = perceptronHeadCur; perceptronCur != NULL && !bFound; perceptronCur = perceptronCur->next)
			{
				for (synapseCur = perceptronCur->synapseHead->next; synapseCur != NULL && !bFound; synapseCur = synapseCur->next)
				{
					if (synapseCur->nID == nID)
					{
						fprintf(pFile, "%d\t%f\t%f\n", nImageID, *synapseCur->fInput, *synapseCur->fWeight);
						bFound = 1;
					}
				}
			}
		}
	}

	return;
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void SetSynapseWeight_ClassLevelNetworks(structCLN *cln, int nID, float fWeight)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structLayer			*layerCur;
	structPerceptron	*perceptronHeadCur = NULL;
	structPerceptron	*perceptronHead = NULL;
	structPerceptron	*perceptronCur;
	structSynapse		*synapseCur = NULL;
	int					bFound;

	for (layerCur = cln->layerHead, bFound = 0; layerCur != NULL && !bFound; layerCur = layerCur->next)
	{
		for (perceptronHeadCur = layerCur->perceptronHead; perceptronHeadCur != NULL && !bFound; perceptronHeadCur = perceptronHeadCur->nextHead)
		{
			for (perceptronCur = perceptronHeadCur; perceptronCur != NULL && !bFound; perceptronCur = perceptronCur->next)
			{
				for (synapseCur = perceptronCur->synapseHead->next; synapseCur != NULL && !bFound; synapseCur = synapseCur->next)
				{
					if (synapseCur->nID == nID)
					{
						*synapseCur->fWeight = fWeight;
						bFound = 1;
					}
				}
			}
		}
	}

	return;
}



/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void AnalyzeSynapseInputs_ClassLevelNetworks(structCLN *cln, structInput *input, float *fInputArray)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structLayer			*layerCur;
	structPerceptron	*perceptronHeadCur = NULL;
	structPerceptron	*perceptronHead = NULL;
	structPerceptron	*perceptronCur;
	structSynapse		*synapseCur = NULL;
	float				fSum = 0.0f;
	float				fValue = 0.0f;
	float				fVariance = 0.0f;
	float				fStandardDeviation;
	float				fAverage;
	float				fMax;
	float				fMin;
	int					nInputIndex;
	int					nCount;
	int					i;

	for (layerCur = cln->layerHead; layerCur != NULL; layerCur = layerCur->next)
	{
		printf("Analyzing Layer Type: %d\n", layerCur->nLayerType);

		for (nInputIndex = 0; nInputIndex < input->nInputCount; ++nInputIndex)
		{
			if (input->data[nInputIndex].bTrained == 1)
				continue;

			fMax = -1.0f;
			fMin = 1.0f;

			for (i = 0; i < cln->nSize; ++i)
			{
				fInputArray[i] = input->data[nInputIndex].fIntensity[i];

				if (fInputArray[i] > fMax)
					fMax = fInputArray[i];
				if (fInputArray[i] < fMin)
					fMin = fInputArray[i];

			}

			ForwardPropagate_Train(cln->macData, cln->nMACCount, NULL);

			for (perceptronHeadCur = layerCur->perceptronHead; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead)
			{
				for (perceptronCur = perceptronHeadCur; perceptronCur != NULL; perceptronCur = perceptronCur->next)
				{
					for (synapseCur = perceptronCur->synapseHead->next; synapseCur != NULL; synapseCur = synapseCur->next)
					{
						synapseCur->fAverage += *synapseCur->fInput;
						++synapseCur->nCount;

						if (*synapseCur->fInput > synapseCur->fMax)
							synapseCur->fMax = *synapseCur->fInput;
						if (*synapseCur->fInput < synapseCur->fMin)
							synapseCur->fMin = *synapseCur->fInput;
					}
				}
			}
		}

		fAverage = 0.0f;
		nCount = 0;

		for (perceptronHeadCur = layerCur->perceptronHead; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead)
		{
			for (perceptronCur = perceptronHeadCur; perceptronCur != NULL; perceptronCur = perceptronCur->next)
			{
				for (synapseCur = perceptronCur->synapseHead->next; synapseCur != NULL; synapseCur = synapseCur->next)
				{
					synapseCur->fAverage /= (float)synapseCur->nCount;
					fAverage += synapseCur->fAverage;



					++nCount;
				}
			}
		}
		fAverage /= (float)nCount;

		for (nInputIndex = 0; nInputIndex < input->nInputCount; ++nInputIndex)
		{
			if (input->data[nInputIndex].bTrained == 1)
				continue;

			for (i = 0; i < cln->nSize; ++i)
				fInputArray[i] = input->data[nInputIndex].fIntensity[i];

			ForwardPropagate_Train(cln->macData, cln->nMACCount, NULL);

			for (perceptronHeadCur = layerCur->perceptronHead; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead)
			{
				for (perceptronCur = perceptronHeadCur; perceptronCur != NULL; perceptronCur = perceptronCur->next)
				{
					for (synapseCur = perceptronCur->synapseHead->next; synapseCur != NULL; synapseCur = synapseCur->next)
					{
						fValue = *synapseCur->fInput - synapseCur->fAverage;
						synapseCur->fSumSquares += (float)pow(fValue, 2.0f);
					}
				}
			}
		}

		//if ((pFile = FOpenMakeDirectory("d:\\Data\\synaptic_output.txt", "wb")) == NULL)
		//{
		//	printf("Could not write file\n\n");
		//	while (1);
		//}

		for (perceptronHeadCur = layerCur->perceptronHead; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead)
		{
			for (perceptronCur = perceptronHeadCur; perceptronCur != NULL; perceptronCur = perceptronCur->next)
			{
				fAverage = 0.0f;
				nCount = 0;


				for (synapseCur = perceptronCur->synapseHead->next; synapseCur != NULL; synapseCur = synapseCur->next)
				{
					fVariance = synapseCur->fSumSquares / (float)synapseCur->nCount;
					fStandardDeviation = (float)sqrt(fVariance);

					*synapseCur->fWeight = MULTIPLIER * UNIFORM_PLUS_MINUS_ONE;

					fAverage += fStandardDeviation;
					++nCount;

					//fprintf(pFile, "%d\t%f\t%f\t%f\n", synapseCur->nID, synapseCur->fAverage, fVariance, fStandardDeviation);
				}

				perceptronCur->fLearningRate = (fAverage / (float)nCount) / 100.0f;
				printf("%f\n", perceptronCur->fLearningRate);
			}
		}

		//fclose(pFile);
	}


}



/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void WriteClassLevelNetwork(structCLN *clnCur, char *sFilePath, FILE *pFile)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structLayer			*layerCur;
	structPerceptron	*perceptronHeadCur = NULL;
	structPerceptron	*perceptronCur;
	structSynapse		*synapseCur;
	int					nLayerCount;
	int					nHeadPerceptronCount;
	int					nPerceptronCount;
	int					nSynapseCount;
	int					bCloseFile = 0;
	int					i;

	if (pFile == NULL)
	{
		if ((pFile = fopen(sFilePath, "wb")) == NULL)
		{
			char	sMessage[256];

			sprintf(sMessage, "File Error: WriteClassLevelNetwork() - Cannot Write To: %s", sFilePath);
			DisplayMessage(sMessage, PAUSE);
		}

		bCloseFile = 1;
	}
	else
		bCloseFile = 0;

	fwrite(&clnCur->nID, sizeof(int), 1, pFile);
	fwrite(&clnCur->nNetworkType, sizeof(int), 1, pFile);
	fwrite(&clnCur->nTargetClass, sizeof(int), 1, pFile);
	fwrite(&clnCur->nStage, sizeof(int), 1, pFile);
	fwrite(&clnCur->fInitialError, sizeof(float), 1, pFile);
	fwrite(&clnCur->fThreshold, sizeof(float), 1, pFile);

	// Layers per CLN Count
	for (layerCur = clnCur->layerHead, nLayerCount = 0; layerCur != NULL; layerCur = layerCur->next, ++nLayerCount);
	fwrite(&nLayerCount, sizeof(int), 1, pFile);

	for (layerCur = clnCur->layerHead; layerCur != NULL; layerCur = layerCur->next)
	{
		fwrite(&layerCur->nID, sizeof(int), 1, pFile);
		fwrite(&layerCur->nIndex, sizeof(int), 1, pFile);
		fwrite(&layerCur->nLayerType, sizeof(int), 1, pFile);
		fwrite(&layerCur->nPaddingMode, sizeof(int), 1, pFile);
		fwrite(&layerCur->nPerceptronCount, sizeof(int), 1, pFile);
		fwrite(&layerCur->nKernelCount, sizeof(int), 1, pFile);
		fwrite(&layerCur->nInputRowCount, sizeof(int), 1, pFile);
		fwrite(&layerCur->nInputColumnCount, sizeof(int), 1, pFile);
		fwrite(&layerCur->nOutputRowCount, sizeof(int), 1, pFile);
		fwrite(&layerCur->nOutputColumnCount, sizeof(int), 1, pFile);
		fwrite(&layerCur->nKernelRowCount, sizeof(int), 1, pFile);
		fwrite(&layerCur->nKernelColumnCount, sizeof(int), 1, pFile);
		fwrite(&layerCur->nStrideRow, sizeof(int), 1, pFile);
		fwrite(&layerCur->nStrideColumn, sizeof(int), 1, pFile);
		fwrite(&layerCur->nInputMapCount, sizeof(int), 1, pFile);

		fwrite(&layerCur->nWeightCount, sizeof(int), 1, pFile);
		fwrite(&layerCur->nConnectionCount, sizeof(int), 1, pFile);
		fwrite(&layerCur->nOutputArraySize, sizeof(int), 1, pFile);
		fwrite(layerCur->sLayerName, sizeof(char), 32, pFile);

		// Write Filter Kernel Weights
		if (layerCur->nLayerType == CONV_2D_LAYER || layerCur->nLayerType == CONV_3D_LAYER)
		{
			for (i = 0; i < layerCur->nWeightCount; ++i)
			{
				fwrite(&layerCur->fxptWeightArray[i], sizeof(fxpt), 1, pFile);
			}
		}

		// Perceptrons per Layer Count
		for (perceptronHeadCur = layerCur->perceptronHead, nHeadPerceptronCount = 0; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead, ++nHeadPerceptronCount);
		fwrite(&nHeadPerceptronCount, sizeof(int), 1, pFile);

		for (perceptronHeadCur = layerCur->perceptronHead, nHeadPerceptronCount = 0; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead, ++nHeadPerceptronCount)
		{
			// Perceptron per Filter
			for (perceptronCur = perceptronHeadCur, nPerceptronCount = 0; perceptronCur != NULL; perceptronCur = perceptronCur->next, ++nPerceptronCount);
			fwrite(&nPerceptronCount, sizeof(int), 1, pFile);

			for (perceptronCur = perceptronHeadCur; perceptronCur != NULL; perceptronCur = perceptronCur->next)
			{
				fwrite(&perceptronCur->nID, sizeof(int), 1, pFile);
				fwrite(&perceptronCur->nIndex, sizeof(int), 1, pFile);
				fwrite(&perceptronCur->fLearningRate, sizeof(float), 1, pFile);

				// Synapse Count
				for (synapseCur = perceptronCur->synapseHead, nSynapseCount = 0; synapseCur != NULL; synapseCur = synapseCur->next, ++nSynapseCount);
				fwrite(&nSynapseCount, sizeof(int), 1, pFile);

				// Bias
				synapseCur = perceptronCur->synapseHead;
				synapseCur->nInputIndex = -1;
				fwrite(&synapseCur->nID, sizeof(int), 1, pFile);
				fwrite(&synapseCur->nInputIndex, sizeof(int), 1, pFile);
				fwrite(synapseCur->fxptWeight, sizeof(fxpt), 1, pFile);

				for (synapseCur = synapseCur->next; synapseCur != NULL; synapseCur = synapseCur->next)
				{
					fwrite(&synapseCur->nID, sizeof(int), 1, pFile);
					fwrite(&synapseCur->nIndexCount, sizeof(int), 1, pFile);

					if (synapseCur->perceptronSource != NULL) // Connect to perceptron output
					{
						fwrite(&synapseCur->perceptronSource->nID, sizeof(int), 1, pFile);
					}
					else
					{
						fwrite(&synapseCur->nInputIndex, sizeof(int), 1, pFile);
					}

					fwrite(synapseCur->fxptWeight, sizeof(fxpt), 1, pFile);

					for (i = 0; i < synapseCur->nIndexCount; ++i)
						fwrite(&synapseCur->nIndexArray[i], sizeof(int), 1, pFile);
				}
			}
		}
	}

	if (bCloseFile)
		fclose(pFile);

	return;
}


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
structCLN *ReadClassLevelNetwork(fxpt *fxptInputArray, int *nID, char *sFilePath, FILE *pFile)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structCLN			*clnNew = NULL;
	structLayer			*layerNew = NULL;
	structPerceptron	*perceptronHeadCur = NULL;
	structPerceptron	*perceptronNewHead = NULL;
	structPerceptron	*perceptronNew = NULL;
	structPerceptron	*perceptronCur = NULL;
	structSynapse		*synapseNew = NULL;
	fxpt				fxptWeight;
	int					nHeadPerceptronCount;
	int					nPerceptronCount;
	int					nSynapseCount;
	int					nConnectToID;
	int					nWeightOffset = 0;
	int					nNoConnectCount = 0;
	int					bCloseFile = 0;
	int					i, j, k, m, n;
	int					p;

	FILE				*pFileOut = NULL;


	if (pFile == NULL)
	{
		if ((pFile = fopen(sFilePath, "rb")) == NULL)
		{
			char	sMessage[256];

			sprintf(sMessage, "File Error: ReadClassLevelNetwork() - Cannot Read from: %s", sFilePath);
			DisplayMessage(sMessage, PAUSE);
		}
		bCloseFile = 1;
	}
	else
		bCloseFile = 0;

	nNoConnectCount = 0;

	if ((clnNew = (structCLN *)calloc(1, sizeof(structCLN))) == NULL)
		DisplayMessage("Memory calloc Error: ReadClassLevelNetwork() clnNew", PAUSE);

	printf("\n");

	fread(nID, sizeof(int), 1, pFile);
	clnNew->nID = *nID;

	fread(&clnNew->nNetworkType, sizeof(int), 1, pFile);
	fread(&clnNew->nTargetClass, sizeof(int), 1, pFile);
	fread(&clnNew->nStage, sizeof(int), 1, pFile);
	fread(&clnNew->fInitialError, sizeof(float), 1, pFile);
	fread(&clnNew->fThreshold, sizeof(float), 1, pFile);
	fread(&clnNew->nLayerCount, sizeof(int), 1, pFile);

	for (j = 0; j < clnNew->nLayerCount; ++j)
	{
		if ((layerNew = (structLayer *)calloc(1, sizeof(structLayer))) == NULL)
			DisplayMessage("Memory calloc Error: ReadClassLevelNetwork() layerNew", PAUSE);

		fread(nID, sizeof(int), 1, pFile);
		layerNew->nID = *nID;

		fread(&layerNew->nIndex, sizeof(int), 1, pFile);
		fread(&layerNew->nLayerType, sizeof(int), 1, pFile);
		fread(&layerNew->nPaddingMode, sizeof(int), 1, pFile);
		fread(&layerNew->nPerceptronCount, sizeof(int), 1, pFile);
		fread(&layerNew->nKernelCount, sizeof(int), 1, pFile);
		fread(&layerNew->nInputRowCount, sizeof(int), 1, pFile);
		fread(&layerNew->nInputColumnCount, sizeof(int), 1, pFile);
		fread(&layerNew->nOutputRowCount, sizeof(int), 1, pFile);
		fread(&layerNew->nOutputColumnCount, sizeof(int), 1, pFile);
		fread(&layerNew->nKernelRowCount, sizeof(int), 1, pFile);
		fread(&layerNew->nKernelColumnCount, sizeof(int), 1, pFile);
		fread(&layerNew->nStrideRow, sizeof(int), 1, pFile);
		fread(&layerNew->nStrideColumn, sizeof(int), 1, pFile);
		fread(&layerNew->nInputMapCount, sizeof(int), 1, pFile);

		fread(&layerNew->nWeightCount, sizeof(int), 1, pFile);
		fread(&layerNew->nConnectionCount, sizeof(int), 1, pFile);
		fread(&layerNew->nOutputArraySize, sizeof(int), 1, pFile);
		fread(layerNew->sLayerName, sizeof(char), 32, pFile);

		//printf("Rebuilding Layer: %s ...\r", layerNew->sLayerName);


		//sprintf(sPath, "Layer_%d.txt", j);

		//if ((pFileOut = fopen(sPath, "wb")) == NULL)
		//{
		//	char	sMessage[256];

		//	sprintf(sMessage, "File Error: ReadClassLevelNetwork() - Cannot Write to: %s", sPath);
		//	DisplayMessage(sMessage, PAUSE);
		//}
		if (pFileOut)
			fwrite(&layerNew->nWeightCount, sizeof(int), 1, pFileOut);


		if ((layerNew->fxptWeightArray = (fxpt *)calloc(layerNew->nWeightCount, sizeof(fxpt))) == NULL)
			DisplayMessage("Memory calloc Error: ReadClassLevelNetwork() layer->fWeightArray", PAUSE);

		layerNew->nInitializeWeights = 0;
		layerNew->nPerceptronCount = 0;
		AddNew_Layer(&clnNew->layerHead, layerNew);

		if (layerNew->nLayerType == CONV_2D_LAYER || layerNew->nLayerType == CONV_3D_LAYER)
		{
			for (i = 0; i < layerNew->nWeightCount; ++i)
			{
				fread(&layerNew->fxptWeightArray[i], sizeof(fxpt), 1, pFile);
				if (pFileOut)
					fwrite(&layerNew->fxptWeightArray[i], sizeof(fxpt), 1, pFileOut);
			}

			nWeightOffset = layerNew->nWeightCount / layerNew->nKernelCount;
		}
		else
			nWeightOffset = 0;


		fread(&nHeadPerceptronCount, sizeof(int), 1, pFile);


		for (k = 0; k < nHeadPerceptronCount; ++k)
		{
			perceptronNewHead = NULL;

			fread(&nPerceptronCount, sizeof(int), 1, pFile);
			for (m = 0; m < nPerceptronCount; ++m)
			{
				printf("Rebuilding Layer: %s ... %d of %d              \r", layerNew->sLayerName, m, nPerceptronCount);

				p = (nWeightOffset * k);

				perceptronNew = (structPerceptron *)calloc(1, sizeof(structPerceptron));

				fread(nID, sizeof(int), 1, pFile);
				perceptronNew->nID = *nID;

				fread(&perceptronNew->nIndex, sizeof(int), 1, pFile);
				fread(&perceptronNew->fLearningRate, sizeof(float), 1, pFile);
				fread(&nSynapseCount, sizeof(int), 1, pFile);

				// Bias
				synapseNew = (structSynapse *)calloc(1, sizeof(structSynapse));

				fread(nID, sizeof(int), 1, pFile);
				synapseNew->nID = *nID;

				fread(&synapseNew->nInputIndex, sizeof(int), 1, pFile);
				fread(&fxptWeight, sizeof(fxpt), 1, pFile);

				synapseNew->fxptInput = (fxpt *)calloc(1, sizeof(fxpt));
				*synapseNew->fxptInput = FXPT32_ONE;

				if (layerNew->nLayerType == CONV_2D_LAYER || layerNew->nLayerType == CONV_3D_LAYER)
					synapseNew->fxptWeight = &layerNew->fxptWeightArray[p++];
				else
				{
					layerNew->fxptWeightArray[layerNew->nWeightIndex] = fxptWeight;
					synapseNew->fxptWeight = &layerNew->fxptWeightArray[layerNew->nWeightIndex++];

					if (pFileOut)
						fwrite(&fxptWeight, sizeof(fxpt), 1, pFileOut);
				}

				AddNew_Synapse(&perceptronNew->synapseHead, synapseNew);

				for (n = 1; n < nSynapseCount; ++n)
				{
					synapseNew = (structSynapse *)calloc(1, sizeof(structSynapse));

					fread(nID, sizeof(int), 1, pFile);
					synapseNew->nID = *nID;
					fread(&synapseNew->nIndexCount, sizeof(int), 1, pFile);

					if (j == 0) // First Layer
					{
						fread(&synapseNew->nInputIndex, sizeof(int), 1, pFile);
						fread(&fxptWeight, sizeof(fxpt), 1, pFile);


						if (layerNew->nLayerType == CONV_2D_LAYER || layerNew->nLayerType == CONV_3D_LAYER)
							synapseNew->fxptWeight = &layerNew->fxptWeightArray[p++];
						else
						{
							layerNew->fxptWeightArray[layerNew->nWeightIndex] = fxptWeight;
							synapseNew->fxptWeight = &layerNew->fxptWeightArray[layerNew->nWeightIndex++];

							if (pFileOut)
								fwrite(&fxptWeight, sizeof(fxpt), 1, pFileOut);
						}

						synapseNew->fxptInput = &fxptInputArray[synapseNew->nInputIndex];
					}
					else
					{
						int bFound = 0;

						fread(&nConnectToID, sizeof(int), 1, pFile);
						fread(&fxptWeight, sizeof(fxpt), 1, pFile);

						if (layerNew->nLayerType == CONV_2D_LAYER || layerNew->nLayerType == CONV_3D_LAYER)
							synapseNew->fxptWeight = &layerNew->fxptWeightArray[p++];
						else
						{
							layerNew->fxptWeightArray[layerNew->nWeightIndex] = fxptWeight;
							synapseNew->fxptWeight = &layerNew->fxptWeightArray[layerNew->nWeightIndex++];

							if (pFileOut)
								fwrite(&fxptWeight, sizeof(fxpt), 1, pFileOut);
						}

						for (perceptronHeadCur = layerNew->prev->perceptronHead; perceptronHeadCur != NULL && !bFound; perceptronHeadCur = perceptronHeadCur->nextHead)
						{
							for (perceptronCur = perceptronHeadCur; perceptronCur != NULL && !bFound; perceptronCur = perceptronCur->next)
							{
								if (perceptronCur->nID == nConnectToID)
								{
									synapseNew->fxptInput = &perceptronCur->fxptOutput;
									synapseNew->perceptronSource = perceptronCur;

									bFound = 1;
								}
							}
						}

						if (!bFound)
							++nNoConnectCount;
					}

					if (synapseNew->nIndexCount > 0)
					{
						if ((synapseNew->nIndexArray = (int *)calloc(synapseNew->nIndexCount, sizeof(int))) == NULL)
							DisplayMessage("Memory calloc Error: ReadClassLevelNetwork() synapseNew->nIndexArray", PAUSE);

						for (i = 0; i < synapseNew->nIndexCount; ++i)
							fread(&synapseNew->nIndexArray[i], sizeof(int), 1, pFile);
					}

					AddNew_Synapse(&perceptronNew->synapseHead, synapseNew);
				}

				AddNew_Perceptron(&perceptronNewHead, perceptronNew);
				++layerNew->nPerceptronCount;
			}

			AddNewV2_Perceptron(&layerNew->perceptronHead, perceptronNewHead);
		}

		if (layerNew->nLayerType == CLASSIFIER_LAYER || layerNew->nLayerType == COMBINING_CLASSIFIER_LAYER)
		{
			clnNew->perceptronClassifier = layerNew->perceptronHead;
			clnNew->layerClassifier = layerNew;
		}

		//printf("Rebuilding Layer: %s ... %d of %d   \n", layerNew->sLayerName, m, nPerceptronCount);

		if (pFileOut)
			fclose(pFileOut);
	}

	clnNew->bRebuilt = 1;

	if (bCloseFile)
		fclose(pFile);

	AnalyzeCLN(clnNew, 0);

	return(clnNew);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void AnalyzeCLN(structCLN *clnHead, int nNumberMode)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	FILE				*pFile = NULL;
	structCLN			*cln;
	structLayer			*layerCur;
	structPerceptron	*perceptronCurHead;
	structSynapse		*synapseCur;
	int					i;

	for (cln = clnHead; cln != NULL; cln = cln->next)
	{
		for (layerCur = cln->layerHead, cln->nLayerCount = 0; layerCur != NULL; layerCur = layerCur->next, ++cln->nLayerCount);

		if (nNumberMode == FLOAT_POINT)
		{
			for (layerCur = cln->layerHead; layerCur != NULL; layerCur = layerCur->next)
			{
				layerCur->fMaxWeight = -999999.9F;
				layerCur->fMinWeight = 999999.9F;

				for (i = 0; i < layerCur->nWeightCount; ++i)
				{
					if (layerCur->fWeightArray[i] > layerCur->fMaxWeight)
						layerCur->fMaxWeight = layerCur->fWeightArray[i];
					if (layerCur->fWeightArray[i] < layerCur->fMinWeight)
						layerCur->fMinWeight = layerCur->fWeightArray[i];
				}
			}
		}

		cln->nWeightCount = 0;
		for (layerCur = cln->layerHead; layerCur != NULL; layerCur = layerCur->next)
		{
			layerCur->nWeightCount = 0;

			if (layerCur->nLayerType == CONV_2D_LAYER || layerCur->nLayerType == CONV_3D_LAYER)
			{
				for (perceptronCurHead = layerCur->perceptronHead, layerCur->nKernelCount = 0; perceptronCurHead != NULL; perceptronCurHead = perceptronCurHead->nextHead, ++layerCur->nKernelCount);

				for (perceptronCurHead = layerCur->perceptronHead; perceptronCurHead != NULL; perceptronCurHead = perceptronCurHead->nextHead)
				{
					for (synapseCur = perceptronCurHead->synapseHead, perceptronCurHead->nSynapseCount = 0; synapseCur != NULL; synapseCur = synapseCur->next, ++perceptronCurHead->nSynapseCount, ++layerCur->nWeightCount);
				}
			}
			else
			{
				for (perceptronCurHead = layerCur->perceptronHead, layerCur->nPerceptronCount = 0; perceptronCurHead != NULL; perceptronCurHead = perceptronCurHead->next, ++layerCur->nPerceptronCount);

				for (perceptronCurHead = layerCur->perceptronHead; perceptronCurHead != NULL; perceptronCurHead = perceptronCurHead->next)
				{
					for (synapseCur = perceptronCurHead->synapseHead, perceptronCurHead->nSynapseCount = 0; synapseCur != NULL; synapseCur = synapseCur->next, ++perceptronCurHead->nSynapseCount, ++layerCur->nWeightCount);
				}
			}

			cln->nWeightCount += layerCur->nWeightCount;
		}
	}


}


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
structCLN *AllocateCLN(int *nID)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structCLN	*newCLN = NULL;

	if ((newCLN = (structCLN *)calloc(1, sizeof(structCLN))) == NULL)
		DisplayMessage("Memory calloc Error: AllocateCLN() newCLN", PAUSE);

	newCLN->nID = (*nID)++;

	return(newCLN);
}






/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void CreateMACArrayFromDNX_ClassLevelNetworks(structCLN *cln)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structLayer			*layerCur;
	structPerceptron	*perceptronHeadCur = NULL;
	structPerceptron	*perceptronHead = NULL;
	structPerceptron	*perceptronCur;
	structSynapse		*synapseCur;
	int					nPerceptronCount = 0;
	int					nSynapseCount = 0;
	int					nKernel = 0;
	int					nClassifierID;


	//Delete Old MAC array
	FreeArray_MAC(&cln->macData, &cln->nMACCount);

	for (layerCur = cln->layerHead, nPerceptronCount = 0; layerCur != NULL; layerCur = layerCur->next)
	{
		if (layerCur->perceptronHead != NULL)
		{
			for (perceptronHeadCur = layerCur->perceptronHead; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead)
			{
				for (perceptronCur = perceptronHeadCur; perceptronCur != NULL; perceptronCur = perceptronCur->next)
				{
					
					if (perceptronCur->synapseHead != NULL)
					{
					++nPerceptronCount;
						for (synapseCur = perceptronCur->synapseHead; synapseCur != NULL; synapseCur = synapseCur->next, ++nSynapseCount);
					}
				}
			}
		}
		else
		{
			nPerceptronCount += layerCur->nOutputArraySize;
			nSynapseCount += layerCur->nConnectionCount;
		}
	}
	
	
	if ((cln->macData = (structMAC *)calloc(nPerceptronCount, sizeof(structMAC))) == NULL)
	{
		HoldDisplay("memory Error: void CreateArray(structNetwork *networkMain, structMAC **macData)\n");
	}

	nClassifierID = 0;
	for (layerCur = cln->layerHead, nPerceptronCount = 0; layerCur != NULL; layerCur = layerCur->next)
	{
		if (layerCur->perceptronHead != NULL)
		{
			for (perceptronHeadCur = layerCur->perceptronHead; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead)
			{
				for (perceptronCur = perceptronHeadCur; perceptronCur != NULL; perceptronCur = perceptronCur->next)
				{
					if (perceptronCur->synapseHead == NULL)
						continue;
					
					cln->macData[nPerceptronCount].nLayerCount = layerCur->nIndex;
					
					for (synapseCur = perceptronCur->synapseHead, nSynapseCount = 0; synapseCur != NULL; synapseCur = synapseCur->next, ++nSynapseCount);

					if (nSynapseCount > 0)
					{
						if ((cln->macData[nPerceptronCount].fInputArray = (float***)calloc(nSynapseCount, sizeof(float**))) == NULL)
						{
							HoldDisplay("memory Error: void CreateArray(structNetwork *networkMain, structMAC **macData)\n");
						}
						if ((cln->macData[nPerceptronCount].fInput = (float**)calloc(nSynapseCount, sizeof(float*))) == NULL)
						{
							HoldDisplay("memory Error: void CreateArray(structNetwork *networkMain, structMAC **macData)\n");
						}


						if ((cln->macData[nPerceptronCount].fInputSum = (float**)calloc(nSynapseCount, sizeof(float*))) == NULL)
						{
							HoldDisplay("memory Error: void CreateArray(structNetwork *networkMain, structMAC **macData)\n");
						}
						if ((cln->macData[nPerceptronCount].fInputAverage = (float**)calloc(nSynapseCount, sizeof(float*))) == NULL)
						{
							HoldDisplay("memory Error: void CreateArray(structNetwork *networkMain, structMAC **macData)\n");
						}



						if ((cln->macData[nPerceptronCount].fWeight = (float**)calloc(nSynapseCount, sizeof(float*))) == NULL)
						{
							HoldDisplay("memory Error: void CreateArray(structNetwork *networkMain, structMAC **macData)\n");
						}

						if ((cln->macData[nPerceptronCount].fConnectToDifferential = (float**)calloc(nSynapseCount, sizeof(float*))) == NULL)
						{
							HoldDisplay("memory Error: void CreateArray(structNetwork *networkMain, structMAC **macData)\n");
						}

						if ((cln->macData[nPerceptronCount].nConnectFromID = (int*)calloc(nSynapseCount, sizeof(int))) == NULL)
						{
							HoldDisplay("memory Error: void CreateArray(structNetwork *networkMain, structMAC **macData)\n");
						}

						if ((cln->macData[nPerceptronCount].fAverage = (float*)calloc(nSynapseCount, sizeof(float))) == NULL)
						{
							HoldDisplay("memory Error: void CreateArray(structNetwork *networkMain, structMAC **macData)\n");
						}

						if ((cln->macData[nPerceptronCount].nAverageCount = (int*)calloc(nSynapseCount, sizeof(int))) == NULL)
						{
							HoldDisplay("memory Error: void CreateArray(structNetwork *networkMain, structMAC **macData)\n");
						}

						if ((cln->macData[nPerceptronCount].fSumSquares = (float*)calloc(nSynapseCount, sizeof(float))) == NULL)
						{
							HoldDisplay("memory Error: void CreateArray(structNetwork *networkMain, structMAC **macData)\n");
						}

						if ((cln->macData[nPerceptronCount].nInputCount = (int*)calloc(nSynapseCount, sizeof(int))) == NULL)
						{
							HoldDisplay("memory Error: void CreateArray(structNetwork *networkMain, structMAC **macData)\n");
						}



						for (synapseCur = perceptronCur->synapseHead, cln->macData[nPerceptronCount].nCount = 0; synapseCur != NULL; synapseCur = synapseCur->next)
						{
							if (synapseCur->bAdjust == 1)
								continue;

							cln->macData[nPerceptronCount].fWeight[cln->macData[nPerceptronCount].nCount] = synapseCur->fWeight;
							cln->macData[nPerceptronCount].fInput[cln->macData[nPerceptronCount].nCount] = synapseCur->fInput;

							if (synapseCur->nInputCount > 0)
							{
								cln->macData[nPerceptronCount].nInputCount[cln->macData[nPerceptronCount].nCount] = synapseCur->nInputCount;

								if ((cln->macData[nPerceptronCount].fInputArray[cln->macData[nPerceptronCount].nCount] = (float**)calloc(synapseCur->nInputCount, sizeof(float*))) == NULL)
								{
									HoldDisplay("memory Error: void CreateArray(structNetwork *networkMain, structMAC **macData)\n");
								}

								for (int i = 0; i < synapseCur->nInputCount; ++i)
								{
									cln->macData[nPerceptronCount].fInputArray[cln->macData[nPerceptronCount].nCount][i] = synapseCur->fInputArray[i];
								}



								if ((cln->macData[nPerceptronCount].fInputSum[cln->macData[nPerceptronCount].nCount] = (float*)calloc(synapseCur->nInputCount, sizeof(float))) == NULL)
								{
									HoldDisplay("memory Error: void CreateArray(structNetwork *networkMain, structMAC **macData)\n");
								}
								if ((cln->macData[nPerceptronCount].fInputAverage[cln->macData[nPerceptronCount].nCount] = (float*)calloc(synapseCur->nInputCount, sizeof(float))) == NULL)
								{
									HoldDisplay("memory Error: void CreateArray(structNetwork *networkMain, structMAC **macData)\n");
								}

							}


							if (synapseCur->perceptronConnectTo != NULL)
							{
								cln->macData[nPerceptronCount].fConnectToDifferential[cln->macData[nPerceptronCount].nCount] = &synapseCur->perceptronConnectTo->fDifferential;
								cln->macData[nPerceptronCount].nConnectFromID[cln->macData[nPerceptronCount].nCount] = synapseCur->perceptronConnectTo->nID;
							}
							else
							{
								cln->macData[nPerceptronCount].fConnectToDifferential[cln->macData[nPerceptronCount].nCount] = NULL;
								cln->macData[nPerceptronCount].nConnectFromID[cln->macData[nPerceptronCount].nCount] = -1;
							}

							++cln->macData[nPerceptronCount].nCount;
						}

						cln->macData[nPerceptronCount].fOutput = &perceptronCur->fOutput;
						cln->macData[nPerceptronCount].fDifferential = &perceptronCur->fDifferential;
						cln->macData[nPerceptronCount].fLearningRate = &perceptronCur->fLearningRate;
						cln->macData[nPerceptronCount].nLayerType = layerCur->nLayerType;
						cln->macData[nPerceptronCount].nKernelID = perceptronHeadCur->nIndex;

						if (cln->macData[nPerceptronCount].nLayerType == CLASSIFIER_LAYER)
							cln->macData[nPerceptronCount].nID = nClassifierID++;
					}

					++nPerceptronCount;
				}
			}
		}
	}

	cln->nMACCount = nPerceptronCount;
}



/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void AlterWeights(structCLN *cln, float fMultiplier)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structLayer			*layerCur;
	structPerceptron	*perceptronHeadCur = NULL;
	structPerceptron	*perceptronHead = NULL;
	structPerceptron	*perceptronCur;
	structSynapse		*synapseCur;

	for (layerCur = cln->layerHead; layerCur != NULL; layerCur = layerCur->next)
	{
		if (layerCur->nLayerType == SINGLE_CONV_LAYER || layerCur->nLayerType == MULTIPLE_CONV_LAYER)
		{
			for (perceptronCur = layerCur->perceptronHead; perceptronCur != NULL; perceptronCur = perceptronCur->nextHead)
			{
				for (synapseCur = perceptronCur->synapseHead; synapseCur != NULL; synapseCur = synapseCur->next)
				{
					*synapseCur->fWeight = *synapseCur->fWeight + (*synapseCur->fWeight * fMultiplier);
				}
			}
		}
		else
		{
			for (perceptronHeadCur = layerCur->perceptronHead; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead)
			{
				for (perceptronCur = perceptronHeadCur; perceptronCur != NULL; perceptronCur = perceptronCur->next)
				{
					for (synapseCur = perceptronCur->synapseHead; synapseCur != NULL; synapseCur = synapseCur->next)
					{
						*synapseCur->fWeight = *synapseCur->fWeight + (*synapseCur->fWeight * fMultiplier);
					}
				}
			}
		}
	}
}
