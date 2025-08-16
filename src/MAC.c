#include "main.h"

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void FreeArray_MAC(structMAC **macData, int *nMACCount)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	int		i;

	if ((*macData) != NULL)
	{
		for (i = 0; i<*nMACCount; ++i)
		{
			free((*macData)[i].nConnectFromID);
			free((*macData)[i].fConnectToDifferential);
			//free((*macData)[i].fInput);
			//free((*macData)[i].fWeight);
			//free((*macData)[i].fAverage);
			//free((*macData)[i].nAverageCount);
			//free((*macData)[i].fSumSquares);
		}

		free(*macData);

		*macData = NULL;
	}

	*nMACCount = 0;
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void CreateConnectionLayer_ClassLevelNetworks(structCLN *cln, int nNeuralCount, int nMode, float fLearningRate, int nRandomMode, int nRowCount, int nColumnCount, int *nPerceptronID, int *nSynapseID, float *fInputArray)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structLayer			*layerNew;
	structLayer			*layerPrev = NULL;
	structPerceptron	*perceptronSubSetCur = NULL;
	structPerceptron	*perceptronNew = NULL;
	structPerceptron	*perceptronNewHead = NULL;
	structPerceptron	*perceptronCurHead = NULL;
	structPerceptron	*perceptronCur = NULL;
	structSynapse		*synapseNew = NULL;
	structSynapse		*synapseTail = NULL;
	int					nNeuralPreviousCount;
	int					nSynapseIndex;
	int					nInputIndex;
	int					nCluster=0;
	int					i, j;


	if ((layerNew = (structLayer *)calloc(1, sizeof(structLayer))) == NULL)
		exit(0);

	layerNew->nLayerType = nMode;
	layerNew->nInputRowCount = 0;
	layerNew->nInputColumnCount = 0;
	layerNew->nPerceptronCount = 0;
	layerNew->nPaddingMode = BACK_UP;
	layerNew->nActivationMode = MTANH;
	layerNew->nNumberFormat = FLOAT_POINT;

	nSynapseIndex = 0;

	if (cln->layerHead != NULL)
	{
		for (layerPrev = cln->layerHead; layerPrev->next != NULL; layerPrev = layerPrev->next);

		nNeuralPreviousCount = layerPrev->nOutputArraySize;
	}
	else
	{
		nNeuralPreviousCount = nRowCount * nColumnCount;
	}

	// Create Weight Array
	layerNew->nWeightCount = nNeuralCount * (nNeuralPreviousCount + 1);
	layerNew->fWeightArray = (float *)calloc(layerNew->nWeightCount, sizeof(float *));
	cln->nWeightCount += layerNew->nWeightCount;

	nSynapseIndex = 0;

	for (i = 0; i<nNeuralCount; ++i)
	{
		perceptronNew = (structPerceptron *)calloc(1, sizeof(structPerceptron));
		perceptronNew->nID = (*nPerceptronID)++;

		perceptronNew->nLayerType = nMode;
		perceptronNew->nIndex = i;
		perceptronNew->fLearningRate = fLearningRate;

		//Add Bias Synapse ////////////////////////////////////////////////////////////////////////
		perceptronNew->synapseHead = (structSynapse *)calloc(1, sizeof(structSynapse));
		perceptronNew->synapseHead->nID = (*nSynapseID);
		perceptronNew->synapseHead->nIndex = nSynapseIndex;
		perceptronNew->synapseHead->fWeight = &layerNew->fWeightArray[perceptronNew->synapseHead->nIndex];
		perceptronNew->synapseHead->fInput = (float *)calloc(1, sizeof(float));
		*(perceptronNew->synapseHead->fInput) = 1; // Bias
		perceptronNew->synapseHead->nInputArrayIndex = -1;

		synapseTail = perceptronNew->synapseHead;

		++(*nSynapseID);
		++nSynapseIndex;

		if (layerPrev != NULL)
		{
			if (layerPrev->perceptronHead != NULL)
			{
				for (perceptronCurHead = layerPrev->perceptronHead; perceptronCurHead != NULL; perceptronCurHead = perceptronCurHead->nextHead)
				{
					for (perceptronCur = perceptronCurHead; perceptronCur != NULL; perceptronCur = perceptronCur->next, ++(*nSynapseID), ++nSynapseIndex)
					{
						synapseNew = (structSynapse *)calloc(1, sizeof(structSynapse));

						synapseNew->nID = (*nSynapseID);
						synapseNew->nIndex = nSynapseIndex;
						synapseNew->nCluster = nCluster++;

						synapseNew->fWeight = &layerNew->fWeightArray[synapseNew->nIndex];
						synapseNew->perceptronConnectTo = perceptronCur;
						synapseNew->fInput = &synapseNew->perceptronConnectTo->fOutput;
						synapseNew->nInputArrayIndex = -999;

						synapseTail->next = synapseNew;
						synapseTail = synapseNew;
					}
				}
			}
			else
			{
				for (j = 0; j<layerPrev->nOutputArraySize; ++j)
				{
					synapseNew = (structSynapse *)calloc(1, sizeof(structSynapse));

					synapseNew->nID = (*nSynapseID);
					synapseNew->nIndex = nSynapseIndex;
					synapseNew->nCluster = nCluster++;

					synapseNew->fWeight = &layerNew->fWeightArray[synapseNew->nIndex];
					synapseNew->perceptronConnectTo = perceptronCur;
					synapseNew->fInput = &layerPrev->fOutputArray[j];
					synapseNew->nInputArrayIndex = -999;

					synapseTail->next = synapseNew;
					synapseTail = synapseNew;
				}
			}
		}
		else
		{
			for (nInputIndex = 0; nInputIndex<nNeuralPreviousCount; ++nInputIndex, ++(*nSynapseID))
			{
				synapseNew = (structSynapse *)calloc(1, sizeof(structSynapse));

				synapseNew->nID = (*nSynapseID);
				synapseNew->nIndex = synapseNew->nID;
				synapseNew->nInputArrayIndex = nInputIndex;
				synapseNew->nCluster = nCluster++;

				synapseNew->fWeight = &layerNew->fWeightArray[synapseNew->nIndex];
				synapseNew->fInput = &fInputArray[synapseNew->nInputArrayIndex];

				synapseTail->next = synapseNew;
				synapseTail = synapseNew;
			}
		}

		++layerNew->nPerceptronCount;
		perceptronNew->nWeightCount = nSynapseIndex;
		AddNewV2_Perceptron(&perceptronNewHead, perceptronNew);
		////////////////////////////////////////////////////////////////////////
	}

	perceptronNewHead->nDimX = 1;
	perceptronNewHead->nDimY = nNeuralCount;

	AddToLayer_Perceptron(&layerNew->perceptronHead, perceptronNewHead, 0);

	layerNew->nConnectionCount = layerNew->nWeightCount - layerNew->nPerceptronCount;
	layerNew->nOutputArraySize = layerNew->nPerceptronCount;

	AddNew_Layer(&cln->layerHead, layerNew);

	for (perceptronCur = layerNew->perceptronHead; perceptronCur != NULL; perceptronCur = perceptronCur->next)
		perceptronCur->nLayerID = layerNew->nID;


	if (layerNew->nLayerType == CLASSIFIER_LAYER)
	{
		cln->perceptronClassifier = layerNew->perceptronHead;
		cln->layerClassifier = layerNew;
	}

	return;
}



/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void ClearAverages_MAC(structMAC *macData, int nPerceptronCount)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	int		i, j;

	for (i = 0; i<nPerceptronCount; ++i)
	{
		for (j = 1; j < macData[i].nCount; ++j)
		{
			macData[i].nAverageCount[j] = 0;
			macData[i].fAverage[j] = 0.0f;
			macData[i].fSumSquares[j] = 0.0f;
		}
	}
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void CalculateAverages_MAC(structMAC *macData, int nPerceptronCount)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	int		i, j;

	for (i = 0; i<nPerceptronCount; ++i)
	{
		for (j = 1; j < macData[i].nCount; ++j)
		{
			if (macData[i].nAverageCount[j] > 0)
				macData[i].fAverage[j] /= (float)macData[i].nAverageCount[j];
			else
				macData[i].fAverage[j] = 0.0f;
		}
	}
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void PrintPerceptronInputData_MAC(structMAC *macData, int nPerceptronCount)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	int		i, j;

	for (i = 0; i<nPerceptronCount; ++i)
	{
		for (j = 1; j < macData[i].nCount; ++j)
		{
			if (macData[i].nLayerType == FULLY_CONNECTED_LAYER)
			{
				printf("%d\t%f\n", macData[i].nConnectFromID[j], *macData[i].fInput[j]);
			}
		}
	}
}




/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
float CalculateStandardDeviations_MAC(structMAC *macData, int nPerceptronCount, int nDisplayMode)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	float	fVariance;
	float	fVarianceAverage;
	float	fStandardDeviationAverage;
	float	fStandardDeviation;
	float	fAverage[100];
	float	fLearningRate[100];
	int		nAverageCount[100];
	int		nCurLayerType;
	int		nLayerTypeCount;
	int		nDivisorCount[100];
	int		i, j;


	nCurLayerType = -1;
	nLayerTypeCount = -1;

	for (i = 0; i < 100; ++i)
	{
		fAverage[i] = 0.0f;
		fLearningRate[i] = 0.0f;
		nAverageCount[i] = 0;
		nDivisorCount[i] = 0;
	}

	nDivisorCount[0] = 0;
	nDivisorCount[1] = 0;
	nDivisorCount[2] = 2;
	nDivisorCount[3] = 3;



	for (i = 0; i<nPerceptronCount; ++i)
	{
		fStandardDeviationAverage = 0.0f;
		fVarianceAverage = 0.0f;

		for (j = 1; j < macData[i].nCount; ++j)
		{
			fVariance = macData[i].fSumSquares[j] / (float)macData[i].nAverageCount[j];
			fStandardDeviation = (float)sqrt(fVariance);

			fVarianceAverage += fVariance;
			fStandardDeviationAverage += fStandardDeviation;
		}

		if (macData[i].nLayerType != nCurLayerType)
		{
			nCurLayerType = macData[i].nLayerType;
			++nLayerTypeCount;
		}

		fLearningRate[nLayerTypeCount] = (fStandardDeviationAverage / (float)macData[i].nCount);
		fAverage[nLayerTypeCount] += fLearningRate[nLayerTypeCount];
		++nAverageCount[nLayerTypeCount];

		*macData[i].fLearningRate = fLearningRate[nLayerTypeCount];

		/*	if (nLayerTypeCount == 0)
		{
		*macData[i].fLearningRate = fLearningRate[nLayerTypeCount];
		}
		else
		{
		fDivisor = (float)pow(10.0, nDivisorCount[nLayerTypeCount]);

		fLearningRate[nLayerTypeCount] = (fAverage[nLayerTypeCount-1] / (float)nAverageCount[nLayerTypeCount-1]) / fDivisor;
		*macData[i].fLearningRate = fLearningRate[nLayerTypeCount];
		}*/


		//macData[i].fAdjustLearningRate = (fVarianceAverage / ((float)macData[i].nCount)) * 0.05f;
	}

	if (nDisplayMode == SHOW_DATA)
	{
		for (i = 0; i <= nLayerTypeCount; ++i)
		{
			printf("%d\t%d\t%f\t%f\n", i, nAverageCount[i], fAverage[i] / (float)nAverageCount[i], fLearningRate[i]);
		}
		printf("\n");
	}

	return(fAverage[nLayerTypeCount] / (float)nAverageCount[nLayerTypeCount]);
}





//
///*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
//void CreateMACArray(structCLN *cln, structLayer *layer)
///*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
//{
//	structPerceptron	*perceptronHeadCur = NULL;
//	structPerceptron	*perceptronHead = NULL;
//	structPerceptron	*perceptronCur;
//	structSynapse		*synapseCur;
//	int					nPerceptronCount = 0;
//	int					nSynapseCount = 0;
//	int					nKernel = 0;
//
//
//	//Delete Old MAC array
//	FreeMACArray(&cln->macData, &cln->nMACCount);
//
//
//	if ((cln->macData = (structMAC *)calloc(layer->nPerceptronCount, sizeof(structMAC))) == NULL)
//	{
//		HoldDisplay("memory Error: void CreateArray(structNetwork *networkMain, structMAC **macData)\n");
//	}
//
//
//	for (perceptronHeadCur = layer->perceptronHead; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead)
//	{
//		for (perceptronCur = perceptronHeadCur; perceptronCur != NULL; perceptronCur = perceptronCur->next, ++nPerceptronCount)
//		{
//			for (synapseCur = perceptronCur->synapseHead, nSynapseCount = 0; synapseCur != NULL; synapseCur = synapseCur->next, ++nSynapseCount);
//
//			if ((cln->macData[nPerceptronCount].nInputCount = (int *)calloc(nSynapseCount, sizeof(int))) == NULL)
//				HoldDisplay("memory Error: void CreateArray(structNetwork *networkMain, structMAC **macData)\n");
//
//			if ((cln->macData[nPerceptronCount].fWeight = (float **)calloc(nSynapseCount, sizeof(float *))) == NULL)
//				HoldDisplay("memory Error: void CreateArray(structNetwork *networkMain, structMAC **macData)\n");
//
//			if ((cln->macData[nPerceptronCount].fInput = (float **)calloc(nSynapseCount, sizeof(float *))) == NULL)
//				HoldDisplay("memory Error: void CreateArray(structNetwork *networkMain, structMAC **macData)\n");
//
//			if ((cln->macData[nPerceptronCount].fInputArray = (float ***)calloc(nSynapseCount, sizeof(float **))) == NULL)
//				HoldDisplay("memory Error: void CreateArray(structNetwork *networkMain, structMAC **macData)\n");
//
//
//			for (synapseCur = perceptronCur->synapseHead, cln->macData[nPerceptronCount].nCount = 0; synapseCur != NULL; synapseCur = synapseCur->next)
//			{
//				cln->macData[nPerceptronCount].fWeight[cln->macData[nPerceptronCount].nCount] = synapseCur->fWeight;
//				cln->macData[nPerceptronCount].fInput[cln->macData[nPerceptronCount].nCount] = synapseCur->fInput;
//
//				if (synapseCur->nInputCount > 0)
//				{
//					cln->macData[nPerceptronCount].nInputCount[cln->macData[nPerceptronCount].nCount] = synapseCur->nInputCount;
//
//					if ((cln->macData[nPerceptronCount].fInputArray[cln->macData[nPerceptronCount].nCount] = (float **)calloc(synapseCur->nInputCount, sizeof(float*))) == NULL)
//					{
//						HoldDisplay("memory Error: void CreateArray(structNetwork *networkMain, structMAC **macData)\n");
//					}
//
//					for (int i = 0; i < synapseCur->nInputCount; ++i)
//					{
//						cln->macData[nPerceptronCount].fInputArray[cln->macData[nPerceptronCount].nCount][i] = synapseCur->fInputArray[i];
//					}
//				}
//
//				++cln->macData[nPerceptronCount].nCount;
//			}
//
//			cln->macData[nPerceptronCount].fOutput = &perceptronCur->fOutput;
//		}
//	}
//
//	cln->nMACCount = nPerceptronCount;
//}
//
///*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
//void FreeMACArray(structMAC **macData, int *nMACCount)
///*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
//{
//	int		i, j, k;
//
//	if ((*macData) != NULL)
//	{
//		for (i = 0; i < *nMACCount; ++i)
//		{
//			for (j = 0; j < (*macData)[i].nCount; ++j)
//			{
//				if ((*macData)[i].nInputCount[j] > 0)
//				{
//					free((*macData)[i].fInputArray[j]);
//				}
//			}
//
//			free((*macData)[i].fInput);
//			free((*macData)[i].fWeight);
//			free((*macData)[i].nInputCount);
//		}
//
//		free(*macData);
//
//		*macData = NULL;
//	}
//
//	*nMACCount = 0;
//}


