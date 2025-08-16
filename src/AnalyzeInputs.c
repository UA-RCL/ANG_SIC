#include "main.h"


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void AnalyzeInputs(float *fInputArray, structInput *inputTrainData, structInput *inputValidateData)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structCLN				*cln;
	structInputPerceptron	*perceptron=NULL;
	structInputLayer		*layer;
	structInput				*inputTrainingSubGroup = NULL;
	int						nPerceptronID=0;
	int						nSynapseID =0;
	int						nClusterCount = 2;
	int						nMaxClusterCount = 2;
	int						nTarget = 1;
	int						nRemainingClassMembers;
	int						nCount = 0;
	int						nIndex = 0;
	int						nRight = 0;
	int						nWrong = 0;
	int						nSearch = 0;
	int						nCycle = 0;
	int						nCorrectCount;
	int						nClassID;
	int						nPerceptronCount;
	int						bCreateNewPerceptron;
	int						nClassCount=10;


	if ((cln = (structCLN *)calloc(1, sizeof(structCLN))) == NULL)
		exit(0);

	cln->nID = 0;
	cln->fLearningRate = 0.001f;
	cln->fInitialError = 1.0f;
	cln->fThreshold = 0.0f;
	cln->nLabelID = -1;
	cln->nSize = inputTrainData->nSize;
	cln->nPerceptronLayerCount = 1;

	//%%%%%%%%%% Build Input Layer %%%%%%%%%%
	layer = NewInputLayer(cln);

	
	//for (nCycle=0; nCycle<5; ++nCycle)
	//{
	//	for (nIndex = 0; nIndex < inputTrainData->nInputCount; ++nIndex)
	//		inputTrainData->data[nIndex].nCluster = -1;

	//	nMaxClusterCount = 0;
	//	for (nClassID = 0; nClassID < nClassCount; ++nClassID)
	//	{
	//		AssignInputError(inputTrainData, nClassID);
	//		nClusterCount = GroupInputError(inputTrainData, nClassID, 25);
	//		if (nClusterCount>nMaxClusterCount)
	//			nMaxClusterCount = nClusterCount;

	//		printf("%d\t%d\n", nClassID, nClusterCount);
	//	}

	//	printf("\n");
	//	CreateTrainingDataSubGroup(inputTrainData, &inputTrainingSubGroup, 1140, nMaxClusterCount, nClassCount);


		for (nClassID = 0; nClassID < 10; ++nClassID)
		{
			nRemainingClassMembers = 1;
			nPerceptronCount = 0;
			bCreateNewPerceptron = 1;
			nIndex = -1;

			while (nRemainingClassMembers)
			{
				if (bCreateNewPerceptron)
					perceptron = NewInputPerceptron(layer, &nPerceptronID, &nSynapseID, nClusterCount, nClassID);

				GetUntrainedClassMember(inputTrainData, nClassID, &nIndex);
				LoadInputArray(inputTrainData, nIndex, fInputArray);

				if (AssignInputConnections(perceptron, fInputArray, inputTrainData->nSize))
				{
					nCorrectCount = TuneInputPerceptron(perceptron, inputTrainData, fInputArray);
					nRemainingClassMembers = MarkCorrectPredictions(perceptron, inputTrainData, fInputArray);

					printf("%d\t%d             \r", nCorrectCount, nRemainingClassMembers);
					fflush(stdout);

					bCreateNewPerceptron = 1;
					++nPerceptronCount;
				}
				else
				{
					inputTrainData->data[nIndex].bTrained = 1;
					bCreateNewPerceptron = 0;
				}
			}

			ResetData(inputTrainData);

			printf("%d\t%d                             \n", nClassID, nPerceptronCount);
			fflush(stdout);
		}

	//	ResetData(inputTrainData);
	//	nPerceptronCount = MarkAllCorrectPredictions(layer->perceptronHead, inputTrainData, fInputArray, &nRight, &nWrong);
	//	printf("\n%d\t%d\t%d                             \n", nPerceptronCount, nRight, nWrong);

	//	DeleteData_InputData(&inputTrainingSubGroup);
	//}


	TestInputLayer(layer, inputTrainData, fInputArray);
	TestInputLayer(layer, inputValidateData, fInputArray);

	DumpLayerWeights(layer);

	while (1);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void TestInputLayer(structInputLayer *layer, structInput *inputData, float *fInputArray)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structInputPerceptron	*perceptronCur;
	float					fPercent=0.0f;
	float					fZeroPercent =0.0f;
	float					fSinglePercent =0.0f;
	float					fMultiplePercent =0.0f;
	int						nInputIndex;
	int						nResponseArray[10];
	int						nClassID;
	int						nSum;
	int						nRight=0;
	int						nWrong=0;
	int						nRightZeroResponse =0;
	int						nRightSingleResponse=0;
	int						nRightMultipleResponse=0;
	int						nWrongZeroResponse =0;
	int						nWrongSingleResponse=0;
	int						nWrongMultipleResponse=0;
	int						nMax =0;
	int						i;


	for (nInputIndex = 0; nInputIndex < inputData->nInputCount; ++nInputIndex)
	{
		nSum = 0;
		for (i = 0; i < 10; ++i)
			nResponseArray[i] = 0;

		LoadInputArray(inputData, nInputIndex, fInputArray);

		for (perceptronCur = layer->perceptronHead; perceptronCur != NULL; perceptronCur = perceptronCur->next)
		{
			ForwardPropagateInputPerceptron(perceptronCur);
			if (perceptronCur->fOutput >= perceptronCur->fLowerThreshold && perceptronCur->fOutput <= perceptronCur->fUpperThreshold)
			{
				++nResponseArray[perceptronCur->nClassID];
				nClassID = perceptronCur->nClassID;
			}
		}

		nSum = 0;
		nMax = 0;
		for (i = 0; i < 10; ++i)
		{
			if (nResponseArray[i] > 0)
			{
				if (nResponseArray[i] > nMax)
				{
					nMax = nResponseArray[i];
					nClassID = i;
				}

				++nSum;
			}
		}

		if (!nSum)
		{
			++nWrongZeroResponse;
			++nWrong;
		}
		else
		{
			if (nSum == 1)
			{
				if (nClassID == inputData->data[nInputIndex].nLabelID)
				{
					++nRightSingleResponse;
					++nRight;
				}
				else
				{
					++nWrongSingleResponse;
					++nWrong;
				}
			}
			else
			{
				if (nClassID == inputData->data[nInputIndex].nLabelID)
				{
					++nRightMultipleResponse;
					++nRight;
				}
				else
				{
					++nWrongMultipleResponse;
					++nWrong;
				}
			}
		}
	}

	if((nRightZeroResponse + nWrongZeroResponse) > 0)
		fZeroPercent = (float)nRightZeroResponse / (float)(nRightZeroResponse + nWrongZeroResponse);

	if((nRightSingleResponse + nWrongSingleResponse) > 0)
		fSinglePercent = (float)nRightSingleResponse / (float)(nRightSingleResponse + nWrongSingleResponse);
	
	if ((nRightMultipleResponse + nWrongMultipleResponse) > 0)
		fMultiplePercent = (float)nRightMultipleResponse / (float)(nRightMultipleResponse + nWrongMultipleResponse);

	if (inputData->nInputCount > 0)
		fPercent = (float)nRight / (float)inputData->nInputCount;


	printf("\n%f\t%f\t%f\t%d\t%d\n", fPercent, fSinglePercent, fMultiplePercent, (nRightSingleResponse + nWrongSingleResponse), (nRightMultipleResponse + nWrongMultipleResponse));
}



/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
int TuneInputPerceptron(structInputPerceptron *perceptron, structInput *inputData, float *fInputArray)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structInputSynapse	*synapse;
	int		nWrong = 0;
	int		nRight = 0;
	int		nPrevRight = 0;
	int		nPrevMultRight = 0;
	int		nInputIndex;
	int		nRightMax = 0;
	int		nMultiplierRight = 0;
	int		nClassID;
	int		i, j, k;

	float	fTarget = 0.0f;
	float	fUpperThreshold;
	float	fLowerThreshold;
	float	fThresholdPrev;

	float	fUpperMultiplier;
	float	fLowerMultiplier;
	float	fMultiplier;

	float	fUpperThresholdMax = 0.0f;
	float	fLowerThresholdMax = 0.0f;

	float	*fBestWeight;

	fBestWeight = (float *)calloc(perceptron->nSynapseCount, sizeof(float));

	perceptron->fThreshold = 1.0f;
	fThresholdPrev = 1.0f;

	fTarget = ForwardPropagateInputPerceptron(perceptron);

	fUpperMultiplier = 1.0f;
	fLowerMultiplier = 0.0f;
	fMultiplier = perceptron->synapseHead->fMultiplier;

	for (i = 0; i < 100; ++i)
	{
		fUpperThreshold = fTarget;
		fLowerThreshold = 0;

		for (j = 0; j < 100; ++j)
		{
			nWrong = 0;
			nRight = 0;
			perceptron->fUpperThreshold = (1.0f + perceptron->fThreshold) * fTarget;
			perceptron->fLowerThreshold = (1.0f - perceptron->fThreshold) * fTarget;

			for (nInputIndex = 0; nInputIndex < inputData->nInputCount; ++nInputIndex)
			{
				if (inputData->data[nInputIndex].bTrained)
					continue;

				nClassID = inputData->data[nInputIndex].nLabelID;
				LoadInputArray(inputData, nInputIndex, fInputArray);
				ForwardPropagateInputPerceptron(perceptron);

				if (perceptron->fOutput >= perceptron->fLowerThreshold && perceptron->fOutput <= perceptron->fUpperThreshold)
				{
					if (nClassID == perceptron->nClassID)
						++nRight;
					else
						++nWrong;
				}
			}

			if (nWrong == 0)
			{
				if (nRight > nRightMax)
				{
					nRightMax = nRight;

					fUpperThresholdMax = perceptron->fUpperThreshold;
					fLowerThresholdMax = perceptron->fLowerThreshold;

					for (synapse = perceptron->synapseHead, k = 0; synapse != NULL; synapse = synapse->next)
						fBestWeight[k++] = synapse->fWeight;
				}

				if (nRight == nPrevRight)
					break;

				nPrevRight = nRight;
			}

			if (nWrong > 0)
				fUpperThreshold = perceptron->fThreshold;
			else
				fLowerThreshold = perceptron->fThreshold;

			fThresholdPrev = perceptron->fThreshold;
			perceptron->fThreshold = (fUpperThreshold + fLowerThreshold) / 2.0f;
		}

		if (nWrong == 0)
		{
			if (nRight == nPrevMultRight)
				break;

			nPrevMultRight = nRight;
		}
		
		nPrevRight = 0;
		
		if (nRight < nMultiplierRight)
			fUpperMultiplier = fMultiplier;
		else
			fLowerMultiplier = fMultiplier;

		fMultiplier = (fUpperMultiplier + fLowerMultiplier) / 2.0f;

		if (fUpperMultiplier == fLowerMultiplier)
			break;


		perceptron->synapseHead->fMultiplier= fMultiplier;
		for (synapse = perceptron->synapseHead->next; synapse != NULL; synapse = synapse->next)
			synapse->fMultiplier = 1.0f - perceptron->synapseHead->fMultiplier;

		for (synapse = perceptron->synapseHead; synapse != NULL; synapse = synapse->next)
		{
			synapse->fWeight = synapse->fMultiplier / synapse->fSum;
		}
		
		nMultiplierRight = nRight;
	}


	perceptron->fUpperThreshold = fUpperThresholdMax;
	perceptron->fLowerThreshold = fLowerThresholdMax;

	for (synapse = perceptron->synapseHead, k = 0; synapse != NULL; synapse = synapse->next)
		synapse->fWeight = fBestWeight[k++];


	free(fBestWeight);

	return(nRightMax);
}


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
int MarkAllCorrectPredictions(structInputPerceptron *perceptronHead, structInput *inputData, float *fInputArray, int *nRight, int *nWrong)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structInputPerceptron	*perceptronCur;
	int		nInputIndex;
	int		nClassID;
	int		nCount = 0;

	(*nRight) = 0;
	(*nWrong) = 0;

	
	for (perceptronCur= perceptronHead; perceptronCur!=NULL; perceptronCur= perceptronCur->next)
	{
		++nCount;

		for (nInputIndex = 0; nInputIndex < inputData->nInputCount; ++nInputIndex)
		{
			if (inputData->data[nInputIndex].bTrained)
				continue;

			nClassID = inputData->data[nInputIndex].nLabelID;
			LoadInputArray(inputData, nInputIndex, fInputArray);
			ForwardPropagateInputPerceptron(perceptronCur);

			if (perceptronCur->fOutput >= perceptronCur->fLowerThreshold && perceptronCur->fOutput <= perceptronCur->fUpperThreshold)
			{
				if (nClassID == perceptronCur->nClassID)
				{
					inputData->data[nInputIndex].bTrained = 1;
					++(*nRight);
				}
				else
				{
					++(*nWrong);
				}
			}
		}
	}
	

	return(nCount);
}


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
int MarkCorrectPredictions(structInputPerceptron *perceptron, structInput *inputData, float *fInputArray)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	int		nWrong = 0;
	int		nRight = 0;
	int		nInputIndex;
	int		nClassID;
	int		nRemainingClassMembers=0;

	for (nInputIndex = 0; nInputIndex < inputData->nInputCount; ++nInputIndex)
	{
		if (inputData->data[nInputIndex].bTrained)
			continue;

		nClassID = inputData->data[nInputIndex].nLabelID;
		LoadInputArray(inputData, nInputIndex, fInputArray);
		ForwardPropagateInputPerceptron(perceptron);

		if (perceptron->fOutput >= perceptron->fLowerThreshold && perceptron->fOutput <= perceptron->fUpperThreshold)
		{
			if (nClassID == perceptron->nClassID)
			{
				inputData->data[nInputIndex].bTrained = 1;
				++nRight;
			}
			else
			{
				++nWrong;
			}
		}

		if (nClassID == perceptron->nClassID && !inputData->data[nInputIndex].bTrained)
			++nRemainingClassMembers;
	}

	return(nRemainingClassMembers);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
int AssignInputConnections(structInputPerceptron *perceptron, float *fInputArray, int nArrayCount)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structInputSynapse	*synapse;
	structData			*dataJenkFish = NULL;
	structData			*dataCluster = NULL;
	int					nFirst;
	int					nInputIndex;
	int					nClusterCount=2;
	int					nSum=0;
	int					i, j;

	dataJenkFish = (structData *)calloc(nArrayCount, sizeof(structData));

	for (i = 0; i < nArrayCount; ++i)
	{
		dataJenkFish[i].nID = i;
		dataJenkFish[i].fOutput = fInputArray[i];
	}

	
	JenkFish(dataJenkFish, &dataCluster, nClusterCount, nArrayCount);
	free(dataJenkFish);

	perceptron->nSynapseCount = nClusterCount;

	for (i=0; i< nClusterCount; ++i)
	{
		synapse = (structInputSynapse *)calloc(1, sizeof(structInputSynapse));
		AddInputSynapse(&perceptron->synapseHead, synapse);

		synapse->nInputCount = dataCluster[i].nCount - 1;
		synapse->fInputArray = (float **)calloc(synapse->nInputCount, sizeof(float *));
		
		if (!i)
			synapse->fMultiplier = 0.85f;
		else
			synapse->fMultiplier = 0.15f;

		synapse->fSum = 0.0f;
		for (j = 0, nInputIndex = 0, nFirst = 1; j < nArrayCount; ++j)
		{
			if (fInputArray[j] >= dataCluster[i].fStart && fInputArray[j] <= dataCluster[i].fEnd)
			{
				if (nFirst)
				{
					synapse->fInput = &fInputArray[j];
					nFirst = 0;
				}
				else
				{
					synapse->fInputArray[nInputIndex++] = &fInputArray[j];
				}

				synapse->fSum += fInputArray[j];
			}
		}

		synapse->fWeight = synapse->fMultiplier / synapse->fSum;
	}

	free(dataCluster);

	return(1);
}


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
float ForwardPropagateInputPerceptron(structInputPerceptron *perceptron)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structInputSynapse	*synapse;
	float				fSum = 0.0f;
	float				fClusterSum;
	int					k;

	for (synapse= perceptron->synapseHead; synapse!=NULL; synapse = synapse->next)
	{
		if (synapse->nInputCount > 0)
		{
			fClusterSum = *synapse->fInput;

			for (k = 0; k < synapse->nInputCount; ++k)
				fClusterSum += *(synapse->fInputArray)[k];

			fSum += synapse->fWeight * fClusterSum;
		}
		else
		{
			fSum += synapse->fWeight * *synapse->fInput;
		}

	}

	perceptron->fOutput = MTanH(fSum);

	return(perceptron->fOutput);
}





/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
structInputLayer *NewInputLayer(structCLN *cln)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structInputLayer			*layerNew;

	if ((layerNew = (structInputLayer *)calloc(1, sizeof(structInputLayer))) == NULL)
		exit(0);

	layerNew->nID = 0;

	return(layerNew);
}



/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
structInputPerceptron *NewInputPerceptron(structInputLayer *layer, int *nPerceptronID, int *nSynapseID, int nSynapseCount, int nClassID)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structInputPerceptron	*perceptronNew;

	perceptronNew = (structInputPerceptron *)calloc(1, sizeof(structInputPerceptron));
	perceptronNew->nID = (*nPerceptronID)++;
	perceptronNew->nClassID = nClassID;

	++layer->nPerceptronCount;
	AddInputPerceptron(&layer->perceptronHead, perceptronNew);

	return(perceptronNew);
}


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void AddInputPerceptron(structInputPerceptron **head, structInputPerceptron *newPerceptron)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structInputPerceptron	*cur = *head;
	int						nID = 0;

	if (*head == NULL)
	{
		*head = newPerceptron;
	}
	else
	{
		while (cur->next != NULL)
		{
			cur = cur->next;

			if (newPerceptron->nID < 1)
				nID = cur->nID;
		}

		if (newPerceptron->nID < 1)
			newPerceptron->nID = ++nID;

		cur->next = newPerceptron;
	}

	return;
}


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void AddInputSynapse(structInputSynapse **head, structInputSynapse *newSynapse)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structInputSynapse	*cur = *head;
	int					nID = 0;

	if (*head == NULL)
	{
		*head = newSynapse;
	}
	else
	{
		while (cur->next != NULL)
		{
			cur = cur->next;

			if (newSynapse->nID < 1)
				nID = cur->nID;
		}

		if (newSynapse->nID < 1)
			newSynapse->nID = ++nID;

		cur->next = newSynapse;
	}

	return;
}



/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void LoadInputArray(structInput *inputData, int nIndex, float *fInputArray)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	int		i;
	
	for (i = 0; i < inputData->nSize; ++i)
		fInputArray[i] = inputData->data[nIndex].fIntensity[i];

	return;
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void GetUntrainedClassMember(structInput *inputData, int nClassID, int *nIndex)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	int	i;

	for (i = *nIndex + 1; i < inputData->nInputCount; ++i)
	{
		if (inputData->data[i].nLabelID == nClassID && !inputData->data[i].bTrained)
		{
			*nIndex = i;
			break;
		}
	}

	return;
}


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void GetUntrainedClassMemberFromCluster(structInput *inputData, int nClassID, int *nIndex, int nCluster)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	int	i;

	for (i = *nIndex + 1; i < inputData->nInputCount; ++i)
	{
		if (inputData->data[i].nLabelID == nClassID && !inputData->data[i].bTrained && inputData->data[i].nCluster == nCluster)
		{
			*nIndex = i;
			break;
		}
	}

	return;
}



/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void ResetData(structInput *inputData)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	int	i;

	for (i = 0; i < inputData->nInputCount; ++i)
		inputData->data[i].bTrained = 0;
}



/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void DumpLayerWeights(structInputLayer *layer)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structInputPerceptron	*perceptronCur;
	structInputSynapse		*synapseCur;

	FILE				*pFile = NULL;

	if ((pFile = fopen("weights.txt", "wb")) == NULL)
	{
		exit(0);
	}

	for (perceptronCur = layer->perceptronHead; perceptronCur != NULL; perceptronCur = perceptronCur->next)
	{
		for (synapseCur= perceptronCur->synapseHead; synapseCur!=NULL; synapseCur= synapseCur->next)
		{
			fprintf(pFile, "%f\n", synapseCur->fWeight);
		}
	}

	fclose(pFile);
}


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
int AssignInputError(structInput *inputData, int nTarget)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	float	*fInputArray;
	float	fA, fB;
	float	fTemp;
	float	fMinError;
	int		nCount;
	int		nInputID=-1;
	int		i, j;
	
	fInputArray = (float *)calloc(inputData->nSize, sizeof(float));

	for (i = 0, nCount=0; i < inputData->nInputCount; ++i)
	{
		if (inputData->data[i].nLabelID == nTarget && !inputData->data[i].bTrained)
		{
			for (j = 0; j < inputData->nSize; ++j)
				fInputArray[j] += inputData->data[i].fIntensity[j];
			++nCount;
		}
	}

	if (nCount > 0)
	{
		for (j = 0; j < inputData->nSize; ++j)
			fInputArray[j] /= (float)nCount;

		fMinError = 9999.9f;
		for (i = 0; i < inputData->nInputCount; ++i)
		{
			if (inputData->data[i].nLabelID == nTarget && !inputData->data[i].bTrained)
			{
				fTemp = 0.0f;
				for (j = 0; j < inputData->nSize; ++j)
				{
					fA = fInputArray[j];
					fB = inputData->data[i].fIntensity[j];

					fTemp += ((fA - fB) * (fA - fB));
				}

				if (fTemp != 0.0f)
				{
					inputData->data[i].fError = (1.0f / (inputData->nSize*inputData->nSize))*fTemp;

					if (inputData->data[i].fError <= fMinError)
					{
						nInputID = i;
						fMinError = inputData->data[i].fError;
					}
				}
			}
		}
	}

	free(fInputArray);

	return(nInputID);
}


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
int GroupInputError(structInput *inputData, int nTarget, int nOutputCount)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structData			*dataJenkFish = NULL;
	structData			*dataCluster = NULL;
	int					nMemberCount;
	int					nClusterCount;
	int					nCluster;
	int					nSum;
	int					i;

	for (i = 0, nMemberCount = 0; i < inputData->nInputCount; ++i)
	{
		if (inputData->data[i].nLabelID == nTarget && !inputData->data[i].bTrained)
		{
			++nMemberCount;
		}
	}

	dataJenkFish = (structData *)calloc(nMemberCount, sizeof(structData));

	while (1)
	{
		nSum = 0;
		for (i = 0, nMemberCount = 0; i < inputData->nInputCount; ++i)
		{
			if (inputData->data[i].nLabelID == nTarget && !inputData->data[i].bTrained)
			{
				dataJenkFish[nMemberCount].nID = nMemberCount;
				dataJenkFish[nMemberCount++].fOutput = inputData->data[i].fError;
			}
		}

		nClusterCount = nMemberCount / nOutputCount;

		JenkFish(dataJenkFish, &dataCluster, nClusterCount, nMemberCount);

		for (i=0; i< nClusterCount; ++i)
		{
			if (dataCluster[i].nCount == 0)
			{
				printf("Error\n");
				break;
			}
			else
				nSum += dataCluster[i].nCount;
		}

		if (nSum == nMemberCount)
		{
			break;
		}
		else
		{
			--nOutputCount;
			free(dataCluster);
		}
	}

	
	
	
	
	free(dataJenkFish);


	for (nCluster=0; nCluster< nClusterCount; ++nCluster)
	{
		for (i = 0, nMemberCount = 0; i < inputData->nInputCount; ++i)
		{
			if (inputData->data[i].nLabelID == nTarget && !inputData->data[i].bTrained)
			{
				if (inputData->data[i].fError >= dataCluster[nCluster].fStart && inputData->data[i].fError <= dataCluster[nCluster].fEnd)
				{
					inputData->data[i].nCluster = nCluster;
				}
			}
		}
	}



	return(nClusterCount);
}



void CreateTrainingDataSubGroup(structInput *inputData, structInput **inputSubGroup, int nTargetCount, int nClusterCount, int nClassCount)
{
	int	nIndex;
	int	nImageIndex;
	int	nInputCount = 0;
	int	nCluster = 12;
	int	nClassID = 12;
	int	j;

	while (nInputCount < nTargetCount)
	{
		for (nCluster = 0; nCluster < nClusterCount && nInputCount < nTargetCount; ++nCluster)
		{
			for (nClassID = 0; nClassID < nClassCount && nInputCount < nTargetCount; ++nClassID)
			{
				for (nIndex = 0; nIndex < inputData->nInputCount; ++nIndex)
				{
					if (!inputData->data[nIndex].bTrained && inputData->data[nIndex].nLabelID == nClassID && inputData->data[nIndex].nCluster == nCluster)
					{
						inputData->data[nIndex].bTrained = 2;
						++nInputCount;
						break;
					}
				}
			}
		}
	}

	for (nIndex = 0; nIndex < inputData->nInputCount; ++nIndex)
	{
		if (inputData->data[nIndex].bTrained == 2)
			inputData->data[nIndex].bTrained = 0;
	}

	*inputSubGroup = (structInput *)calloc(1, sizeof(structInput));



	(*inputSubGroup)->nDataSource = inputData->nDataSource;
	(*inputSubGroup)->nInputCount = nInputCount;
	(*inputSubGroup)->nChannels = inputData->nChannels;
	(*inputSubGroup)->nRowCount = inputData->nRowCount;
	(*inputSubGroup)->nColumnCount = inputData->nColumnCount;
	(*inputSubGroup)->nSize = inputData->nSize;

	if (((*inputSubGroup)->data = (structInputData *)calloc((*inputSubGroup)->nInputCount, sizeof(structInputData))) == NULL)
		exit(0);

	nImageIndex = 0;
	nIndex = 0;
	while (nIndex < nTargetCount)
	{
		for (nCluster = 0; nCluster < nClusterCount && nIndex < nTargetCount; ++nCluster)
		{
			for (nClassID = 0; nClassID < nClassCount && nIndex < nTargetCount; ++nClassID)
			{
				for (nImageIndex = 0; nImageIndex < inputData->nInputCount; ++nImageIndex)
				{
					if (!inputData->data[nImageIndex].bTrained && inputData->data[nImageIndex].nLabelID == nClassID && inputData->data[nImageIndex].nCluster == nCluster)
					{
						(*inputSubGroup)->data[nIndex].fIntensity = (float *)calloc((*inputSubGroup)->nSize, sizeof(float));

						(*inputSubGroup)->data[nIndex].nID = nIndex;
						(*inputSubGroup)->data[nIndex].nLabelID = inputData->data[nImageIndex].nLabelID;
						(*inputSubGroup)->data[nIndex].nGroupA = inputData->data[nImageIndex].nGroupA;
						(*inputSubGroup)->data[nIndex].nGroupB = inputData->data[nImageIndex].nGroupB;
						strcpy((*inputSubGroup)->data[nIndex].sLabel, inputData->data[nImageIndex].sLabel);

						for (j = 0; j < inputData->nSize; ++j)
						{
							(*inputSubGroup)->data[nIndex].fIntensity[j] = inputData->data[nImageIndex].fIntensity[j];
						}

						inputData->data[nImageIndex].bTrained = 2;
						++nIndex;
						break;
					}
				}
			}
		}
	}

	for (nIndex = 0; nIndex < inputData->nInputCount; ++nIndex)
	{
		if (inputData->data[nIndex].bTrained == 2)
			inputData->data[nIndex].bTrained = 0;
	}
}